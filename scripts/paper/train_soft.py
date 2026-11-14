"""SFT and RL for the paper runs, on long prompts an 18 GB Mac can hold. One loop, two losses.

  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/train_soft.py sft --stage week --step sft --arm read_eps --calls 400 --passes 2 [--block rag]
  PYTHONPATH=src:scripts .venv/bin/python -W ignore scripts/paper/train_soft.py rl  --stage week --step rl --arm business --start <adapter> --calls 250 [--block rag]

Stock mlx training runs out of memory from about 2,200 tokens. Here the frozen layers run once in the fast
path, only the top 8 run with gradients, each recomputed in the backward pass, and output scores are taken
only at the answer's value tokens. Values are scored over their allowed list, as read.py reads them.

SFT: cross-entropy on read lines and Next EPS, soft cross-entropy on the action.
RL: per call, 4 rollouts sample the read at temperature 0.8, and each is scored exactly. With so few choices
per line, the best policy under a KL limit has a closed form: the starting policy tilted by reward over beta.
Training moves the model toward that target by cross-entropy, on the call's best rollout. A plain policy
gradient collapsed to always AVOID in the number-rule control: its gradient vanishes there, with nothing to
pull it back. Checkpoints hold adapter, optimizer, position and random state, so a restart continues at the
same example.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import blocks  # noqa: E402
import kb  # noqa: E402
import prompt as P  # noqa: E402
from common import DATA, ROOT, RUNS, Status, write_atomic  # noqa: E402
from read import MODEL, masked_text  # noqa: E402

TOP, RANK, SCALE = 8, 16, 20.0
ACCUM, SAVE_SFT, SAVE_RL, ROLLOUTS, TEMP, TOP_P, KL = 4, int(os.environ.get("PAPER_SAVE", 50)), int(os.environ.get("PAPER_SAVE", 25)), 4, .8, .95, .05
SFT_LR, RL_LR = (1e-4, 1e-5), (2e-5, 2e-6)
ROUND_TRIP, SIDE, BETA, EPS_GAIN = .008, .004, 1.0, 2.0
ARM_LINES = {"score": "action", "hard": "action", "read": "read", "read_eps": "all", "placebo": "all"}


def labels():
    return pd.DataFrame(map(json.loads, Path(os.environ.get("PAPER_LABELS", DATA / "teacher" / "labels_v2.jsonl")).open()))


def targets(arm, rows):
    """One row per call: read lines, next EPS, and the soft action target."""
    s = rows.copy()
    o = pd.read_parquet(DATA / "outcomes.parquet", columns=["row", "y", "next_yoy"])
    s = s.merge(o, on="row", how="left")
    lab = labels()
    if arm != "hard":
        g = lab.groupby("row")
        first = g.first()
        view = "score" if "score" in lab else "rating"
        s = s.merge(g[view].mean().rename("view").reset_index(), on="row").merge(first[[n.lower() for n, _ in P.LINES[:5]]].reset_index(), on="row")
        s["p_pos"] = rank_target(s, "view")
    else:
        s["p_pos"] = (rank_target(s, "y") > .5).astype(float)
    if arm == "placebo":
        rng = np.random.default_rng(0)
        cols = ["p_pos"] + [n.lower() for n, _ in P.LINES[:5]]
        for _, i in s.groupby("month").groups.items():
            s.loc[i, cols] = s.loc[rng.permutation(i), cols].values
    s["eps"] = s.next_yoy.map(lambda x: None if pd.isna(x) else "up" if x > 0 else "down")
    s["position"] = np.where(s.row % 2 == 0, "flat", "long")
    return s


def rank_target(s, col, least=10):
    """Rank within the month, or within the quarter where a month has fewer than `least` calls, so small
    sets do not collapse to ties at 0.5. Returns a probability between 0 and 1."""
    when = pd.to_datetime(s.date)
    small = s.groupby("month")[col].transform("size") < least
    group = np.where(small, when.dt.to_period("Q").astype(str), s.month)
    r = s.groupby(group)[col].rank(pct=True)
    n = s.groupby(group)[col].transform("size")
    return r - .5 / n


class Net:
    """Trunk once, top with gradients, scores only where asked."""

    def __init__(self, adapter=None):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        from mlx_lm.tuner.utils import linear_to_lora_layers
        import tagent.train_lora  # noqa: F401  chunked gated-delta path for the layers with gradients
        self.mx, self.nn, self.masks = mx, nn, (create_attention_mask, create_ssm_mask)
        self.model, self.tok = load(MODEL)
        self.model.freeze()
        mx.random.seed(0)  # every arm starts from the same weights, and resume tests can compare runs
        linear_to_lora_layers(self.model, TOP, {"rank": RANK, "scale": SCALE, "dropout": 0.0})
        if adapter:
            self.model.load_weights(str(Path(adapter) / "adapters.safetensors"), strict=False)
        self.lm = self.model.language_model
        self.tm = self.lm.model
        self.layers = self.tm.layers
        self.cut = len(self.layers) - TOP
        self.mode(False)

    def mode(self, training):
        self.model.eval()
        if training:
            for x in self.layers[self.cut:]:
                x.train()

    def trunk(self, ids):
        mx = self.mx
        h = self.tm.embed_tokens(mx.array([ids]))
        fa, ssm = self.masks[0](h, None), self.masks[1](h, None)
        for x in self.layers[:self.cut]:
            h = x(h, mask=ssm if x.is_linear else fa, cache=None)
        mx.eval(h)
        return mx.stop_gradient(h), fa, ssm

    def top(self, h, fa, ssm, checkpoint=True):
        mx = self.mx
        for x in self.layers[self.cut:]:
            if checkpoint:
                call = type(x).__call__

                def inner(params, h, layer=x, mask=ssm if x.is_linear else fa, call=call):
                    layer.update(params)
                    return call(layer, h, mask=mask, cache=None)
                h = mx.checkpoint(inner)(x.trainable_parameters(), h)
            else:
                h = x(h, mask=ssm if x.is_linear else fa, cache=None)
        return h

    def scores(self, h, where):
        h = self.tm.norm(h[:, where])
        out = self.tm.embed_tokens.as_linear(h) if self.lm.args.tie_word_embeddings else self.lm.lm_head(h)
        return out[0].astype(self.mx.float32)

    def save(self, folder, opt, state):
        mx = self.mx
        from mlx.utils import tree_flatten
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / ".adapters.safetensors"
        mx.save_safetensors(str(tmp), dict(tree_flatten(self.model.trainable_parameters())))
        tmp.replace(folder / "adapters.safetensors")
        tmp = folder / ".optimizer.safetensors"
        mx.save_safetensors(str(tmp), dict(tree_flatten(opt.state)))
        tmp.replace(folder / "optimizer.safetensors")
        write_atomic(folder / "adapter_config.json", json.dumps({"fine_tune_type": "lora", "num_layers": TOP, "lora_parameters": {"rank": RANK, "scale": SCALE, "dropout": 0.0}}))
        write_atomic(folder / "state.json", json.dumps(state))

    def resume(self, folder, opt):
        from mlx.utils import tree_unflatten
        f = folder / "state.json"
        if not f.exists():
            return None
        self.model.load_weights(str(folder / "adapters.safetensors"), strict=False)
        opt.state = tree_unflatten(list(self.mx.load(str(folder / "optimizer.safetensors")).items()))
        return json.loads(f.read_text())


def sequence(tok, opts, prompt_ids, read):
    """Prompt plus answer tokens. Returns ids and, per line, the position predicting its value."""
    ids, where = list(prompt_ids), []
    for name, vals, after in opts:
        where.append(len(ids) - 1)
        ids += vals[read[name]] + after
    return ids, where


def allowed(mx, opts):
    return [mx.array([vals[v][0] for v in vals]) for _, vals, _ in opts]


def schedule(lr, updates, warmup):
    """Warmup matters: Adam's first steps move every weight by the full rate, which swung the policy at once."""
    import mlx.optimizers as optim
    warmup = min(warmup, max(updates // 4, 1))
    return optim.join_schedules([optim.linear_schedule(lr[0] / 100, lr[0], warmup), optim.cosine_decay(lr[0], max(updates - warmup, 1), lr[1])], [warmup])


def prompts(db, tok, r, a, lines):
    text = "" if a.no_text else masked_text(db, r.row)
    block = (blocks.KINDS.get(a.block) or blocks.saved(ROOT / a.block))(db, r, text or masked_text(db, r.row), tok)
    return P.rendered(tok, P.messages(r, text, block, r.position, "plain"), lines)


def add(mx, total, grads):
    from mlx.utils import tree_map
    return grads if total is None else tree_map(lambda x, y: x + y, total, grads)


def sft(a):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_map
    net = Net(a.start)
    tok, nn, lines = net.tok, net.nn, ARM_LINES[a.arm]
    sets = pd.read_parquet(DATA / "sets.parquet")
    s = targets(a.arm, sets[sets.set == "sft"].sort_values("rank").iloc[a.skip:a.skip + a.calls])
    order = [i for p in range(a.passes) for i in np.random.default_rng(p).permutation(len(s))]
    updates = len(order) // ACCUM
    opt = optim.Adam(learning_rate=schedule(a.lr or SFT_LR, updates, 10))
    name = a.name or a.arm
    folder = ROOT / "adapters" / "paper" / a.stage / name
    log = RUNS / a.stage / "train" / f"{name}.csv"
    state = net.resume(folder, opt) or {"at": 0}
    if not log.exists():
        write_atomic(log, "update,examples,loss_read,loss_eps,loss_action,seconds\n")
    status = Status(a.stage, a.step)
    db = kb.connect()

    def loss(_m, h, fa, ssm, where, picks, soft, spots):
        lg = net.scores(net.top(h, fa, ssm), mx.array(where))
        parts = []
        for i, sel in enumerate(spots):
            lp = lg[i][sel] - mx.logsumexp(lg[i][sel])
            parts.append(-(soft[i] * lp).sum())
        return sum(parts), mx.stack(parts)

    vg = nn.value_and_grad(net.model, loss)
    grads, sums, start = None, [0, 0, 0, 0], time.time()
    net.mode(True)
    for k in range(state["at"], len(order)):
        r = next(s.iloc[[order[k]]].itertuples())
        opts = P.options(tok, r.position, lines)
        spots = allowed(mx, opts)
        read = {n: getattr(r, n.lower().replace(" ", "_"), None) for n, _, _ in opts}
        known = isinstance(r.eps, str)
        read["Next EPS"] = r.eps if known else "up"
        read["Action"] = P.ACTIONS[r.position][0 if r.p_pos >= .5 else 1]
        ids, where = sequence(tok, opts, prompts(db, tok, r, a, lines), read)
        soft = []
        for n, vals, _ in opts:
            v = list(vals)
            if n == "Action":
                soft.append(mx.array([r.p_pos, 1 - r.p_pos]))
            elif n == "Next EPS" and not known:
                soft.append(mx.zeros(len(v)))
            else:
                soft.append(mx.array([float(x == read[n]) for x in v]))
        h, fa, ssm = net.trunk(ids)
        (_, parts), g = vg(net.model, h, fa, ssm, where, None, soft, spots)
        grads = add(mx, grads, g)
        mx.eval(grads, parts)
        names = [n for n, _, _ in opts]
        got = dict(zip(names, parts.tolist()))
        sums[0] += sum(v for n, v in got.items() if n not in ("Next EPS", "Action"))
        sums[1] += got.get("Next EPS", 0)
        sums[2] += got["Action"]
        sums[3] += 1
        if (k + 1) % ACCUM == 0:
            opt.update(net.model, tree_map(lambda x: x / ACCUM, grads))
            mx.eval(net.model.trainable_parameters(), opt.state)
            grads = None
            u = (k + 1) // ACCUM
            with log.open("a") as f:
                f.write(f"{u},{k + 1},{sums[0] / sums[3]:.4f},{sums[1] / sums[3]:.4f},{sums[2] / sums[3]:.4f},{time.time() - start:.1f}\n")
            sums, start = [0, 0, 0, 0], time.time()
            status.save()
            if u % SAVE_SFT == 0 or k + 1 == len(order):
                net.save(folder, opt, {"at": k + 1})
        mx.clear_cache()
    status.save()
    return folder


def rewards(s, a):
    """Trade reward per position in clean-outcome units; the month's mean is removed and tails clipped."""
    s = s.copy()
    if a.arm == "shuffled":
        rng = np.random.default_rng(1)
        for _, i in s.groupby("month").groups.items():
            s.loc[i, ["y", "eps"]] = s.loc[rng.permutation(i), ["y", "eps"]].values
    s["z"] = (s.y - s.groupby("month").y.transform("mean")).clip(-3, 3)
    unit = s.risk * math.sqrt(60)
    s["r_pos"] = np.where(s.position == "flat", s.z - ROUND_TRIP / unit, s.z)
    s["r_neg"] = np.where(s.position == "flat", 0, -SIDE / unit)
    s[["r_pos", "r_neg"]] = s[["r_pos", "r_neg"]].fillna(0.)  # a call with no price outcome gives no trade reward; its NaN poisoned the week run's both arm
    if a.control:  # number-rule control: reward follows a fact on the page
        s["r_pos"], s["r_neg"] = np.where(s.past > 0, 3., -3.), 0.
    return s


def sample(rng):
    def choose(name, vals, p):
        q = np.power(p, 1 / TEMP)
        q = q / q.sum()
        i = np.argsort(-q)
        keep = i[:np.searchsorted(np.cumsum(q[i]), TOP_P) + 1]
        w = q[keep] / q[keep].sum()
        return vals[rng.choice(keep, p=w)]
    return choose


def rl(a):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from mlx_lm.models.cache import make_prompt_cache
    from read import step, walk
    net = Net(a.start)
    tok, nn = net.tok, net.nn
    sets = pd.read_parquet(DATA / "sets.parquet")
    s = sets[sets.set == "rl"].sort_values("rank").head(a.calls).merge(pd.read_parquet(DATA / "outcomes.parquet", columns=["row", "y", "next_yoy"]), on="row", how="left")
    s["eps"] = s.next_yoy.map(lambda x: None if pd.isna(x) else "up" if x > 0 else "down")
    s["position"] = np.where(s.row % 2 == 0, "flat", "long")
    s = rewards(s, a)
    updates = -(-len(s) // ACCUM)
    opt = optim.Adam(learning_rate=schedule(RL_LR, updates, 20))
    name = a.name or a.arm
    folder = ROOT / "adapters" / "paper" / a.stage / name
    log = RUNS / a.stage / "train" / f"{name}.csv"
    ref = dict(tree_flatten(net.model.trainable_parameters()))
    state = net.resume(folder, opt) or {"at": 0, "rng": 0}
    rng = np.random.default_rng(state["rng"] + state["at"])
    if not log.exists():
        write_atomic(log, "update,calls,reward,advantage_gap,kl,p_pos,seconds\n")
    status = Status(a.stage, a.step)
    db = kb.connect()

    def reward(r, probs):
        pos, neg = P.ACTIONS[r.position]
        trade = probs["Action"][pos] * r.r_pos + probs["Action"][neg] * r.r_neg
        business = 0.0 if not isinstance(r.eps, str) else math.log(max(probs["Next EPS"][r.eps], 1e-6)) + math.log(2)
        return {"business": business, "trade": trade, "both": business / .7 + trade, "shuffled": trade}[a.arm]

    def loss(_m, h, fa, ssm, where, spots, targets_):
        lg = net.scores(net.top(h, fa, ssm), mx.array(where))
        total, kl = mx.array(0.), mx.array(0.)
        for i, sel in enumerate(spots):
            lp = lg[i][sel] - mx.logsumexp(lg[i][sel])
            total = total - (targets_[i] * lp).sum()
            kl = kl + (mx.exp(lp) * (lp - refs[0][i])).sum()
        return total, kl

    def tilt(ref_lp, gain):
        """The best policy under a KL limit, exactly: the starting policy tilted by reward over beta."""
        return mx.softmax(ref_lp + mx.array(gain) / BETA)

    vg = nn.value_and_grad(net.model, loss)
    grads, seen, tally, start, refs = None, 0, [0, 0, 0, 0], time.time(), [None]
    for k in range(state["at"], len(s)):
        r = next(s.iloc[[k]].itertuples())
        opts = P.options(tok, r.position)
        spots = allowed(mx, opts)
        names = [n for n, _, _ in opts]
        ids = prompts(db, tok, r, a, "all")
        net.mode(False)
        cache = make_prompt_cache(net.model)
        logits, _ = step(mx, net.model, ids, cache)
        saved = [c.state for c in cache]
        rolls = []
        for _ in range(ROLLOUTS):
            c = make_prompt_cache(net.model)
            for x, y in zip(c, saved):
                x.state = y
            read, probs, _ = walk(mx, net.model, opts, logits, c, sample(rng))
            rolls.append((read, probs, reward(r, probs)))
        got = np.array([x[2] for x in rolls])
        adv = (got - got.mean()) / (got.std() + 1e-4)
        del cache, saved, c
        net.mode(True)
        j = int(got.argmax())
        read, probs, _ = rolls[j]
        full, where = sequence(tok, opts, ids, read)
        h, fa, ssm = net.trunk(full)
        live = dict(tree_flatten(net.model.trainable_parameters()))
        net.model.update(tree_unflatten(list(ref.items())))
        base = net.scores(net.top(h, fa, ssm, checkpoint=False), mx.array(where))
        ref_lp = [base[i][sel] - mx.logsumexp(base[i][sel]) for i, sel in enumerate(spots)]
        mx.eval(ref_lp)
        net.model.update(tree_unflatten(list(live.items())))
        refs[0] = ref_lp
        goal = []
        for i, (n, vals, _) in enumerate(opts):
            v = list(vals)
            if n == "Action":
                goal.append(tilt(ref_lp[i], [r.r_pos, r.r_neg] if a.arm != "business" else [0., 0.]))
            elif n == "Next EPS":
                known = isinstance(r.eps, str) and a.arm in ("business", "both")
                goal.append(tilt(ref_lp[i], [float(x == r.eps) * EPS_GAIN for x in v] if known else [0.] * len(v)))
            else:  # read lines lean toward the call's best rollout, by how far ahead it was
                goal.append(tilt(ref_lp[i], [float(x == read[n]) * float(adv[j]) for x in v]))
        (_, kl), g = vg(net.model, h, fa, ssm, where, spots, goal)
        grads = add(mx, grads, g)
        mx.eval(grads, kl)
        seen += 1
        tally[2] += float(kl)
        tally[0] += float(got.mean())
        tally[1] += float(got.max() - got.min())
        tally[3] += float(np.mean([x[1]["Action"][P.ACTIONS[r.position][0]] for x in rolls]))
        if (k + 1) % ACCUM == 0 or k + 1 == len(s):
            opt.update(net.model, tree_map(lambda x: x / max(seen, 1), grads))
            mx.eval(net.model.trainable_parameters(), opt.state)
            n = (k % ACCUM) + 1
            u = k // ACCUM + 1
            with log.open("a") as f:
                f.write(f"{u},{k + 1},{tally[0] / n:.4f},{tally[1] / n:.4f},{tally[2] / max(seen, 1):.4f},{tally[3] / n:.4f},{time.time() - start:.1f}\n")
            grads, seen, tally, start = None, 0, [0, 0, 0, 0], time.time()
            status.save()
            if u % SAVE_RL == 0 or k + 1 == len(s):
                net.save(folder, opt, {"at": k + 1, "rng": state["rng"]})
        mx.clear_cache()
    status.save()
    return folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["sft", "rl"])
    for k in ("stage", "step", "arm"):
        p.add_argument(f"--{k}", required=True)
    p.add_argument("--name", help="folder and log name when an arm runs more than once")
    p.add_argument("--skip", type=int, default=0, help="calls to skip first, for growing a run on new calls")
    p.add_argument("--calls", type=int, required=True)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--block", default="rag", help="none, rag, random, shuffled, or a pi log path")
    p.add_argument("--start")
    p.add_argument("--lr", type=lambda s: tuple(float(x) for x in s.split(",")),
                   help="peak,floor for SFT; growing a trained adapter at the full 1e-4 peak diverged on the week run")
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--control", action="store_true")
    a = p.parse_args()
    print({"sft": sft, "rl": rl}[a.mode](a), flush=True)


if __name__ == "__main__":
    main()
