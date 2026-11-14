# It Sounds Like the Teacher but Doesn't Trade Like It

*Nityanand Rai · November 13, 2026*

*I spent a week trying to teach a small AI model on my laptop to read company earnings calls the way a large model does, and to pick stocks from them. It learned to talk like the large model. It didn't learn to pick better stocks. This post shows exactly what the model saw, what it wrote, how it was trained, and how every number was calculated, with links to the real files.*

## The short version

- I tested six ways of using a small model (Qwen3.5-9B on a MacBook) to read earnings calls:
    - with nothing else;
    - with reference documents;
    - with its own searches;
    - trained on Claude Opus's answers;
    - then trained with rewards;
    - plus a version that never saw the call.
- On 800 calls it had never seen, every version ranked stocks a little better than chance, at a score of about 0.10. None beat the others by more than luck.
- Training on Opus's answers made the small model agree with Opus more often (66% → 71%). Much of that came from copying one habit: calling management "direct". It didn't improve stock picking.
- The proof is in the controls. A model trained on *scrambled* Opus answers ranked stocks as well as the real one.
- The model's decisions do follow its stated reasons, so the weak link is the quality of its reading.

---

## 1. The question

Large models like Claude Opus read an earnings call well, but running them on thousands of calls costs money. Small open models run free on a laptop, but they read less well. **Can a small model learn to read calls well enough to pick stocks, by copying a large one?**

I also wanted to test, one at a time, the tricks people usually add: reference documents, letting the model search, fine-tuning and reinforcement learning. Each one gets a deliberately broken version alongside it (a **control**), so a lucky result can't pass as a real one.

## 2. The data

**Calls.** Every quarter a public company holds an earnings call: management talks, then analysts ask questions. I used transcripts from the public [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data) dataset.

**Which companies.** I kept smaller US companies only:
- dollar trading volume (closing price × shares traded) between $1M and $50M a day, using the median day of the 60 trading days before the call;
- a share price of at least $3;
- at least 300 words of management answers and 100 words of analyst questions.

That left 59,753 calls from 2016 to 2026 ([`data/paper/calls.parquet`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/calls.parquet)).

**Masking.** The company's name, ticker and years are replaced in every text the model sees. Other details, like project names, stay in. A well-read person could sometimes still guess the company.

**Sets.** Before running anything, every call was assigned to one set, and the assignment was frozen with checksums in [`configs/paper/freeze.json`](https://github.com/nityarai08/earnings-call-distillation/blob/main/configs/paper/freeze.json):

| Set | Calls | Dates | Used for |
| --- | --- | --- | --- |
| SFT | 3,000 (800 used) | 2021 to 2024, even quarters | Training on Opus's answers |
| RL | 800 (400 used) | 2021 to 2024, odd quarters | Training with rewards |
| Validation | 3,000 (up to 500 used) | 2025 | Every comparison made while developing |
| Check | 1,000 (300 used) | 2023 to 2024 | Comparing each model's reading with Opus's on the same calls |
| **Test-W** | **800** | **March to August 2026** | **The final test in this post** |
| Test-F | 1,757 | March to August 2026 | Kept sealed for a larger future run |

**Why the test is trustworthy.** The model came out in February 2026, so it can't have seen what happened to these stocks. The test outcomes sat in a separate sealed file. Only the final scoring script opened it, once, after everything else had finished. The second test set (Test-F) is still sealed: no model has been scored on it, and its outcomes have never been opened.

## 3. One test call, start to finish

Here is the first call of the 800-call test set exactly as the models saw it. It was picked by the set's fixed random order, not for how it turned out. All the files for it are in [`examples/1_test_call/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/1_test_call).

### What the model reads

Every setup gets the same prompt layout. This is the RAG version ([full text](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/prompt_rag.txt), 4,511 tokens), shortened:

```
[system] You are an experienced stock analyst. The company's name is masked. Use only what is below.

[user]
Facts: sector Industrials; daily dollar volume $5M to $20M; past 60-session return -15%;
daily risk 2.0%; move around the call -1.4%; EPS up 0.95 on a year ago; position none.

Knowledge: This company, earlier call: Regarding oil, currently, we don't have any land to
significantly increase production. ... [2 earlier calls of this company]
Similar past call: ... we are in for a good sustained boom ... Then lagged its sector, and
next-quarter EPS fell on a year ago. [3 other companies' calls, each with what happened next]
Company figures: ... Analyst note: ...

Management's answers:
Great. So let's go to the next question. ... How are the royalties settled? Are they include
in the hedge? ... No, royalties do not have any connection to the hedging. ... [up to 2,500 words]

Reply with exactly these seven lines, choosing one value from each list and writing nothing else.
Demand: up, flat or down (over the next two quarters)
Pricing: up, flat or down (over the next two quarters)
Margins: up, flat or down (over the next two quarters)
Guidance: raised, held, cut or none (what management did)
Candor: direct or evasive (how they answered)
Next EPS: up or down (next quarter's EPS against a year earlier)
Action: BUY or AVOID (for the next 60 sessions)

[assistant] Demand:
```

The prompt has four parts:

1. **Facts.** One line of numbers known at the time of the decision: the stock's trading size, its return over the last 60 trading days, how much it moves on a typical day, how much it moved around the call, and the latest earnings change.
2. **Knowledge.** For RAG, a block of up to 1,500 tokens from a shared library of past calls. It holds two earlier calls from the same company, three similar calls from other companies (each with a note on how that stock did afterwards), the company's recent figures, and one general analyst note. The library only returns material dated before the decision.
3. **Management's answers.** The Q&A answers, masked, cut at 2,500 words.
4. **The instruction**, with the seven lines to fill in.

The setups differ only in what goes into these slots:

| Setup | Facts | Knowledge | Answers | Model |
| --- | --- | --- | --- | --- |
| Memory only | yes | none | "not provided" ([prompt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/prompt_memory.txt)) | base 9B |
| Raw | yes | none | yes ([prompt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/prompt_raw.txt)) | base 9B |
| RAG | yes | fixed search | yes ([prompt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/prompt_rag.txt)) | base 9B |
| Pi | yes | the model's own searches | yes ([prompt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/prompt_pi.txt)) | base 9B |
| SFT | yes | fixed search | yes (same as RAG) | 9B + [SFT adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/sft_grown) |
| RL | yes | fixed search | yes (same as RAG) | 9B + [RL adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/trade) |

For Pi, the model first runs an agent loop with two tools: `kb_search` (search the library) and `kb_read` (open one document). It made about 4 tool calls per call and kept about 2 documents. Those documents became its knowledge block.

### How an answer is read

The model doesn't write freely. At each line, I look at the probability it gives to each allowed value, keep the most likely one, and move on to the next line. That gives two exact numbers per call:

- **p(Next EPS up):** how sure it is that next quarter's earnings will rise.
- **p(BUY):** how sure it is about BUY rather than AVOID. **This is the stock score.**

### What each setup answered

([`answers.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/answers.csv))

| Setup | Demand | Pricing | Margins | Guidance | Candor | p(Next EPS up) | p(BUY) | Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Memory only | down | flat | flat | none | evasive | 0.47 | 0.00 | AVOID |
| Raw | flat | flat | up | none | evasive | 0.62 | 0.20 | AVOID |
| RAG | up | flat | up | none | direct | 0.41 | 0.03 | AVOID |
| Pi | flat | flat | up | none | evasive | 0.87 | 0.08 | AVOID |
| SFT | up | flat | up | none | direct | 0.68 | 0.69 | BUY |
| RL | up | flat | up | none | direct | 0.66 | 0.67 | BUY |

### What actually happened

([`outcome.json`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/1_test_call/outcome.json))

The call was on 2026-03-03. The model decides at the close of the next trading day (03-04), buys at the following open, and sells at the close 60 trading days later (05-29).

| Step | Value |
| --- | --- |
| Return over the 60 sessions, dividends included | +11.3% |
| Average return of the same sector's stocks | +9.8% |
| **Excess return** = 11.3% − 9.8% | **+1.5%** |
| The stock's typical daily move (volatility) | 1.96% |
| **Outcome y** = excess ÷ (daily volatility × √60) = 0.0149 ÷ (0.0196 × 7.75) | **+0.098** |
| Next quarter's EPS minus a year earlier | +0.98, so **up** |

The stock beat its sector slightly, and earnings rose. SFT and RL said BUY and forecast EPS up, so they were right on this call. The untrained setups said AVOID. One call proves nothing, though, so everything below is scored over all 800.

**Why divide by volatility?** A 1.5% excess return means little for a stock that swings 5% a day, and more for one that barely moves. Dividing by the volatility expected over 60 sessions (daily volatility × √60) puts every stock on the same scale.

## 4. How the student was trained

### The teacher's answers

Claude Opus got the same prompt the student gets (facts, knowledge block and answers) and returned the same lines, plus a 0–100 score and up to three quotes. It read each training call twice: once with a neutral instruction and once told to be skeptical. Here are both answers for the first training call ([`opus_labels.jsonl`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/2_sft_training_call/opus_labels.jsonl)):

| | Demand | Pricing | Margins | Guidance | Candor | Next EPS | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Opus, neutral | up | flat | up | none | direct | up | 61 |
| Opus, skeptical | up | flat | flat | none | direct | up | 58 |

The quotes Opus picked were: *"For the first quarter, chassis shipments were at record levels."*, *"the supply chain base for all of our products seems extremely strong"*, and *"with the strong demand for the carrier product, we plan to keep it dedicated on carrier production"*.

All 2,200 Opus and Sonnet answers are in [`data/paper/teacher/labels_v2.jsonl`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/teacher/labels_v2.jsonl). They cover 800 training calls × 2 instructions, plus 300 check calls read by Opus and 300 by Sonnet. Together they cost about $37.

### Fine-tuning (SFT)

Each training call becomes one target ([`target.json`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/2_sft_training_call/target.json)):

| Line | Target | Where it comes from |
| --- | --- | --- |
| Demand … Candor | up, flat, up, none, direct | Opus's first (neutral) answer |
| Next EPS | up | **What really happened**, not Opus's guess |
| Action | **BUY with probability 0.84** | Opus's average score (59.5), ranked among the 40 training calls from the same month |

**How 0.84 is calculated.** The average score of 59.5 ranks 34th of the 40 calls that month. The percentile rank is 34 ÷ 40 = 0.85. Subtracting half a step (0.5 ÷ 40 = 0.0125) gives **0.8375**, which keeps targets away from exactly 0 or 1. The model is trained to say BUY with 84% confidence, not 100%.

Here is the training text: the prompt, then the target answer ([`training_text.txt`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/2_sft_training_call/training_text.txt)):

```
... (prompt as above) ...
[assistant] Demand: up
Pricing: flat
Margins: up
Guidance: none
Candor: direct
Next EPS: up
Action: BUY            ← trained toward 0.84 BUY / 0.16 AVOID
```

**The loss.** At each line, the model's probabilities over the allowed values are compared with the target using cross-entropy. Cross-entropy is small when the model puts high probability on the target value. Only those seven positions count, not the rest of the text.

**Training settings:**
- **What gets trained:** small add-on weights (a LoRA adapter, rank 16) on the top 8 layers of the 9B. The original weights never change.
- **Batches:** 4 calls per update.
- **Calls and passes:** 400 calls, 2 passes, at a learning rate of 1e-4. Then 400 more calls, 2 passes, at 3e-5, continuing from there.

The result is the [SFT adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/sft_grown), and the training curve is in [`train/sft_grown.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/train/sft_grown.csv).

**The other SFT versions, all compared on the same 300 validation calls:**

| Version | Trained on |
| --- | --- |
| read + EPS (the one kept) | the read lines, next EPS and action, as above |
| score only | the action target alone |
| read | the read lines and action, without next EPS |
| hard | the action target = 1 if the stock *actually* beat its month's median outcome, else 0 |
| **placebo** (control) | Opus's read lines and score **shuffled between calls in the same month**, so they no longer belong to the call. Next EPS stays real |

### Reinforcement learning (RL)

RL starts from the SFT model and uses 400 other training calls. For each call it:

1. Samples 4 different readings of the call (at temperature 0.8, so the readings vary).
2. Scores each reading with a reward.
3. Moves the model toward the best reading, and moves its BUY probability toward what the reward favors, while staying close to where it started.

**The four rewards:**

| Reward | Formula | Pays for |
| --- | --- | --- |
| trade | p(BUY) × reward_if_BUY + p(AVOID) × 0 | Buying stocks that then beat the month's average |
| business | log p(the true Next EPS) + log 2 | A confident, correct earnings forecast (0 when 50/50) |
| both | business ÷ 0.7 + trade | Both |
| **shuffled** (control) | trade, but with outcomes shuffled within the month | Nothing real |

**A worked example of the trade reward** ([`reward.json`](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/3_rl_training_call/reward.json)). This is the fifth RL call, chosen because its outcome is clear:

| Step | Value |
| --- | --- |
| Outcome y of this stock | −0.240 |
| Average y of the RL calls that month | +0.558 |
| z = y − month average (kept within ±3) | −0.798 |
| Trading cost in the same units: 0.8% ÷ (daily risk × √60) = 0.008 ÷ 0.313 | 0.026 |
| **Reward if BUY** = z − cost | **−0.824** |
| Reward if AVOID | 0 |
| The SFT model's p(BUY) before RL | 0.25 |
| Expected reward = 0.25 × (−0.824) + 0.75 × 0 | −0.206 |
| **RL target for p(BUY)** = 0.25·e^(−0.824) ÷ (0.25·e^(−0.824) + 0.75) | **0.13** |

The stock lagged its month badly, so RL pushes p(BUY) from 0.25 down to 0.13.

Why this formula: it's the best policy that stays close to the starting model. The target is the starting probability multiplied by e^reward, then rescaled so the probabilities add to 1. A plain policy-gradient version collapsed into always saying AVOID, so I used this closed form instead.

The run made 100 updates per reward (400 calls ÷ 4). The best reward on validation calls was trade, which became the [RL adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/trade). Its curve is in [`train/trade.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/train/trade.csv).

## 5. How every number is calculated

All scores come from one script, [`scripts/paper/score.py`](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/score.py). Its inputs are each setup's per-call answers ([`runs/paper/week/reads/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/reads), for example [`rl_best_test.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/reads/rl_best_test.csv)) and the test outcomes ([`sealed/test_w.parquet`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sealed/test_w.parquet)).

### Stock ranking score (IC)

The IC asks whether stocks the model liked more did better. It's computed in three steps:

1. **Rank within each month.** For each month, every call's p(BUY) becomes a percentile rank among that month's calls, from 0 to 1. The outcome y is ranked the same way. Ranking within the month means a market-wide rally or slump can't count as skill.
2. **Correlate.** Take the ordinary correlation between the two rank columns across all 800 calls.
3. **Read it:** 1 is a perfect ordering, 0 is no relation, and −1 is backwards.

A small example: five calls in one month.

| Call | p(BUY) | Its rank | Outcome y | Its rank | Gap d | d² |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.80 | 1 | +0.6 | 1 | 0 | 0 |
| B | 0.60 | 2 | −0.2 | 4 | −2 | 4 |
| C | 0.50 | 3 | +0.3 | 2 | 1 | 1 |
| D | 0.30 | 4 | +0.1 | 3 | 1 | 1 |
| E | 0.10 | 5 | −0.5 | 5 | 0 | 0 |

With a single month, this is the Spearman rank correlation:

```latex
IC = 1 - \frac{6 \sum d^2}{n(n^2 - 1)} = 1 - \frac{6 \times 6}{5 \times 24} = 0.70
```

In the real test, the example call from section 3 fell in March 2026, alongside 129 other test calls. Among them, RL's p(BUY) for it sat at the 65th percentile and its outcome at the 60th: a small agreement. Over all 800 calls, RL's IC is **0.113**.

Real ICs are far below 0.70. Across hundreds of stocks, a steady 0.05 to 0.10 is already useful, because small edges add up over many trades.

### The other scores

| Score | How it's calculated | RL on the test |
| --- | --- | --- |
| **Next-EPS IC** | Same as the stock IC, using p(Next EPS up) and the real EPS change (743 of the 800 calls have one) | 0.278 |
| **Stock accuracy** | Share of calls where "p(BUY) > 0.5" matches "the stock beat its sector" | 53.6% |
| **EPS accuracy** | Share of calls where "p(Next EPS up) > 0.5" matches the real direction | 64.2% |
| **Agrees with Opus's read** | On 300 check calls that Opus also read, the share of the five read lines with the same value as Opus, averaged over the lines | 70.6% |
| **Return vs sector after costs** | Average excess return of the calls where p(BUY) > 0.5, minus a 0.8% round-trip trading cost | 438 BUYs averaging +1.86% → **+1.06%** |

A read-match example: on check call 62273, Opus read demand up, pricing flat, margins down, guidance none and candor direct. The raw model matched 3 of the 5 lines and the SFT model matched 2.

### Is one setup really better than another?

Two setups are compared on the same calls by the difference in their ICs. To see how much that difference could move by luck, I resample: draw 22 weeks at random, with repeats, from the 22 weeks the test covers; recompute both ICs; record the difference; repeat 2,000 times. Whole weeks are drawn because calls in the same week share market news. The middle 95% of the 2,000 differences is the range shown. If it includes zero, the two setups are **not distinguishable** at this size.

### Do the actions follow the reasons?

On 200 validation calls, I ran the model twice, once forcing "Demand: up" and once "Demand: down", and compared p(BUY). For the first call, the SFT model gave 0.27 versus 0.22. Averaged over 200 calls, it gave 0.54 versus 0.31.

## 6. Results

On the 800 test calls, every AI setup ranked stocks at about 0.10, and no step beat the previous one by more than luck.

![Stock-ranking score by setup](figures/stock_ic_by_setup.svg)

Reading the transcript helps a little: memory only (0.075) and the simple rules (0.06–0.085) sit lower.

The simple rules use no AI:
- **Known-EPS rule:** ranks stocks by the latest earnings change divided by price.
- **Word list:** ranks by the tone of management's answers, scored from a fixed list of positive and negative words.
- **Numbers only:** a small tree model on the facts line, fitted to the training calls.

![Gain from each ladder step with its 95% range](figures/ladder_step_gains.svg)

All four ranges cross zero. RL's range is the tightest, and even there the honest summary is "somewhere between slightly worse and slightly better."

| Setup | Stock IC | Next-EPS IC | EPS accuracy | Agrees with Opus | BUY calls | Return vs sector after costs |
| --- | --- | --- | --- | --- | --- | --- |
| Memory only | 0.075 | 0.253 | 64.5% | — | 229 | −0.17% |
| Raw | 0.104 | 0.236 | 58.4% | 65.8% | 302 | +0.36% |
| RAG | 0.096 | 0.229 | 56.1% | 67.3% | 276 | −0.38% |
| Pi | 0.104 | 0.179 | 56.0% | 67.3% | 280 | +0.72% |
| SFT | 0.105 | 0.262 | 64.9% | 69.9% | 483 | +0.50% |
| RL | 0.113 | 0.278 | 64.2% | 70.6% | 438 | +1.06% |
| Known-EPS rule | 0.085 | 0.244 | 66.5% | — | 381 | +0.94% |
| Word list | 0.063 | 0.055 | 51.7% | — | 400 | +1.42% |
| Buy every call | — | — | — | — | 796 | −0.68% |

Full tables: [`results.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/results.csv) and [`findings.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/findings.csv).

## 7. What I learned

### 1. It learned to sound like Opus, not to trade like Opus

After training, the small model agreed with Opus's read more often: 66% for raw, 71% after RL. Its earnings forecasts improved too, from 0.236 to 0.278. Its stock ranking didn't move.

Look at which lines changed, though. Most of the agreement gain is on Candor:
- Opus calls management "direct" 95% of the time.
- The trained model learned to say "direct" 98% of the time.
- The match on that line rose from 82% to 94%.

So part of "sounding like Opus" is copying its habits, not its judgment.

The controls, run on 300 validation calls from 2025, make the point sharper:
- **The placebo model.** It was trained on Opus answers that didn't belong to the call. It collapsed into never saying BUY, yet it ranked stocks at 0.099, the same as RAG (0.094). The best real SFT model beat it by only 0.04, with a range of −0.07 to +0.17.
- **The shuffled reward.** RL with the real trade reward scored 0.109; with the shuffled reward it scored 0.102.

Training on noise did about as well as training on the real thing.

### 2. Its actions follow its reasons

Forcing Demand from "down" to "up" raised the SFT model's p(BUY) from 0.31 to 0.54, in the right direction on all 200 calls. After RL the shift was +0.21, right on 199 of 200. The model isn't writing a story and then ignoring it. The weak spot is the reading itself: on the check calls, Opus called the EPS direction right 72% of the time and ranked stocks at 0.15, above anything the small model reached.

### 3. Searching on its own made it worse

With search tools, the model's earnings forecasts fell to 0.179 on the test, against 0.229 for fixed RAG. To find out why, I ran the same agent loop but handed it RAG's documents instead of its own picks ([`pi_rag_docs.csv`](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/reads/pi_rag_docs.csv)). On 200 shared validation calls the scores were:
- with its own picks: 0.14;
- the same loop with RAG's documents: 0.22;
- fixed RAG: 0.28.

The loop itself is fine; the small model is bad at choosing what to read.

### 4. No money to be made yet

Returns after costs ranged from −0.68% to +1.42% over 60 days. The best came from the word list, with no AI at all. Differences this small are noise.

## 8. Conclusions

**Teaching a small model to copy a big one made it sound more like the big one, not invest better.** What I'd take away:

- **Always run a placebo.** Without the shuffled-label and shuffled-reward controls, I'd have reported that SFT and RL "work".
- **Check which lines changed.** "Agrees with the teacher more" can mean "copies the teacher's most common answer".
- **Check that the reasoning drives the decision.** Here it does, so the thing to fix is the reading.
- **Don't let a small model pick its own sources** without checking its picks.
- **Simple rules are a high bar.** The one-line earnings rule came within 0.03 of the best model.

**What I wouldn't claim yet:**
- **The sample is small.** 800 test calls means "no difference" here is "no difference visible at this size", not "proven zero".
- **The scope is narrow.** It's one small model, one holding period and one slice of the market.
- **The trained models only work in constrained mode.** They put almost no probability on the allowed answer words compared with the rest of the vocabulary (`p_mass` near 0 in their files). Their scores compare the allowed values with each other, so the results stand. But left to write freely, they wouldn't follow the format.

**Running it on a laptop.** Three things broke along the way:
- a learning rate that was too high;
- one call with no price outcome, which crashed RL;
- a closed lid on battery, which put the Mac to sleep for an hour.

All three are written up in [`docs/FIXES.md`](https://github.com/nityarai08/earnings-call-distillation/blob/main/docs/FIXES.md).

**Next, for the paper:**
1. Score the trained models on the sealed 1,757-call Test-F to tighten the ranges.
2. Give the small model Opus's read and let it make only the trade. If its ranking climbs toward Opus's 0.15, the reading is the bottleneck.

## Files

Everything below is in the [repo](https://github.com/nityarai08/earnings-call-distillation). `bash reproduce/rescore.sh` recomputes every number in section 6 from these files in about a minute.

| What | File |
| --- | --- |
| A test call: every setup's exact prompt, answer and outcome | [`examples/1_test_call/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/1_test_call) |
| A training call: Opus's answers, the target and the training text | [`examples/2_sft_training_call/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/2_sft_training_call) |
| An RL call: the prompt and the reward arithmetic | [`examples/3_rl_training_call/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/3_rl_training_call) |
| Every call, its set and the facts the model saw | [`data/paper/sets.parquet`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sets.parquet) |
| Every Opus answer used for training | [`data/paper/teacher/labels_v2.jsonl`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/teacher/labels_v2.jsonl) |
| Every model answer, one file per setup | [`runs/paper/week/reads/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/reads) |
| Test outcomes (opened at the end) | [`data/paper/sealed/test_w.parquet`](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sealed/test_w.parquet) |
| Training curves | [`runs/paper/week/train/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/train) |
| The trained adapters | [`adapters/paper/week/`](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week) |
| The prompt, training and scoring code | [`scripts/paper/prompt.py`](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/prompt.py), [`train_soft.py`](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/train_soft.py), [`score.py`](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/score.py) |
| What every column means | [`docs/DATA.md`](https://github.com/nityarai08/earnings-call-distillation/blob/main/docs/DATA.md) |
