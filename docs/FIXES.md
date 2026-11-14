## 2026-11-08 20:55 PT — week sft_grown diverged; restarted at lower learning rate
- Seen: runs/paper/week/train/sft_grown.csv read loss 4 -> 20-68 from update 10 (end of warmup to peak 1e-4). Grown continues the trained read_eps adapter with a fresh Adam and a schedule restarted at the full SFT peak.
- Kept the curve: runs/paper/week/train/sft_grown_diverged_lr1e-4.csv
- Commands: kill 63638 (run_step), kill 51291-51293 (training); no FAILED.json written.
- Change: train_soft.py adds --lr peak,floor (default unchanged, SFT_LR); make_commands.py passes --lr 3e-5,3e-6 to the grown arm only; regenerated configs/paper/commands.yaml (diff: the three grown lines only).
- Restart: nohup .venv/bin/python scripts/paper/run_step.py week >> logs/paper_week.log 2>&1 &  (20:56 PT, grown started from update 0)
- Cost: about 55 min of training lost.

## 2026-11-09 08:58 PT — week SFT score arm diverged; score and hard arms moved to lower learning rate
- Seen: runs/paper/week/train/score.csv action loss 4-30 for updates 44-55 (29.8 at 47), after spikes from update 7; read_eps action loss stayed ~0.6. The score arm trains on one target token (Action), so each update at the 1e-4 peak flips its confidence.
- Kept: curve runs/paper/week/train/score_diverged_lr1e-4.csv; broken adapter moved to the session scratchpad (score_adapter_diverged).
- Commands: kill run_step 54581 and score training 89629-89631; no FAILED.json written.
- Change: make_commands.py adds --lr 3e-5,3e-6 to arms whose lines are "action" only (score, hard), as for sft_grown; regenerated commands.yaml (diff: those six lines only). read and placebo arms are unchanged at 1e-4.
- Paper note: score and hard now differ from read_eps in learning rate as well as target, so their comparison with read_eps carries this caveat.
- Restart: nohup .venv/bin/python scripts/paper/run_step.py week >> logs/paper_week.log 2>&1 &
- Cost: about 2 h 15 min of training lost.

## 2026-11-11 07:35 PT: RL "both" arm NaN at update 69

- Symptom: `both.csv` update 69 had reward nan; update 70 crashed in `sample()` with "Probabilities contain NaN". The runner retried from the update-50 checkpoint (attempt 2).
- Cause: RL call 275 (row 91918, 2023-08) has no outcome `y` and no `risk`, so `r_pos`/`r_neg` were NaN. The trade part of the reward became NaN, then the Action tilt target, then the gradients and weights. The business arm survived because its reward does not use `r_pos`/`r_neg`.
- Fix: `rewards()` in `scripts/paper/train_soft.py` now fills NaN `r_pos`/`r_neg` with 0, so a call with no price outcome gives no trade reward. All arms keep the same 400 calls.
- Commands: `kill <run_step, llm_locked, caffeinate, train_soft pids>`; crashed log kept at `runs/paper/week/train/both_crashed_nan.csv.bak`; `both.csv` trimmed to updates 1-50; restart `nohup .venv/bin/python scripts/paper/run_step.py week >> logs/paper_week.log 2>&1 &` (resumes from the checkpoint at call 200).
- Cost: about 2 hours of the both arm (updates 51-69 twice). Trade and shuffled arms have not run yet, so they get the fix from the start.
