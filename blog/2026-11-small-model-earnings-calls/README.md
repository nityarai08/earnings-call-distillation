# Teaching a Laptop Model to Read Earnings Calls: What Held Up and What Did Not

*Nityanand Rai · November 13, 2026*

*I tried to teach a small AI model to read company earnings calls like a large model. The small model runs on my laptop. The large model is Claude Opus. After training, the small model's answers looked more like Opus's answers. Some signals held up in two separate periods, and some did not. This post shows both.*

*This post shows what the small model read, what it wrote, how I trained it, and how I calculated each number. Each section links to the real files in the [repo](https://github.com/nityarai08/earnings-call-distillation). All examples in this post are real calls. The folder [examples](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/README.md) has real inputs and outputs for every experiment and test.*

## The short version

- I compared six setups of one small model, Qwen3.5-9B, on 800 earnings calls that it had never seen.
- **Held up:** every setup forecast the next EPS change better than chance, in 2026 and again in 2025. But a simple rule that uses only the last EPS change did almost as well.
- **Held up:** the small model's BUY decision follows its own read lines. Opus reads the same calls better (next bullet). So better read lines are the most direct way to improve the model.
- **Held up:** on the same 800 test calls, Opus ranked stocks better than the trained small model (IC 0.170 against 0.113 for RL). Its BUY calls also earned more after costs (+2.09% against +1.06%). After Opus's knowledge date (July and August), the IC gap is not distinguishable, but the return gap remains.
- **A lead:** the model and two simple rules pick partly different stocks. Their mix ranked stocks a little better than the rules alone, in both periods. The gain is not yet larger than chance can explain.
- **Did not hold up:** on the 2026 test calls, the setups ranked stocks better than chance (IC values of approximately 0.10). On the 2025 validation calls, the same setups did not.
- **Did not hold up:** RL had the highest return after costs on the test calls, +1.06%. On the 2025 validation calls, the same RL model had −2.08%.
- Training on Opus's answers made the small model agree with Opus more often: 66% for Call only, 71% after RL. About half of that gain came from one habit. The small model learned to call management "direct" almost every time, as Opus does.
- No training step improved stock ranking by more than chance can explain. A control model that I trained on shuffled Opus answers ranked stocks about as well.

Section 8 lists each signal, with its values in both periods.

---

## Words used in this post

| Word | Meaning in this post |
| --- | --- |
| **Call** | One quarterly earnings call of one company |
| **Small model** | Qwen3.5-9B, an open model with 9 billion parameters (numbers that the model learned), compressed to 4 bits so that it runs on a laptop |
| **Opus** | Claude Opus, the large model that I used as the teacher |
| **Answer** | The seven lines that a model writes for one call (shown in section 3) |
| **Read lines** | The first five lines of an answer: Demand, Pricing, Margins, Guidance and Candor |
| **Managers' replies** | The managers' answers to the analysts' questions, from the transcript of the call. The prompt calls this part "Management's answers" |
| **p(BUY)** | The probability of BUY divided by the sum of the probabilities of BUY and AVOID. The script reads it from the model. The model does not write it (section 3). This is the stock score |
| **Setup** | One way to use the small model. There are six setups (section 3) |
| **Control** | A setup that I broke on purpose. If a control scores as well as the real setup, the real setup's gain is not real |
| **Library** | A searchable store of past calls, company figures and analyst notes |
| **Knowledge block** | The part of the library that a setup adds to the prompt, up to 1,500 tokens |
| **Adapter** | A small set of extra weights that training adds to the small model. The original weights do not change |
| **Outcome y** | How well a stock did after a call, compared with its sector (section 3) |
| **IC** | Information coefficient: how well a list sorted by p(BUY) matches a list sorted by outcome y (section 5) |
| **Step** | A change from one setup to the next: Call only to RAG, RAG to Pi, RAG to SFT, and SFT to RL |
| **Simple rule** | A method that ranks stocks without AI (section 6) |
| **Daily volatility** | The usual daily price change of a stock. The prompt calls it "daily risk" |
| **Session** | One trading day |
| **Token** | A word or a part of a word. 1,000 tokens are approximately 750 words |
| **Test set** | 800 calls from March to August 2026. Nobody looked at their outcomes until the final step |

## 1. The question

Opus reads an earnings call well. But it costs money to run Opus on thousands of calls. The small model runs free on a laptop, but it reads less well.

My question was: **can the small model learn from Opus to read calls well enough to pick stocks?**

I also tested four usual methods, one at a time:

1. Add documents from a library to the prompt.
2. Let the model search the library itself.
3. Train the model on Opus's answers.
4. Train the model further with rewards.

I also ran controls (see "Control" in the table above), so that a lucky result cannot look like a real gain.

## 2. The data

**Calls.** In an earnings call, the managers of a company talk about the last quarter, and analysts ask them questions. I used call transcripts from the public dataset [defeatbeta/yahoo-finance-data](https://huggingface.co/datasets/defeatbeta/yahoo-finance-data).

**Companies.** I used only smaller US companies. A call stayed in the data only if all of these were true:

- The stock's dollar trading volume (closing price × shares traded) was between $1M and $50M a day. I used the median day of the 60 trading days before the call.
- The share price was $3 or more.
- The managers' replies had 300 words or more, and the analysts' questions had 100 words or more.

This filter kept 59,753 calls from 2016 to 2026 ([calls.parquet](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/calls.parquet)).

**Masking.** Before a model reads a call, a script replaces the company name, the ticker and the years. Other names, for example project names, stay in the text. Thus a person or a model with wide knowledge can sometimes identify the company.

**Sets.** Before I ran any model, a script put each call into one set. The script recorded the sets and their checksums in [freeze.json](https://github.com/nityarai08/earnings-call-distillation/blob/main/configs/paper/freeze.json).

| Set | Calls in set | Calls I used | Dates | Purpose |
| --- | --- | --- | --- | --- |
| SFT training | 3,000 | 800 | 2021 to 2024, even quarters | Train on Opus's answers |
| RL training | 800 | 400 | 2021 to 2024, odd quarters | Train with rewards |
| Validation | 3,000 | up to 500 | 2025 | Compare setups during development |
| Check | 1,000 | 300 | 2023 to 2024 | Compare the small model's read lines with Opus's read lines |
| **Test set** | **800** | **800** | **March to August 2026** | **The final test in this post** |
| Second test set | 1,757 | 0 | March to August 2026 | Other calls from the same months, kept sealed for a larger run later |

**Why the test is fair.** Qwen released the small model in February 2026. Thus it cannot know what happened to these stocks after March 2026. The test outcomes stayed in a separate sealed file. Only the final scoring script opened that file, and only after all other steps were complete. The outcomes of the second test set are still closed.

## 3. One test call, from input to outcome

This section follows one real call from the test set. I did not choose it for its result. It is the first call in the test set's fixed random order. Its answers are in [examples/week/test](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/week/test), one folder for each setup.

### The prompt

All setups use the same prompt layout. Below is a shortened copy of the prompt that the RAG setup read. The full prompt, [218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt), has 4,511 tokens.

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

1. **Facts.** One line of numbers that a trader could see on the day of the decision. The numbers are:
    - the stock's trading volume
    - its return over the last 60 trading days
    - its daily volatility, which the prompt calls "daily risk"
    - its price change around the call
    - the change in its last earnings per share (EPS).
2. **Knowledge.** The knowledge block from the library. In the RAG setup it contains:
    - two earlier calls of the same company
    - three similar calls of other companies, each with a note about how that stock did after its call
    - the company's recent figures
    - one general analyst note.

    The library returns only material dated before the decision.
3. **Managers' replies.** The managers' replies from the question-and-answer part of the call, masked and cut at 2,500 words.
4. **The instruction.** The seven lines that the model must complete.

### The library and the RAG search

**Where the study uses the library.** Five parts of the study read from the library:

- the RAG setup, through a fixed search
- the SFT and RL models, in training and in testing, through the same fixed search
- Opus, which received the same knowledge block as the small model
- the Pi setup, through its two search tools
- the two RAG controls, which replace the search results with random passages or with shuffled outcomes.

**What the library contains.** The script [build_kb.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/build_kb.py) built the library as one SQLite file:

| Content | Size | Date when a search can return it |
| --- | --- | --- |
| Passages of managers' replies | 604,728 passages from 59,753 calls | The decision date of the call |
| Outcome cards, for example "Then beat its sector, and next-quarter EPS rose on a year ago." | 56,013 cards | The sell date of the trade, 60 sessions after the decision |
| Quarterly figures: EPS, revenue and gross margin | 214,433 quarters of 3,103 companies | The report date, or 60 days after the quarter end if the report date is not known |
| Analyst notes, written once by Opus | 50 notes | Always |

Test calls have no outcome cards.

**How the calls are cut into passages.** For each call, the script takes the first 3,000 words of the masked managers' replies. It cuts these words into passages of 220 words, one after the other, with no overlap. It deletes a piece that has fewer than 200 characters. The mean passage has 212 words.

**No embedding model.** The search does not use embeddings. It uses the full-text index of SQLite (FTS5). This index ranks passages with the BM25 formula. BM25 gives a higher score to a passage that contains the query words more often, and to query words that are rare in the library.

**How the script makes the query.** The query comes from the call's own text, in four steps ([kb.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/kb.py)):

1. Count each word that has 4 or more letters. Ignore 37 common words, for example "that", "quarter" and "company".
2. Keep the 80 most frequent words.
3. Give each word a score: its count in the call ÷ (1 + the number of passages that contain the word).
4. Keep the 12 words with the highest scores. The query is these 12 words joined by OR.

**What goes into the knowledge block.** The block contains:

- 2 passages from earlier calls of the same company
- 3 passages from calls of other companies, each with its outcome card if the card is available
- the company's last 4 quarters of figures
- 1 analyst note.

The script cuts each passage to its first 120 words. Then it cuts the whole block at 1,500 tokens.

**The date rule.** A search for one decision date returns only passages, cards and figures that were available before that date. A search never returns the call that the model is reading. The command `build_kb.py audit` checks these rules.

**A real search.** For the test call above, the decision date was 2026-03-04. The 12 query words were: rinc, cisa, rigi, aranda, alejandro, cubic, binding, meters, vertically, impairment, asking and hedge. The search returned these passages:

| Part | Source call | Date of the passage | Outcome card |
| --- | --- | --- | --- |
| Same company | 71023 | 2022-08-15 | Then beat its sector, and next-quarter EPS rose on a year ago. |
| Same company | 71030 | 2024-05-10 | Then lagged its sector, and next-quarter EPS fell on a year ago. |
| Other company | 80649 | 2021-02-05 | Then lagged its sector, and next-quarter EPS fell on a year ago. |
| Other company | 80645 | 2020-02-10 | Then beat its sector, and next-quarter EPS rose on a year ago. |
| Other company | 80643 | 2019-08-09 | Then beat its sector, and next-quarter EPS fell on a year ago. |

No analyst note matched the query. This search shows two limits of keyword search:

- The word "rinc" is a part of "Rincón". The word filter keeps only the letters a to z, so the accent cuts the word. "alejandro" is the first name of an analyst.
- The three passages of other companies match mainly on "cubic meters". One of them is from a forestry company.

The searches for all 10 example test calls are in [searches.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/rag/searches.csv).

### The six setups

The six setups differ only in the knowledge block, the managers' replies and the model weights. All setups get the facts line. The prompt files start with the same lines. They differ in the part after "Knowledge:", in the part after "Management's answers:", or in both.

| Setup | What it means | Knowledge block | Managers' replies | Model | Prompt file |
| --- | --- | --- | --- | --- | --- |
| No call | The model sees the facts line but not the call | none | not given | small model | [218058_nocall_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_nocall_noblock.txt) |
| Call only | The model reads the call, with no knowledge block | none | given | small model | [218058_noblock.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_noblock.txt) |
| RAG | A fixed search adds documents ("retrieval-augmented generation") | from a fixed search | given | small model | [218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt) |
| Pi | The model searches the library itself, with the pi agent tool | from the model's own searches | given | small model | [218058_pi.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_pi.txt) |
| SFT | RAG, after training on Opus's answers ("supervised fine-tuning") | from a fixed search | given | small model + [SFT adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/sft_grown) | [218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt) (same as RAG) |
| RL | SFT, after more training with rewards ("reinforcement learning") | from a fixed search | given | small model + [RL adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/trade) | [218058_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/218058_rag.txt) (same as RAG) |

The run files and example folders use short names. The No call setup is `memory`, and the Call only setup is `raw`. For example, the test answers of the Call only setup are in `runs/paper/week/reads/raw_test.csv`.

In the Pi setup, the model first uses two tools. The tool `kb_search` searches the library, and the tool `kb_read` opens one document. On average, the model made about 4 searches or reads for each earnings call and kept about 2 documents. These documents became its knowledge block. The searches for 10 calls are in [agent_runs.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/pi/agent_runs.csv).

### How the answer becomes numbers

The prompt asks for seven lines of words. The model writes only these seven lines. It does not write any probability. The numbers come from the model itself, in this way.

**How a language model writes.** A language model writes one token at a time. A token is a word or a part of a word. Before each token, the model gives a probability to every token in its vocabulary. This model has a vocabulary of about 248,000 tokens. Usually, a program then picks one token and continues.

**What the script does instead.** The script does not let the model write freely. It does these steps for each of the seven lines ([read.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/read.py)):

1. The text before the value ends with the label of the line, for example `Demand:`. The prompt itself ends with `Demand:`.
2. The script reads the probability that the model gives to each allowed value, for example " up", " flat" and " down".
3. It divides each of these probabilities by their sum. Now the allowed values add up to 1.
4. It writes the most probable value after the label. Then it writes the next label, for example `Pricing:`, and does steps 2 to 4 again.

At the line `Next EPS:`, the script keeps the value for "up". This is **p(Next EPS up)**. At the line `Action:`, it keeps the value for "BUY". This is **p(BUY)**, the stock score.

**A real example.** These are three of the seven lines for the test call above, in the RAG setup. The [full trace](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/rag_test/sample_output.md) shows all seven lines.

| Line | Allowed value | Probability out of all 248,000 tokens | Probability among the allowed values | Written |
| --- | --- | --- | --- | --- |
| Demand | up | 0.7590 | 0.7664 | yes |
| Demand | flat | 0.2175 | 0.2196 | |
| Demand | down | 0.0139 | 0.0140 | |
| Next EPS | up | 0.4047 | 0.4073 | |
| Next EPS | down | 0.5888 | 0.5927 | yes |
| Action | BUY | 0.0286 | 0.0293 | |
| Action | AVOID | 0.9484 | 0.9707 | yes |

Thus p(BUY) = 0.0286 ÷ (0.0286 + 0.9484) = **0.0293**, and p(Next EPS up) = 0.4047 ÷ (0.4047 + 0.5888) = **0.4073**. The script ran the model again for this post, and it got the same values as the published results.

**Why the scores use the probability and not the word.** The words BUY and AVOID put the calls into only two groups. The IC needs an order of all calls. p(BUY) gives that order: a call with p(BUY) = 0.69 ranks above a call with 0.20.

**Values with more than one token.** Some values are more than one token. " AVOID" is " A" + "VOID", " SELL" is " S" + "ELL", and " evasive" is " ev" + "asive". The script uses the probability of the first token. The first tokens " A" and " S" also start other words. Thus their probability can include other words that start with these letters.

**The trained models.** After training, the SFT and RL models give very small probabilities to all tokens. In the [RL trace](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/rl_best_test/sample_output.md), " BUY" gets 3.16e-05 and " AVOID" gets 1.54e-05 of the vocabulary. But the ratio between them is still clear: p(BUY) = 3.16e-05 ÷ (3.16e-05 + 1.54e-05) = 0.67. The stock score uses only this ratio.

**Opus and the simple rules.** Opus writes its answer as JSON with the same lines and a rating from 0 to 100. Opus gives no probabilities. When the post compares Opus's IC, the stock score of Opus is its rating ÷ 100. The simple rules read no prompt. Each rule's score is the percentile rank of one number, for example the EPS change ÷ the share price. The Numbers only rule uses the output of a small decision-tree model.

**Sample output for each experiment.** Each experiment folder in [examples/README.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/README.md) has a sample_output.md file for its first call. The file shows what the model wrote and how the script turned the probabilities into the numbers.

### The answers of the six setups

The data comes from the answers files in [examples/week/test](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/week/test).

| Setup | Demand | Pricing | Margins | Guidance | Candor | p(Next EPS up) | p(BUY) | Action | Sample output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No call | down | flat | flat | none | evasive | 0.47 | 0.00 | AVOID | [memory_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/memory_test/sample_output.md) |
| Call only | flat | flat | up | none | evasive | 0.62 | 0.20 | AVOID | [raw_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/raw_test/sample_output.md) |
| RAG | up | flat | up | none | direct | 0.41 | 0.03 | AVOID | [rag_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/rag_test/sample_output.md) |
| Pi | flat | flat | up | none | evasive | 0.87 | 0.08 | AVOID | [pi_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/pi_test/sample_output.md) |
| SFT | up | flat | up | none | direct | 0.68 | 0.69 | BUY | [sft_best_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/sft_best_test/sample_output.md) |
| RL | up | flat | up | none | direct | 0.66 | 0.67 | BUY | [rl_best_test/sample_output.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/test/rl_best_test/sample_output.md) |

### What happened to the stock

The data comes from [test_w.parquet](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sealed/test_w.parquet). Each answers file repeats it. The timeline of the trade is:

1. The call was on 2026-03-03.
2. The model decides at the close of the next trading day, 2026-03-04.
3. The trade buys at the next opening price.
4. The trade sells at the close 60 trading days after the decision, on 2026-05-29.

| Step | Value |
| --- | --- |
| Return of the stock over the 60 trading days, with dividends | +11.3% |
| Average return of the stocks in the same sector | +9.8% |
| **Excess return** = 11.3% − 9.8% | **+1.5%** |
| Daily volatility of the stock | 1.96% |
| **Outcome y** = excess return ÷ (daily volatility × √60) = 0.0149 ÷ (0.0196 × 7.75) | **+0.098** |
| Next quarter's EPS minus the EPS a year earlier | +0.98, which is **up** |

The stock did slightly better than its sector, and its EPS went up. SFT and RL said BUY, so they were right on this call. The other four setups said AVOID. Call only, Pi, SFT and RL forecast EPS up, which was also right. One call proves nothing.

**Why I divide by volatility.** A 1.5% excess return is small for a stock that usually moves 5% in one day. It is large for a stock that usually moves 0.5%. Daily volatility × √60 is the usual price change over 60 trading days. Division by this number puts all stocks on the same scale.

### All 10 example test calls

The table shows p(BUY) of each setup and the real outcome y for the 10 example test calls. These are the first 10 test calls of March 2026, in the fixed order.

| Call | No call | Call only | RAG | Pi | SFT | RL | Outcome y |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 218058 | 0.00 | 0.20 | 0.03 | 0.08 | 0.69 | 0.67 | +0.10 |
| 218368 | 0.00 | 0.13 | 0.32 | 0.16 | 0.70 | 0.66 | −0.68 |
| 207363 | 0.00 | 0.68 | 0.88 | 0.53 | 0.84 | 0.80 | −1.46 |
| 219078 | 0.00 | 0.78 | 0.53 | 0.41 | 0.87 | 0.84 | +1.45 |
| 215434 | 0.00 | 0.04 | 0.04 | 0.08 | 0.51 | 0.25 | −0.52 |
| 213996 | 0.00 | 0.04 | 0.47 | 0.04 | 0.75 | 0.70 | +2.20 |
| 218039 | 0.00 | 0.05 | 0.10 | 0.08 | 0.30 | 0.29 | −1.82 |
| 203377 | 0.82 | 0.15 | 0.95 | 0.03 | 0.81 | 0.80 | −1.22 |
| 202800 | 0.00 | 0.06 | 0.01 | 0.06 | 0.57 | 0.55 | −0.71 |
| 225264 | 0.00 | 0.08 | 0.12 | 0.09 | 0.53 | 0.33 | +0.95 |

The trained models (SFT and RL) give higher p(BUY) values than the other setups. Section 5 uses the RL column to show how the IC is calculated. All measures in sections 5 to 7 use all 800 test calls.

## 4. How I trained the small model

### Opus's answers

Opus received the same prompt as the small model. Opus wrote the same seven lines. It also gave a rating from 0 (much worse than its sector) to 100 (much better), and up to three quotes from the call. Opus read each training call two times:

- one time with a neutral instruction
- one time with an instruction to be skeptical.

Here are the two Opus answers for the first training call. The answers for the first 10 training calls are in [examples/week/opus_labels/answers.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/opus_labels/answers.csv).

| Instruction | Demand | Pricing | Margins | Guidance | Candor | Next EPS | Rating |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Neutral | up | flat | up | none | direct | up | 61 |
| Skeptical | up | flat | flat | none | direct | up | 58 |

Opus chose these quotes:

- *"For the first quarter, chassis shipments were at record levels."*
- *"the supply chain base for all of our products seems extremely strong"*
- *"with the strong demand for the carrier product, we plan to keep it dedicated on carrier production"*

The file [labels_v2.jsonl](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/teacher/labels_v2.jsonl) contains all 2,200 answers from Claude:

- 1,600 Opus answers for 800 training calls, with two instructions each
- 300 Opus answers for the check calls
- 300 answers from Claude Sonnet for the same check calls.

These answers cost approximately $37.

### Training on Opus's answers (SFT)

For each training call, a script makes one target. Here is the target for the first training call:

| Line | Target | Source |
| --- | --- | --- |
| Demand to Candor | up, flat, up, none, direct | Opus's neutral answer |
| Next EPS | up | **The real result**, not Opus's answer |
| Action | **BUY with probability 0.84** | Opus's mean rating (59.5), ranked against the other training calls of the same month |

The read lines come from Opus's neutral answer. The skeptical answer changes only the mean rating.

**How I get 0.84.** The training set has 40 calls in this month. The mean rating of 59.5 is in position 34 of 40, from lowest to highest. So its percentile rank is 34 ÷ 40 = 0.85. Then I subtract half a step, 0.5 ÷ 40 = 0.0125. The result is **0.8375**. The half step keeps each target away from exactly 0 and exactly 1. Thus the model learns to say BUY with 84% confidence, not 100%.

The training text is the prompt, [60332_rag.txt](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/prompts/60332_rag.txt), followed by the target answer:

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

**Two kinds of training calls.** Calls with an even row number ask BUY or AVOID. Calls with an odd row number say that the model already holds the stock, and ask HOLD or SELL. All test calls ask BUY or AVOID.

**The first 10 training calls.** The table shows the real targets. The full targets of all 800 training calls are in [sft_targets.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/training/sft_targets.csv).

| Call | Actions | Demand, Pricing, Margins, Guidance, Candor | Next EPS (real) | Opus ratings | Target for the first action |
| --- | --- | --- | --- | --- | --- |
| 60332 | BUY or AVOID | up, flat, up, none, direct | up | 61, 58 | 0.84 |
| 36055 | HOLD or SELL | flat, flat, down, held, direct | down | 41, 38 | 0.23 |
| 98340 | BUY or AVOID | up, flat, flat, held, direct | down | 56, 53 | 0.56 |
| 64438 | BUY or AVOID | up, up, up, none, direct | up | 62, 61 | 0.92 |
| 4974 | BUY or AVOID | down, up, down, cut, direct | up | 36, 34 | 0.13 |
| 32467 | HOLD or SELL | flat, up, up, none, direct | down | 58, 46 | 0.55 |
| 50400 | BUY or AVOID | up, flat, down, none, direct | up | 34, 36 | 0.09 |
| 83894 | BUY or AVOID | flat, up, up, cut, direct | up | 56, 47 | 0.54 |
| 19678 | BUY or AVOID | up, up, flat, none, direct | down | 57, 54 | 0.67 |
| 13037 | HOLD or SELL | flat, flat, down, none, evasive | not known | 56, 46 | 0.50 |

For call 13037, next quarter's EPS was not known. Thus its target has no Next EPS line, and the loss ignores that line.

**The training loss.** At each of the seven lines, the loss compares the model's probabilities for the allowed values with the target. The loss is cross-entropy. Cross-entropy is small when the model gives a high probability to the target value. The loss uses only these seven positions. It ignores all other text.

**Training settings:**

- The adapter is a LoRA adapter of rank 16. Rank 16 sets the size of the adapter. Training changes only the adapter on the top 8 layers of the small model.
- Each update uses 4 calls.
- The first part uses 400 calls, two times each, at a learning rate of 1e-4. The learning rate sets the size of each change to the adapter.
- The second part continues with 400 more calls, two times each, at a learning rate of 3e-5.

The result is the [SFT adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/sft_grown). The training curve is in [sft_grown.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/train/sft_grown.csv).

**Other SFT variants.** I trained five variants and compared them on the same 300 validation calls. The first 10 training calls of each variant are in [examples/README.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/README.md).

| Variant | Targets |
| --- | --- |
| Read + EPS (the variant that I kept) | Read lines, Next EPS and Action, as shown above |
| Score only | Action only |
| Read | Read lines and Action, but not Next EPS |
| Hard | Action = 1 if the stock really did better than the month's median outcome, otherwise 0 |
| **Placebo** (control) | The read lines and ratings from Opus, shuffled among the calls of the same month. Thus they do not match their call. Next EPS stays real |

### Training with rewards (RL)

RL starts from the SFT model and uses 400 different training calls. For each call, the training does these steps:

1. It makes the model write 4 different answers. The answers differ because the training adds randomness. A temperature of 0.8 sets the amount of randomness.
2. It gives each answer a reward.
3. It moves the read lines toward the best answer of the four.
4. It moves p(BUY) toward the value that the reward prefers. It also keeps the model close to the start model.

**The four rewards:**

| Reward | Formula | What it rewards |
| --- | --- | --- |
| Trade | p(BUY) × (reward if BUY) + p(AVOID) × 0 | BUY on stocks that then do better than the month's average |
| Business | log p(the real Next EPS) + log 2 | An EPS forecast that is right and confident. The reward is 0 at 50/50 |
| Both | Business ÷ 0.7 + Trade | Both of the above. The factor 1 ÷ 0.7 (approximately 1.4) gives the business reward more weight |
| **Shuffled** (control) | Trade, with outcomes shuffled among the calls of the same month | Nothing real |

**Example of the trade reward.** This is the fifth RL training call. Its outcome is clear, so the calculation is easy to follow.

| Step | Value |
| --- | --- |
| Outcome y of this stock | −0.240 |
| Mean outcome y of the RL calls in the same month | +0.558 |
| z = outcome y − month mean, limited to the range −3 to +3 | −0.798 |
| Trading cost in the same units = 0.8% ÷ (daily volatility × √60) = 0.008 ÷ 0.313 | 0.026 |
| **Reward if BUY** = z − cost | **−0.824** |
| Reward if AVOID | 0 |
| p(BUY) of the SFT model before RL | 0.25 |
| Expected reward = 0.25 × (−0.824) + 0.75 × 0 | −0.206 |
| **RL target for p(BUY)** = 0.25 × e^(−0.824) ÷ (0.25 × e^(−0.824) + 0.75) | **0.13** |

This stock did much worse than the month's average. Thus RL moves p(BUY) down from 0.25 to 0.13.

**Why this target formula.** The formula multiplies each probability by e^reward. Then it scales the probabilities so that their sum is 1. This target gives the highest expected reward for a model that must stay close to the start model. I first tried a standard RL method, policy gradient. That method always chose AVOID after some steps. Thus I used this exact formula.

**The first 10 RL training calls.** The table shows the trade reward for each call. Its data comes from [examples/week/rl_training/trade/rewards.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/rl_training/trade/rewards.csv). If the model already holds the stock, HOLD gets z and SELL pays a cost of 0.4%. The start p and the target p are for the first action: BUY or HOLD.

| Call | Actions | Outcome y | Month mean | z | Reward of first action | Reward of second action | Start p | Target p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 98177 | HOLD or SELL | +0.404 | +0.420 | −0.017 | −0.017 | −0.014 | 0.67 | 0.67 |
| 26308 | BUY or AVOID | −0.207 | +0.053 | −0.260 | −0.320 | 0 | 0.78 | 0.72 |
| 97154 | BUY or AVOID | +0.471 | +0.072 | +0.398 | +0.358 | 0 | 0.82 | 0.87 |
| 79510 | BUY or AVOID | −0.006 | +0.053 | −0.059 | −0.102 | 0 | 0.47 | 0.45 |
| 4412 | BUY or AVOID | −0.240 | +0.558 | −0.798 | −0.824 | 0 | 0.25 | 0.13 |
| 87478 | BUY or AVOID | −0.899 | +0.053 | −0.952 | −1.006 | 0 | 0.27 | 0.12 |
| 34656 | BUY or AVOID | +0.750 | +0.072 | +0.678 | +0.646 | 0 | 0.25 | 0.38 |
| 10696 | BUY or AVOID | +0.888 | +0.558 | +0.330 | +0.255 | 0 | 0.88 | 0.91 |
| 97223 | HOLD or SELL | −0.136 | +0.072 | −0.209 | −0.209 | −0.018 | 0.28 | 0.24 |
| 30947 | HOLD or SELL | −0.206 | +0.072 | −0.278 | −0.278 | −0.018 | 0.22 | 0.18 |

Each reward got 100 updates (400 calls ÷ 4 calls per update). On the validation calls, the trade reward gave the best result. So it became the [RL adapter](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week/trade). Its training curve is in [trade.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/train/trade.csv).

## 5. How I calculated each number

One script calculates all measures: [score.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/score.py). The script reads two inputs:

- the answers of each setup for each call, in [runs/paper/week/reads](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/reads) (for example [rl_best_test.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/reads/rl_best_test.csv))
- the test outcomes, in [test_w.parquet](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sealed/test_w.parquet).

### The IC

The IC shows whether stocks with a higher p(BUY) did better. The script calculates it in three steps:

1. **Rank in each month.** In each month, the script converts each p(BUY) into a percentile rank among that month's calls, from 0 to 1. It does the same for outcome y. Because of this step, a rise or fall of the whole market does not count as skill.
2. **Correlate.** The script calculates the usual correlation between the two rank columns, over all 800 calls.
3. **Read the result.** 1 means a perfect order. 0 means no relation. −1 means a reversed order.

Here is the calculation for the 10 example test calls of section 3, with the p(BUY) of RL. All 10 calls are from March 2026. Rank 1 is the highest value.

| Call | p(BUY) of RL | Rank | Outcome y | Rank | Difference d | d² |
| --- | --- | --- | --- | --- | --- | --- |
| 218058 | 0.672 | 5 | +0.10 | 4 | 1 | 1 |
| 218368 | 0.658 | 6 | −0.68 | 6 | 0 | 0 |
| 207363 | 0.795 | 3 | −1.46 | 9 | −6 | 36 |
| 219078 | 0.835 | 1 | +1.45 | 2 | −1 | 1 |
| 215434 | 0.253 | 10 | −0.52 | 5 | 5 | 25 |
| 213996 | 0.701 | 4 | +2.20 | 1 | 3 | 9 |
| 218039 | 0.290 | 9 | −1.82 | 10 | −1 | 1 |
| 203377 | 0.798 | 2 | −1.22 | 8 | −6 | 36 |
| 202800 | 0.550 | 7 | −0.71 | 7 | 0 | 0 |
| 225264 | 0.328 | 8 | +0.95 | 3 | 5 | 25 |
| **Sum** | | | | | | **134** |

For one group of calls, this calculation gives the Spearman rank correlation:

```latex
IC = 1 - \frac{6 \sum d^2}{n(n^2 - 1)} = 1 - \frac{6 \times 134}{10 \times 99} = 0.19
```

For these 10 calls, the IC of RL is 0.19. Two calls, 207363 and 203377, had a high p(BUY) but a poor outcome. A group of 10 calls is too small, because one or two calls change its IC a lot. Thus the script ranks each call among all test calls of its month (130 calls in March 2026). Then it calculates one correlation over all 800 calls. Over all 800 calls, the IC of RL is **0.113**.

For a list of hundreds of stocks, an IC of 0.05 to 0.10 is already useful. Small advantages add up over many trades.

### The other measures

| Measure | Calculation | Value for RL on the test set |
| --- | --- | --- |
| **EPS IC** | The same as the IC, with p(Next EPS up) and the real EPS change. 743 of the 800 calls have an EPS change | 0.278 |
| **Stock accuracy** | The percentage of calls where "p(BUY) > 0.5" agrees with "the stock did better than its sector" | 53.6% |
| **EPS accuracy** | The percentage of calls where "p(Next EPS up) > 0.5" agrees with the real EPS direction | 64.2% |
| **Agreement with Opus** | For the 300 check calls, the percentage of read lines that have the same value as Opus's read lines | 70.6% |
| **Return after costs** | The mean excess return of the calls with p(BUY) > 0.5, minus a trading cost of 0.8% | 438 BUY calls with a mean of +1.86%. Result: **+1.06%** |

**Example of agreement with Opus.** On the first 10 check calls, the Call only setup agreed with Opus on 35 of 50 read lines (70%). The SFT and RL setups agreed on 39 of 50 (78%). The answers are in [examples/week/check](https://github.com/nityarai08/earnings-call-distillation/tree/main/examples/week/check) and [examples/week/opus_check/answers.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/opus_check/answers.csv).

### When is one setup better than another?

To compare two setups, the script uses the same calls for both. It calculates the IC of each setup and takes the difference. Then it measures how much chance can change that difference:

1. The test covers 22 weeks. The script picks 22 weeks at random from them. The same week can occur more than one time.
2. It calculates the two ICs and their difference again, with only the calls from the picked weeks.
3. It does steps 1 and 2 2,000 times.
4. It keeps the middle 95% of the 2,000 differences. This is the 95% range.

The script picks whole weeks because calls in the same week share the same market news. If the 95% range includes zero, the two setups are **not distinguishable** with this number of calls.

### Does the action follow the read lines?

I used 200 validation calls. For each call, I ran the model two times:

- one time with the Demand line forced to "up"
- one time with the Demand line forced to "down".

Then I compared the two values of p(BUY). On the first 10 of these calls, the SFT model gave a higher p(BUY) with "up" for all 10 calls. The difference was between 0.02 and 0.58, with a mean of 0.26 ([sft_demand_up/answers.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/rl/sft_demand_up/answers.csv), [sft_demand_down/answers.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/week/rl/sft_demand_down/answers.csv)). Over all 200 calls, the mean was 0.54 with "up" and 0.31 with "down".

## 6. Results

On the 800 test calls, all six setups had an IC of approximately 0.10. Each step adds one method: Call only to RAG, RAG to Pi, RAG to SFT, and SFT to RL. No step gave a gain larger than chance can explain.

![IC of each setup](figures/stock_ic_by_setup.svg)

The call text helps a little. The No call setup (0.075) and the simple rules (0.06 to 0.085) have lower ICs than the setups that read the call.

The simple rules do not use AI:

- **Known-EPS rule:** ranks the stocks by the last EPS change divided by the share price.
- **Word list:** ranks the stocks by the tone of the managers' answers. The tone comes from a fixed list of positive and negative words.
- **Numbers only:** a small decision-tree model that uses only the facts line. I fitted it on the training calls.

![Gain of each step, with its 95% range](figures/ladder_step_gains.svg)

All four 95% ranges include zero. The RL range is the narrowest. But it also includes zero, so RL can be slightly worse or slightly better than SFT.

| Setup or rule | IC | EPS IC | EPS accuracy | Agreement with Opus | BUY calls | Return after costs |
| --- | --- | --- | --- | --- | --- | --- |
| No call | 0.075 | 0.253 | 64.5% | — | 229 | −0.17% |
| Call only | 0.104 | 0.236 | 58.4% | 65.8% | 302 | +0.36% |
| RAG | 0.096 | 0.229 | 56.1% | 67.3% | 276 | −0.38% |
| Pi | 0.104 | 0.179 | 56.0% | 67.3% | 280 | +0.72% |
| SFT | 0.105 | 0.262 | 64.9% | 69.9% | 483 | +0.50% |
| RL | 0.113 | 0.278 | 64.2% | 70.6% | 438 | +1.06% |
| Known-EPS rule | 0.085 | 0.244 | 66.5% | — | 381 | +0.94% |
| Word list | 0.063 | 0.055 | 51.7% | — | 400 | +1.42% |
| Numbers only | 0.069 | — | — | — | 397 | +0.58% |
| BUY on every call | — | — | — | — | 796 | −0.68% |

Section 8 compares Opus with these setups on the same 800 calls. The full tables are in [results.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/results.csv) and [findings.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/findings.csv).

## 7. What I learned

### 1. Training made the answers more like Opus's answers, but did not improve the stock ranking

After training, the small model agreed with Opus more often: 66% for Call only, 71% for RL. Its EPS IC also increased, from 0.236 to 0.278. But each step changed the stock IC by less than chance can explain (section 6).

About half of the gain in agreement comes from one line, Candor:

- Opus writes "direct" for 95% of calls.
- After training, the small model writes "direct" for 98% of calls.
- Agreement on the Candor line increased from 82% to 94%.

Thus about half of the gain is a copy of Opus's most frequent Candor value. It is not a better read of the call.

The controls on 300 validation calls from 2025 show the same result:

- **Placebo.** I trained this model on Opus answers that did not match their calls. The model never chose BUY, because its p(BUY) stayed below 0.5. But the IC uses only the order of p(BUY), so the model still had an IC of 0.099. RAG had 0.094 on the same validation calls. The best real SFT variant had an IC only 0.04 higher, with a 95% range of −0.07 to +0.17.
- **Shuffled reward.** RL with the real trade reward had an IC of 0.109. RL with the shuffled reward had an IC of 0.102.

Training on noise gave approximately the same result as training on real data.

### 2. The action follows the read lines

When I changed the Demand line from "down" to "up", the SFT model's p(BUY) increased from 0.31 to 0.54. It increased for all 200 calls. After RL, the mean increase was 0.21, and p(BUY) increased for 199 of 200 calls. Thus the model does not ignore its own read lines.

The weak part is the read lines themselves. On the 300 check calls, I compared Opus with the small model. For Opus, the stock score is its rating divided by 100.

| Model | IC | EPS accuracy |
| --- | --- | --- |
| Opus | 0.151 | 71.9% |
| Call only | 0.056 | 60.7% |
| RAG | −0.001 | 56.6% |
| SFT | −0.003 | 58.3% |
| RL | 0.031 | 58.0% |

The check calls are from 2023 and 2024. Their facts line does not include the price change around the call. Thus these values differ from the test set values. On the same calls, Opus is clearly better than every setup of the small model.

### 3. The model's own searches made its EPS forecasts worse

In the Pi setup, the EPS IC on the test set was 0.179. In the RAG setup, it was 0.229.

To find the cause, I ran the same Pi tool loop again. But this time the loop received RAG's documents, not the documents that the model chose ([pi_rag_docs.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/reads/pi_rag_docs.csv)). The EPS ICs on 200 shared validation calls were:

- Pi with its own documents: 0.14
- Pi with RAG's documents: 0.22
- RAG: 0.28

With RAG's documents, the EPS IC of the tool loop increased from 0.14 to 0.22. Thus the documents that the small model chooses cause most of the problem.

### 4. No setup made more money than chance can explain

The returns after costs were between −0.68% and +1.42% over 60 trading days. The highest return came from the word list, which uses no AI. Differences this small can come from chance. Section 8 examines the RL return of +1.06% in more detail.

## 8. Signals: what held up and what did not

I looked for every positive signal in the results. Then I checked each signal again on the 500 validation calls from 2025. A real signal should appear in both periods. I chose these checks after I saw the test results. Thus a signal that appears only on the test calls is a lead, not a finding.

The script [signals.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/signals.py) does all the checks. All values and ranges are in [signals.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/signals.csv). Each 95% range comes from the same bootstrap of whole weeks as in section 5.

| Signal | Test calls (2026) | Validation calls (2025) | Status |
| --- | --- | --- | --- |
| The setups forecast the next EPS change | EPS IC 0.18 to 0.28. All ranges are above zero | EPS IC 0.16 to 0.26. All ranges are above zero | **Holds.** But the facts line gives most of it |
| The action follows the read lines | — | p(BUY) 0.31 with Demand "down", 0.54 with "up" (SFT, 200 calls) | **Holds** |
| Opus ranks stocks better than the trained small model | All 800 calls: Opus minus RL +0.057 (+0.007 to +0.111). July and August: +0.026 (−0.037 to +0.211) | Check calls (2023 to 2024): IC 0.151 for Opus, 0.056 or less for the small model | **Holds** on the check calls and on all 800 test calls. Not distinguishable after Opus's knowledge date |
| The model plus two simple rules ranks better than the rules alone | IC gain +0.025 (−0.016 to +0.056) | IC gain +0.026 (−0.036 to +0.097) | **Lead.** Same direction in both periods, but both ranges include zero |
| The setups rank stocks better than chance | IC 0.096 to 0.113. Four of five ranges are above zero | IC −0.012 to +0.007 | **Did not repeat** |
| Training improves the EPS forecast | SFT minus RAG: +0.033 (−0.004 to +0.076) | SFT minus RAG: −0.001 (−0.044 to +0.037) | **Did not repeat** |
| RL earns more after costs than SFT | +0.56% (+0.04% to +1.15%) | −0.15% (−1.30% to +0.87%) | **Did not repeat** |
| RL earns money after costs | +1.06% (−2.28% to +3.93%) | −2.08% (−6.29% to +1.80%) | **Not a signal** |
| Opus's BUY calls earn more than RL's | +1.06% (+0.27% to +2.19%). July and August: +0.80% (+0.07% to +1.46%) | — | **Holds** in both test periods |

### Signals that held up

**1. The setups forecast the next EPS change.** All six setups forecast the direction of the next EPS change better than chance, in both periods. But the No call setup (EPS IC 0.25 on the test calls) and the Known-EPS rule (0.24) did almost as well. Both use only the facts line. Thus most of this signal comes from the last EPS change, not from the call text. The call text did not add a clear gain. Call only minus No call was −0.017 on the test calls and +0.028 on the validation calls. Both ranges include zero.

*For training:* a reward for the EPS line should pay only for what the call adds. One way is to reward the model for the part of the EPS change that the Known-EPS rule does not forecast.

**2. The action follows the read lines.** When I forced the Demand line from "down" to "up", p(BUY) increased for all 200 calls (section 7). Thus the model uses its own read lines to choose BUY or AVOID.

**3. Opus ranks stocks better than the trained small model.** On the same 300 check calls, Opus had an IC of 0.151. The small model had 0.056 or less (section 7). On all 800 test calls, Opus had an IC of 0.170, and RL had 0.113. Opus's BUY calls also earned more (see "Opus on the same test calls" below). Together with signal 2, this shows where to work. The small model already acts on its read lines. Its read lines are the weak part.

*For training:* give the small model Opus's read lines and let it choose only the action. If its IC moves toward 0.15, then the read lines are the part to train. I did not run this test yet.

### Opus on the same test calls

To compare the small model with its teacher, I gave Opus the same 800 test calls. Opus read the same prompt and knowledge block as the RAG setup, with the neutral instruction. Its stock score is its rating divided by 100, as in section 7. The answers are in [labels_v2.jsonl](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/teacher/labels_v2.jsonl) (step `opus_test`). This run cost $9.91.

**A limit of this comparison.** Opus's knowledge runs to approximately June 2026. Thus Opus can know what happened after some calls from March to May. The prompt tells Opus not to use what it remembers, but I cannot check that. The 224 calls from July and August came after that date. For these calls, the comparison is fair.

| Setup or rule | IC, all 800 calls | Return after costs, all 800 calls | IC, July and August | Return after costs, July and August |
| --- | --- | --- | --- | --- |
| **Opus** | **0.170** | **+2.09%** | **0.165** | **+1.81%** |
| No call | 0.075 | −0.17% | 0.181 | +3.09% |
| Call only | 0.104 | +0.36% | 0.193 | +3.49% |
| RAG | 0.096 | −0.38% | 0.119 | +1.91% |
| Pi | 0.104 | +0.72% | 0.144 | +2.83% |
| SFT | 0.105 | +0.50% | 0.137 | +0.75% |
| RL | 0.113 | +1.06% | 0.139 | +1.09% |
| Known-EPS rule | 0.085 | +0.94% | 0.127 | +3.50% |
| Word list | 0.063 | +1.42% | 0.108 | +2.95% |
| BUY on every call | — | −0.68% | — | +0.65% |

The paired gains of Opus over the trained small model, on the same calls:

| Comparison | IC gain, all 800 calls | IC gain, July and August | Return gain, all 800 calls | Return gain, July and August |
| --- | --- | --- | --- | --- |
| Opus minus SFT | +0.065 (+0.004 to +0.120) | +0.029 (−0.049 to +0.131) | +1.62% (+0.52% to +3.11%) | +1.15% (+0.57% to +1.88%) |
| Opus minus RL | +0.057 (+0.007 to +0.111) | +0.026 (−0.037 to +0.211) | +1.06% (+0.27% to +2.19%) | +0.80% (+0.07% to +1.46%) |

What these values show:

- **On all 800 calls, Opus ranked stocks better than SFT and RL.** Both IC ranges are above zero. Opus also had the highest IC of all setups and rules.
- **Opus's BUY calls earned more than the trained small model's BUY calls, in both periods.** All four return ranges are above zero.
- **After Opus's knowledge date, the IC gap is not distinguishable.** In July and August, every setup and rule had a higher IC, also the No call setup. Thus these two months were easier to rank. Opus's IC stayed approximately the same: 0.170 for all calls and 0.165 for July and August. So what Opus remembers does not seem to explain its IC.

Thus the trained small model agrees more with Opus's words (section 7), but it does not yet pick stocks as well as Opus. The gap is clearest in the return of the BUY calls.

### A lead to test on the second test set

**4. The model and the simple rules pick partly different stocks.** In each month, I ranked the calls by RL's p(BUY), by the word list and by the Known-EPS rule. RL's ranks had a correlation of only 0.24 with the word list and 0.37 with the Known-EPS rule. The mean of the three ranks had a higher IC than the mean of the two rule ranks:

- Test calls: 0.117 against 0.092, a gain of +0.025 (−0.016 to +0.056).
- Validation calls (300): 0.085 against 0.058, a gain of +0.026 (−0.036 to +0.097).

The gain had the same size in both periods. But both ranges include zero, so chance can explain it. The second test set has 1,757 calls. It can show whether this gain is real.

### Signals that did not hold up

**5. Stock ranking better than chance.** On the test calls, four setups had an IC range above zero: Call only, Pi, SFT and RL. The top third of calls by p(BUY) beat the bottom third by 4% to 6% over 60 sessions. On the 500 validation calls, the five setups had ICs between −0.012 and +0.007. RL had 0.109 on its 300 validation calls, but its range includes zero. The placebo had 0.099 on the same calls. Thus the ranking signal depends on the period.

**6. Training improves the EPS forecast.** On the test calls, the EPS IC increased at each training step: 0.229 for RAG, 0.262 for SFT and 0.278 for RL. On the validation calls, the same steps gave −0.001 and +0.006. Both ranges include zero.

**7. RL earns more than SFT.** On the test calls, RL earned 0.56% more than SFT after costs, and the range was just above zero. RL dropped 49 of SFT's BUY calls. These calls did 3.5% worse than their sectors, on average. On the validation calls, RL earned 0.15% less than SFT. This check is one of many that I ran after I saw the results. Thus one range above zero is not enough.

**8. RL earns money after costs.** RL earned +1.06% on the test calls. The range was −2.28% to +3.93%. Most of the gain came from one month: in April 2026, its 51 BUY calls earned +7.25%. In March and May, its BUY calls lost money after costs. On the validation calls, the same RL model earned −2.08%. The word list, which uses no AI, earned +1.42% on the test calls.

## 9. Conclusions

**Training the small model on Opus's answers made its answers more similar to Opus's answers. It did not yet make the small model a better stock picker. On the same test calls, Opus still picked better stocks. The signals that held up show that the read lines are the part to improve.**

What I recommend:

- **Always run a placebo.** Without the shuffled controls, the gain in agreement with Opus could look like a real gain.
- **Check which lines changed.** "Agrees more with the teacher" can mean "copies the teacher's most frequent answer".
- **Check that the action follows the read lines.** Here it does. So the next step is to improve the read lines.
- **Check the documents that a small model chooses.** In this test, a fixed search gave better documents.
- **Compare with simple rules.** The one-line Known-EPS rule had an IC only 0.03 lower than the best setup.

**Limits of this study:**

- **The test set is small.** With 800 calls, "not distinguishable" means "no difference that this test can detect". It does not prove that the difference is zero.
- **The scope is narrow.** I tested one small model, one holding period of 60 trading days and one group of companies.
- **The trained models work only with fixed answer values.** After training on the read lines, the models give almost no probability to the allowed answer words. In the RL trace, " BUY" gets 3.2e-05 and " AVOID" gets 1.5e-05 of the vocabulary. Before training, the two words together get more than 0.9. The two SFT variants that trained only on the Action line keep larger values (0.79 and 0.25). The stock scores compare only the allowed values with each other, so the results are valid. But in free text, the models trained on the read lines would not follow the seven-line format.

**Problems during the run.** Three problems occurred on the laptop:

- One training run failed because its learning rate was too high.
- One RL call had no price outcome, and the missing value stopped the RL training.
- The laptop went to sleep for one hour because I closed its lid while it ran on battery.

The file [FIXES.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/docs/FIXES.md) describes each problem and its fix.

**Next steps for the paper:**

1. Give the small model Opus's read lines, and let it choose only the action. If its IC moves toward Opus's 0.15, train the read lines more.
2. Reward the EPS line only for what the call adds to the facts line.
3. Write down the checks for the second test set (1,757 calls) before I open it. The checks are the model-and-rules mix, RL against SFT on return, Opus against RL, and the IC of each setup. Then run each check one time only. More calls give narrower 95% ranges.

## Files

All files below are in the [repo](https://github.com/nityarai08/earnings-call-distillation). The command `bash reproduce/rescore.sh` calculates all the numbers in section 6 again from these files. It takes about one minute.

| Content | File |
| --- | --- |
| Real examples of every experiment and test: prompts, answers, targets, rewards and outcomes | [examples/README.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/examples/README.md) |
| The full training data: SFT targets and RL reward inputs of every training call | [sft_targets.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/training/sft_targets.csv), [rl_rewards.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/training/rl_rewards.csv) |
| All calls, their sets and their facts lines | [sets.parquet](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sets.parquet) |
| All Opus answers used for training | [labels_v2.jsonl](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/teacher/labels_v2.jsonl) |
| All answers of the small model, one file for each setup | [runs/paper/week/reads](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/reads) |
| Test outcomes, opened at the final step | [test_w.parquet](https://github.com/nityarai08/earnings-call-distillation/blob/main/data/paper/sealed/test_w.parquet) |
| Signal checks on the test and validation calls (section 8) | [signals.csv](https://github.com/nityarai08/earnings-call-distillation/blob/main/runs/paper/week/signals.csv), made by [signals.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/signals.py) |
| Training curves | [runs/paper/week/train](https://github.com/nityarai08/earnings-call-distillation/tree/main/runs/paper/week/train) |
| Trained adapters | [adapters/paper/week](https://github.com/nityarai08/earnings-call-distillation/tree/main/adapters/paper/week) |
| Code for the prompt, the training and the scores | [prompt.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/prompt.py), [train_soft.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/train_soft.py), [score.py](https://github.com/nityarai08/earnings-call-distillation/blob/main/scripts/paper/score.py) |
| The meaning of each column | [DATA.md](https://github.com/nityarai08/earnings-call-distillation/blob/main/docs/DATA.md) |
