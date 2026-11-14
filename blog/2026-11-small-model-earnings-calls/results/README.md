# Result files

All numbers come from the week run (`runs/paper/week/`), finished 2026-11-13. "Test" means the 800 sealed calls from March to August 2026. "Dev" means the 2025 calls used for tuning; the test outcomes were not opened until the final step.

## Columns

| Column | Meaning |
| --- | --- |
| `setup`, `arm` | The model setup, control or baseline being scored |
| `calls` | Number of earnings calls scored |
| `stock_ic` | Stock-ranking score (IC). The BUY probability and the 60-day sector-adjusted, volatility-scaled return are each turned into a percentile rank within their month. The IC is the correlation of the two ranks across all calls. 0 = coin flip |
| `stock_lo`, `stock_hi` | 95% bootstrap range for `stock_ic` |
| `stock_acc` | Share of calls where BUY/AVOID matched whether the stock beat its sector |
| `eps_ic` | Same rank score, for the "Next EPS: up" probability against next quarter's actual EPS change vs a year earlier |
| `eps_lo`, `eps_hi` | 95% bootstrap range for `eps_ic` |
| `eps_acc` | Share of calls where the EPS direction was right |
| `read_match` | Share of the five read lines (Demand to Candor) that match Opus's read of the same call, on the 300 student-check calls (not the test calls) |
| `book_net` | Average 60-day return above the sector for calls where BUY was more likely, minus a 0.8% round-trip cost, as a fraction (0.0106 = +1.06%) |
| `buy_share` | Share of calls where BUY was the more likely action |
| `mean_p` | Average BUY probability |
| `move` | (faithfulness) BUY probability with Demand forced "up" minus with Demand forced "down" |
| `share_right_way` | (faithfulness) Share of calls where that move was positive |
| `question`, `result`, `verdict` | (findings) The paired comparison, its gain with 95% range, and whether the range excludes zero |

Empty cells mean the score does not apply (for example, a rule with no EPS forecast).
