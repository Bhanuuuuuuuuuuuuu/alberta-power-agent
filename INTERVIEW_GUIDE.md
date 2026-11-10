# Interview Guide: Alberta Battery Trading Agent

Learn this file before any interview where the project is on your resume.
Read each answer, then close the file and say it out loud in your own words.

## The 30 second pitch

> I built an agent that trades a virtual 10 MWh battery in Alberta's power
> market. Every midnight it forecasts tomorrow's 24 hourly prices with XGBoost,
> then a linear program decides when to charge and when to sell, given the
> battery's power, capacity, efficiency losses and wear. I tested it walk
> forward on two and a half years of AESO data it never trained on, and
> compared it to a simple fixed schedule and to perfect foresight, which is
> the most money anyone could have made.

Then the numbers:
> It made 23% more than a fixed schedule ($824k vs $671k over 943 days),
> captured 44% of the perfect foresight maximum, with a Sharpe ratio of 8.4
> and a max drawdown of $4.4k, a third of the simple rule's.

## Your numbers (memorize these)

| | |
|---|---|
| Test period | Jan 2023 to Jul 2025, 943 days, walk forward, monthly retrain |
| Forecast error | $55.76 vs $72.94/MWh MAE, 24% better than naive |
| Spikes | 70% of 1,494 hours above $300 caught, precision 21% |
| P10 to P90 coverage | 65% (target 80%): ranges too narrow, a known weakness |
| Best strategy profit | $824k vs $671k fixed rule (+23%) |
| Share of perfect foresight | 44% (rule: 36%) |
| Sharpe / max drawdown | 8.4 / $4.4k (rule: 7.5 / $11.8k) |
| Expected price vs median forecast | trading on the mean made 13% more ($824k vs $730k) |

## Questions you WILL get

**1. How did you avoid look ahead bias?**
Three ways. (a) Every feature for day D is known at midnight before day D:
only lags of 24 hours or more, and "yesterday" stats are shifted a full day.
(b) Walk forward testing: each month is predicted by a model trained only on
earlier months. (c) A unit test changes day D's prices and checks that none of
day D's features change.

**2. Why not a 1 hour lag? It would be more accurate.**
Because the decision is made at midnight for all 24 hours. When predicting
5 PM, the 4 PM price doesn't exist yet. Using it would make the backtest look
great and fail in real life.

**3. Why XGBoost and not an LSTM?**
Tabular features (hour, lags, yesterday's stats) where gradient boosting is
strong, it trains in seconds so monthly retraining is cheap, and it handles
the heavy tailed prices well. An LSTM is a reasonable next experiment, but it
should have to beat this baseline first.

**4. What's the difference between your P50 and mean forecasts, and why does it matter?**
P50 is the median, the typical price. The mean is the average, and Alberta's
spikes pull it far above the median. Profit is an average over many days, so a
risk neutral trader should plan on the mean. It mattered: planning on the
mean made 13% more money than planning on the median ($824k vs $730k).

**5. What are P10 and P90 for?**
They give a range: the price should land between them about 80% of the time.
Mine landed inside 65% of the time, so the ranges are too narrow, meaning the
model is overconfident, mostly because 2023 and 2024 were more volatile than
the training years. Say this yourself; admitting a weakness and knowing the
fix (conformal calibration, or weighting recent data more) scores points.

**6. Explain the linear program.**
Variables: charge, discharge and state of charge for each of 24 hours.
Objective: maximize price times (discharge minus charge), minus wear cost.
Constraints: power at most 5 MW, energy between 0 and 10 MWh, energy balance
with square root of 0.9 efficiency on each side, at most one full cycle per
day, start and end empty. It's linear, so HiGHS solves it in milliseconds and
the answer is the true optimum for that forecast.

**7. Why does the battery sometimes do nothing?**
A round trip loses 10% and every MWh costs $10 of wear. If tomorrow's
forecast spread is smaller than those costs, trading loses money, so the
optimal plan is to sit idle. There's a test for this.

**8. What is "perfect foresight" and why include it?**
The same LP, given the real prices. Nobody can achieve it, so it's the upper
bound. Saying "my agent captured X% of the maximum possible profit" is more
honest than a raw dollar number, because it adjusts for how volatile that year
was.

**9. Why did profit change so much between years?**
Prices averaged $134/MWh in 2023 and $39 in 2025 after new supply came online.
Less volatility means smaller spreads to trade. This is also why the test
period is hard: the model trained mostly on calmer years then faced 2023 and
then a collapse.

**10. What's the Sharpe ratio here?**
Mean daily profit divided by the standard deviation of daily profit, times the
square root of 365. It measures profit per unit of risk. Batteries that only
trade when the spread is profitable have high Sharpe ratios because they rarely
lose money on a day, so also look at max drawdown and the worst 5% of days.

**11. What would break this in real life?**
Read the Limitations section in README.md. Biggest ones: real Alberta
batteries earn mostly from operating reserves, the model assumes the battery
doesn't move prices, and Alberta's market rules are changing (REM).

**12. What would you do next?**
Add AESO wind and solar forecasts, co optimize with operating reserves, re
optimize during the day, and run it live as paper trading on the AESO API.

**13. Your Sharpe of 8.4 seems too high. Is something wrong?**
No, but it's a fair challenge. It's high because the battery only trades when
the forecast spread beats losses and wear, so most days are small wins. It
leaves out real costs (transmission charges, market fees) and assumes the
battery doesn't move prices. That's why I also report drawdown and the worst
5% of days, and compare against perfect foresight.

## Desk questions (systems, Goldman engineering style)

**14. Walk me through what happens at midnight.**
Nine steps: ingest new hours from AESO and quality check them, settle
yesterday's plan at real prices, check loss limits, retrain if it's a new
month, check the data is fresh and the model current, forecast, optimize with
the spike overlay, run an independent physics check, then save the plan and
the reason to the ledger and send an alert. Every step writes to the audit
log.

**15. How do you know replay mode isn't cheating?**
The replay feed has a clock (`now`) and clips every request at it, so it
can't return a future hour even if the engine asks. The engine code is the
same in replay and live; only the feed object changes. There's a test that
asks the replay feed for future data and checks it's refused.

**16. What does the kill switch do, and why can't it reset itself?**
It trips when yesterday's loss, the 7 day loss or the drawdown passes its
limit. While it's on, every cycle records the day as halted and trades
nothing. Only an operator can reset it, with a note that goes in the audit
log, because a system that resets its own safety switch isn't safe: a human
has to decide whether the loss was bad luck or a bug.

**17. Why check the plan's physics if the optimizer already enforces it?**
Defense in depth. The check recomputes the battery's state from scratch with
separate code. If the optimizer, a refactor or a bad input ever produced an
impossible plan, the desk stops instead of sending it. Real desks never trust
one component for safety.

**18. What's the spike overlay?**
If the spike model says an hour has over 50% odds of a spike, the battery
isn't allowed to buy in that hour, whatever the price forecast says. Buying
into a spike is the most expensive mistake a battery can make, so it's a hard
rule on top of the optimizer.

**19. How do you handle the API being down?**
The fetch raises a FeedError, which is logged and alerted. The cycle then
sees stale data in its input check and skips the day rather than trading on
old information. Next run, ingest picks up from the last stored hour, so the
gap fills itself.

**Desk numbers (memorize):** replayed 577 days as if live, $279k profit vs
$292k for the backtest without risk controls; the controls cost 4% of profit
but cut max drawdown from $3.2k to $3.0k and losing days from 49% to 46%.
Perfect foresight matched the backtest to the dollar ($759,529), which
verifies the desk's settlement code against an independent implementation.

**20. How would you scale this to 50 batteries?**
Move the ledger to Postgres, run one cycle per asset in parallel from a job
queue, share one forecast across assets (it's the same market price), and
add portfolio level risk limits on top of per asset ones. Once the fleet is
big enough to move prices, the price taker assumption breaks and the
optimizer has to model its own price impact.

## Resume bullet

> Built an AI battery trading agent for Alberta's AESO power market in Python:
> XGBoost quantile and spike models cut price forecast error 24% vs baseline;
> a linear programming optimizer (SciPy HiGHS) dispatched a 10 MWh battery,
> earning 23% more than a fixed schedule and capturing 44% of perfect
> foresight profit (Sharpe 8.4) in a 943 day walk forward backtest.

Two line version, once the desk is on the resume:

> Built an AI battery trading desk for Alberta's AESO power market (Python, XGBoost, SciPy LP, SQLite):
> forecasts cut price error 24% and the optimizer beat a fixed schedule by 23% over 943 backtest days.
> Engineered a live paper trading system with an AESO API feed, versioned monthly retraining, pre trade
> risk checks, loss limits with a kill switch, an audit log, Discord alerts and 28 CI tested unit tests.

## Files to read, in order
1. `src/powerbot/features.py`: every line. Question 1 lives here.
2. `src/powerbot/battery.py`: the LP. Draw the constraints on paper.
3. `src/powerbot/forecast.py`: the walk forward loop.
4. `src/powerbot/backtest.py`: how strategies are settled.
5. `tests/test_trading.py`: the look ahead test.

## Self test
Without notes, can you:
- [ ] draw the pipeline from data to dashboard
- [ ] write the LP objective and 4 constraints on a whiteboard
- [ ] explain why the test split is by time, not random
- [ ] give your 3 headline numbers
- [ ] name 2 limitations and how you'd fix them
