# RITC_backup — ETF / Basket Arbitrage Bot (RITC Simulator)

Python trading bot designed for the Rotman International Trading Competition (RITC) simulator.
It trades an ETF (`RITC`, quoted in USD) against a two-stock basket (`BULL` + `BEAR`, quoted in CAD),
using FX conversion via `USD` (USD/CAD) and multiple signal layers (spread arb, OU z-score, regime logic, tenders).

## Instruments
- `BULL` (CAD)
- `BEAR` (CAD)
- `RITC` (USD) — ETF
- `USD` (CAD per 1 USD; i.e., USD/CAD)
- `CAD` (currency instrument)

## What the bot does
This script polls the simulator order books and trades when it detects a profitable edge:

1. **ETF vs Basket Arbitrage (CAD)**
   - Compute:
     - Basket executable values: `BULL_bid + BEAR_bid` (sell basket), `BULL_ask + BEAR_ask` (buy basket)
     - Convert ETF USD quotes to CAD using `USD` book
   - Trades:
     - Basket rich: **SELL** `BULL`, **SELL** `BEAR`, **BUY** `RITC`
     - ETF rich: **BUY** `BULL`, **BUY** `BEAR`, **SELL** `RITC`
   - Uses a conservative threshold to cover fees/slippage.

2. **OU Mean-Reversion on NAV–ETF Spread**
   - Tracks spread: `NAV_mid - RITC_mid_CAD`
   - Fits a discrete-time OU model via AR(1) on a rolling window
   - Trades based on z-score with:
     - entry/exit bands
     - stop-loss
     - time-based exit
     - warm-up gating before first entries

3. **Kalman Filter Smoothing**
   - Uses a 1D Kalman filter to stabilize spread mean estimation used in z-scores.

4. **Lightweight Gaussian HMM Regime Detection**
   - A small internal (no external deps) Gaussian HMM is fit on recent spreads
   - Used to adjust or suppress OU entries in “TRENDING” regimes and tighten behavior in high volatility.

5. **FX-Sided Arbitrage**
   - Computes implied USD/CAD from `NAV_mid / RITC_mid_usd`
   - Trades `USD` if implied deviates from market beyond a threshold.

6. **Tender Handling (Relative Value)**
   - Checks active tenders via `/tenders`
   - Attempts to accept tenders only when profitable and hedge appropriately.

7. **Risk Guardrails**
   - Simple gross/net exposure checks before trading
   - Inventory-aware sizing reduces order sizes when near limits

## Requirements
- Python 3.9+ recommended
- Packages:
  - `requests`
  - `numpy`
  - `matplotlib` (optional — plotting currently disabled by default)

Install:
```bash
pip install requests numpy matplotlib
