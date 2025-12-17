import requests
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime, timezone

API = "http://localhost:9999/v1"
API_KEY = "Rotman"                     # <-- your key
HDRS = {"X-API-key": API_KEY}          # change to X-API-Key if your server needs it

# Tickers
CAD  = "CAD"    # currency instrument quoted in CAD
USD  = "USD"    # price of 1 USD in CAD (i.e., USD/CAD)
BULL = "BULL"   # stock in CAD
BEAR = "BEAR"   # stock in CAD
RITC = "RITC"   # ETF quoted in USD

# Per problem statement
FEE_MKT = 0.02           # $/share (market)
REBATE_LMT = 0.01        # $/share (passive) - not used in this baseline
MAX_SIZE_EQUITY = 10000 # per order for BULL/BEAR/RITC
MAX_SIZE_FX = 2500000  # per order for CAD/USD

# Basic risk guardrails (adjust as needed)
MAX_LONG_NET  = 25000
MAX_SHORT_NET = -25000
MAX_GROSS     = 500000
ORDER_QTY     = 5000    # child order size for arb legs

# Cushion to beat fees & slippage.
# 3 legs with market orders => ~0.06 CAD/sh cost; add a bit more for safety.
ARB_THRESHOLD_CAD = 0.17
LEADLAG_MIN_DIFF = 0.0  # minimum stock difference to consider lead/lag (CAD)
# FX arbitrage params
FX_ARB_THRESHOLD = 0.002   # minimum CAD-per-USD discrepancy to act (approx 0.2 cents)
FX_ORDER_QTY = 10000       # USD notional per FX arb order

# OU / plotting settings
OU_WINDOW = 120
EWMA_ALPHA = 0.05
# disable interactive plotting for headless/simulator runs
PLOT_ON = False

# Basket imbalance params
IMBALANCE_THRESHOLD = 0.50   # CAD difference between BULL and BEAR to consider imbalance
IMBALANCE_SCALE_MAX = 2.0    # max multiplier of ORDER_QTY for imbalance sizing
IMBALANCE_DIV = 1.0          # divisor to scale imbalance to sensible multiplier

# Inventory-aware sizing params
INVENTORY_BUFFER = 0.8       # keep to this fraction of exposure limits


# runtime buffers (persist across calls)
spreads = []
running_mean = None
ou_stats = None
# per-tick cache for book mids to reduce API calls
tick_price_cache = {}

# Track last OU entry quantity so exits close the exact amount
ou_entry_qty = None
ou_entry_side = 0
# Trade log CSV
TRADE_LOG_CSV = "etf_trades.csv"
# ensure header exists
try:
    with open(TRADE_LOG_CSV, 'x', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "tick", "action", "side", "z", "qty", "bull_mid", "bear_mid", "ritc_mid_usd", "usd_mid", "notes"])
except FileExistsError:
    pass
# Trading rules for OU z-score
ENTRY_Z = 1.5
EXIT_Z = 0.5
# trade_state: 0 flat, 1 = long_ritc (sell basket, buy RITC), -1 = short_ritc (buy basket, sell RITC)
trade_state = 0
last_trade_tick = None
# Position scaling and exit strategy params
MAX_SCALE = 3.0         # max multiple of ORDER_QTY
STOP_Z = 3.0            # stop-loss z threshold (adverse move)
MAX_HOLD_TICKS = 50     # force exit after this many ticks
MIN_QTY = 1             # minimum order quantity
# OU warmup: number of ticks to wait after the first valid OU fit before entering
MIN_OU_WARMUP_TICKS = 50
# Tick when OU fit first deemed valid/stable
first_valid_ou_tick = None
# Kalman filter for spread mean
class Kalman1D:
    def __init__(self, mean0=0.0, var0=1.0, process_var=1e-4, meas_var=1e-2):
        self.mean = float(mean0)
        self.var = float(var0)
        self.process_var = float(process_var)
        self.meas_var = float(meas_var)

    def update(self, measurement):
        # Predict
        self.var += self.process_var
        # Update
        K = self.var / (self.var + self.meas_var)
        self.mean = self.mean + K * (measurement - self.mean)
        self.var = (1.0 - K) * self.var
        return self.mean, self.var

# global kalman instance for spread
kalman_spread = Kalman1D(mean0=0.0, var0=1.0, process_var=1e-3, meas_var=1e-2)

# Regime switching params
VOL_HIGH_THRESH = 0.02      # rolling spread std above which we consider high volatility
HALF_LIFE_THRESH = 200.0    # half-life above which we treat spread as trending/slow mean-revert

def compute_regime(spreads_window, ou_stats):
    """Return regime name and multipliers for entry/exit sizing.
    Regimes: 'MEAN_REVERT', 'HIGH_VOL', 'TRENDING'
    """
    try:
        # rolling std
        vol = float(np.std(spreads_window)) if spreads_window and len(spreads_window) > 1 else 0.0
        half_life = ou_stats.get('half_life') if ou_stats else float('inf')
        if vol >= VOL_HIGH_THRESH:
            return 'HIGH_VOL'
        if half_life is None or np.isinf(half_life) or half_life > HALF_LIFE_THRESH:
            return 'TRENDING'
        return 'MEAN_REVERT'
    except Exception:
        return 'MEAN_REVERT'


# ---------- Gaussian HMM (lightweight, no external deps) ----------
HMM_STATES = 3
HMM_WINDOW = 120
HMM_EM_ITERS = 15
MIN_HMM_DATA = 30

class GaussianHMM:
    def __init__(self, n_states=3, min_var=1e-6):
        self.n = int(n_states)
        self.min_var = float(min_var)
        self.pi = np.full(self.n, 1.0 / self.n)
        self.A = np.full((self.n, self.n), 1.0 / self.n)
        self.means = np.zeros(self.n)
        self.vars = np.ones(self.n)

    def _log_gauss(self, x, mean, var):
        # log pdf of univariate Gaussian
        return -0.5 * (np.log(2 * np.pi * var) + ((x - mean) ** 2) / var)

    def fit(self, obs, n_iter=HMM_EM_ITERS):
        x = np.asarray(obs, dtype=float)
        T = len(x)
        if T < 2:
            return
        # initialize means by quantiles
        try:
            qs = np.linspace(0, 100, self.n + 2)[1:-1]
            self.means = np.percentile(x, qs)
        except Exception:
            self.means = np.linspace(np.min(x), np.max(x), self.n)
        # init vars
        v = np.var(x) if np.var(x) > 0 else 1.0
        self.vars = np.full(self.n, max(v, self.min_var))
        # init A to slightly favor self-transitions
        self.A = np.eye(self.n) * 0.7 + (0.3 / (self.n - 1)) * (1 - np.eye(self.n))
        self.A = (self.A.T / self.A.sum(axis=1)).T
        self.pi = np.full(self.n, 1.0 / self.n)

        for _ in range(n_iter):
            # forward-backward in log-space with scaling
            B = np.vstack([np.exp(self._log_gauss(x, m, s)) for m, s in zip(self.means, self.vars)])  # shape (K, T)
            B = np.clip(B, 1e-300, None)
            # forward
            alpha = np.zeros((self.n, T))
            alpha[:, 0] = self.pi * B[:, 0]
            c = np.zeros(T)
            c[0] = alpha[:, 0].sum()
            if c[0] == 0:
                c[0] = 1e-300
            alpha[:, 0] /= c[0]
            for t in range(1, T):
                alpha[:, t] = (alpha[:, t - 1] @ self.A) * B[:, t]
                c[t] = alpha[:, t].sum()
                if c[t] == 0:
                    c[t] = 1e-300
                alpha[:, t] /= c[t]

            # backward
            beta = np.zeros((self.n, T))
            beta[:, -1] = 1.0 / c[-1]
            for t in range(T - 2, -1, -1):
                beta[:, t] = (self.A @ (B[:, t + 1] * beta[:, t + 1])) / c[t]

            # gamma and xi
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1)
            xi_sum = np.zeros((self.n, self.n))
            for t in range(T - 1):
                xi = (np.outer(alpha[:, t], (B[:, t + 1] * beta[:, t + 1])) * self.A)
                s = xi.sum()
                if s == 0:
                    s = 1e-300
                xi /= s
                xi_sum += xi

            # M-step
            # update pi
            self.pi = gamma[:, 0] / gamma[:, 0].sum()
            # update A
            row_sums = xi_sum.sum(axis=1)
            for i in range(self.n):
                if row_sums[i] > 0:
                    self.A[i, :] = xi_sum[i, :] / row_sums[i]
            # update means and vars
            for k in range(self.n):
                gk = gamma[k, :]
                denom = gk.sum()
                if denom > 0:
                    self.means[k] = (gk @ x) / denom
                    diff = x - self.means[k]
                    self.vars[k] = max(self.min_var, (gk @ (diff ** 2)) / denom)

    def viterbi(self, obs):
        x = np.asarray(obs, dtype=float)
        T = len(x)
        if T == 0:
            return np.array([], dtype=int)
        logA = np.log(np.clip(self.A, 1e-300, None))
        logpi = np.log(np.clip(self.pi, 1e-300, None))
        logB = np.vstack([self._log_gauss(x, m, s) for m, s in zip(self.means, self.vars)])

        dp = np.zeros((self.n, T))
        ptr = np.zeros((self.n, T), dtype=int)
        dp[:, 0] = logpi + logB[:, 0]
        for t in range(1, T):
            for j in range(self.n):
                seq = dp[:, t - 1] + logA[:, j]
                ptr[j, t] = np.argmax(seq)
                dp[j, t] = seq[ptr[j, t]] + logB[j, t]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(dp[:, -1]))
        for t in range(T - 2, -1, -1):
            states[t] = ptr[states[t + 1], t + 1]
        return states

    def predict_last(self, obs):
        seq = self.viterbi(obs)
        return int(seq[-1]) if len(seq) else None

# instantiate global HMM
hmm_model = GaussianHMM(n_states=HMM_STATES)


def fit_ou(series):
    """Fit discrete-time OU via AR(1) x_{t+1} = alpha + beta*x_t + eps.
    Returns dict with alpha, beta, mu, half_life, resid_std, stationary_std.
    """
    s = np.asarray(series)
    if len(s) < 5:
        return None
    x0 = s[:-1]
    x1 = s[1:]
    X = np.vstack([np.ones_like(x0), x0]).T
    coef, *_ = np.linalg.lstsq(X, x1, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    pred = alpha + beta * x0
    resid = x1 - pred
    resid_std = float(np.std(resid, ddof=1))
    if abs(1 - beta) < 1e-8:
        mu = float('nan')
    else:
        mu = alpha / (1.0 - beta)
    try:
        if beta <= 0 or beta >= 1:
            half_life = float('inf')
        else:
            half_life = -np.log(2.0) / np.log(beta)
    except Exception:
        half_life = float('inf')
    if abs(1 - beta**2) < 1e-8:
        stationary_std = float('nan')
    else:
        stationary_var = resid_std**2 / (1.0 - beta**2)
        stationary_std = float(np.sqrt(max(0.0, stationary_var)))
    return {
        'alpha': alpha,
        'beta': beta,
        'mu': mu,
        'half_life': half_life,
        'resid_std': resid_std,
        'stationary_std': stationary_std
    }


def init_spread_plot():
    global fig, ax, line_spread, line_mean, line_upper, line_lower
    plt.ion()
    fig, ax = plt.subplots(figsize=(8,4))
    line_spread, = ax.plot([], [], label='spread (NAV - ETF)')
    line_mean, = ax.plot([], [], label='running mean')
    line_upper, = ax.plot([], [], label='+1 sigma', linestyle='--', color='gray')
    line_lower, = ax.plot([], [], label='-1 sigma', linestyle='--', color='gray')
    ax.legend()
    ax.set_xlabel('Tick')
    ax.set_ylabel('CAD')
    ax.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

# --------- SESSION ----------
s = requests.Session()
s.headers.update(HDRS)

# --------- HELPERS ----------
def get_tick_status():
    # Gets simulation status (active or stopped) for the tick
    r = s.get(f"{API}/case")
    r.raise_for_status()
    j = r.json()
    return j["tick"], j["status"]

def best_bid_ask(ticker):
    # Returns best bid and ask prices for a ticker
    r = s.get(f"{API}/securities/book", params={"ticker": ticker})
    r.raise_for_status()
    book = r.json()
    'Why choose [0] here, not [1]? Is the price for bids and asks also generated by r'
    bid = float(book["bids"][0]["price"]) if book["bids"] else 0.0
    ask = float(book["asks"][0]["price"]) if book["asks"] else 1e12
    return bid, ask

def positions_map():
    # Tracks current positions (number of shares currently hold for a ticker/instrument), to help risk management
    r = s.get(f"{API}/securities") # after switching /positions to /securities, no error popup.
    r.raise_for_status()
    out = {p["ticker"]: int(p.get("position", 0)) for p in r.json()}
    for k in (BULL, BEAR, RITC, USD, CAD):
        out.setdefault(k, 0)
    return out


def get_mid_prices(ticker):
    """Return (bid, ask) using per-tick cache if available to reduce API calls."""
    try:
        if ticker in tick_price_cache:
            return tick_price_cache[ticker]
    except Exception:
        pass
    bid, ask = best_bid_ask(ticker)
    try:
        tick_price_cache[ticker] = (bid, ask)
    except Exception:
        pass
    return bid, ask


def _compute_close_qtys(side, intended_qty):
    """Compute safe close quantities for BULL, BEAR, RITC based on current positions.
    side: 1 means we are long RITC (sell basket, buy RITC) so to close we BUY basket and SELL RITC.
    side: -1 means we are short RITC (buy basket, sell RITC) so to close we SELL basket and BUY RITC.
    Returns tuple (q_bull, q_bear, q_ritc) of non-negative ints to use for closing orders.
    """
    try:
        pos = positions_map()
        # intended_qty is the intended RITC leg size (shares)
        intended = int(intended_qty) if intended_qty is not None else int(min(ORDER_QTY, MAX_SIZE_EQUITY))
        # how many RITC shares can we actually close based on current position
        ritc_pos = pos.get(RITC, 0)
        if side == 1:
            # we expect to be long RITC; to close we SELL RITC up to our long position
            q_ritc = min(intended, max(0, ritc_pos))
            # for basket legs: we previously SOLD basket (short BULL/BEAR), so to close we BUY
            bull_pos = pos.get(BULL, 0)
            bear_pos = pos.get(BEAR, 0)
            q_bull = min(intended, max(0, -bull_pos))
            q_bear = min(intended, max(0, -bear_pos))
        else:
            # side == -1: we expect to be short RITC; to close we BUY RITC up to abs(short)
            q_ritc = min(intended, max(0, -ritc_pos))
            # for basket legs: we previously BOUGHT basket (long BULL/BEAR), so to close we SELL
            bull_pos = pos.get(BULL, 0)
            bear_pos = pos.get(BEAR, 0)
            q_bull = min(intended, max(0, bull_pos))
            q_bear = min(intended, max(0, bear_pos))
        return int(q_bull), int(q_bear), int(q_ritc)
    except Exception:
        return 0, 0, 0


def log_trade(action, tick, side, z_val, qty, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes=""):
    try:
        ts = datetime.now(timezone.utc).isoformat()
    except Exception:
        ts = datetime.utcnow().isoformat()
    try:
        with open(TRADE_LOG_CSV, 'a', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([ts, tick, action, side, f"{z_val:.6f}" if z_val is not None else "", qty, f"{bull_mid:.6f}" if bull_mid is not None else "", f"{bear_mid:.6f}" if bear_mid is not None else "", f"{ritc_mid_usd:.6f}" if ritc_mid_usd is not None else "", f"{usd_mid:.6f}" if usd_mid is not None else "", notes])
    except Exception:
        # don't fail trading if logging fails
        pass


def micro_edge_adjustment(edge, sigma=None, z_val=None):
    """Apply a small conservative buffer to a raw edge value (CAD).
    The buffer accounts for fees/slippage and optionally increases with spread volatility (sigma)
    or signal magnitude (z_val). Returns adjusted_edge = edge - buffer.
    """
    # base buffer: approximate three market legs fees/slippage
    base = FEE_MKT * 3.0
    # volatility contribution (small): scale sigma to CAD buffer
    vol_buf = 0.0
    try:
        if sigma is not None and not np.isnan(sigma):
            vol_buf = 0.01 * float(abs(sigma))
    except Exception:
        vol_buf = 0.0
    # z contribution (small) to require slightly larger edge for larger signals
    z_buf = 0.0
    try:
        if z_val is not None:
            z_buf = 0.001 * abs(z_val)
    except Exception:
        z_buf = 0.0
    buffer = base + vol_buf + z_buf
    return edge - buffer


def inventory_aware_qty(desired_qty, ticker=None, side=1):
    """Reduce desired_qty if current positions/exposure would exceed limits.
    side: +1 for buy (long), -1 for sell (short)
    Returns adjusted qty (int).
    """
    try:
        pos = positions_map()
        # compute current net exposure in shares (simple sum across equities)
        net_equity_shares = pos.get(BULL, 0) + pos.get(BEAR, 0)
        # rough CAD exposure estimate using mid prices (use cached calls)
        bull_mid_bid, bull_mid_ask = get_mid_prices(BULL)
        bear_mid_bid, bear_mid_ask = get_mid_prices(BEAR)
        bull_mid = 0.5 * (bull_mid_bid + bull_mid_ask)
        bear_mid = 0.5 * (bear_mid_bid + bear_mid_ask)
        ritc_mid_bid, ritc_mid_ask = get_mid_prices(RITC)
        usd_bid, usd_ask = get_mid_prices(USD)
        ritc_mid_usd = 0.5 * (ritc_mid_bid + ritc_mid_ask)
        usd_mid = 0.5 * (usd_bid + usd_ask)
        ritc_equiv = pos.get(RITC, 0) * ritc_mid_usd * (usd_mid if usd_mid else 1.0)
        bull_equiv = pos.get(BULL, 0) * (bull_mid if bull_mid else 0)
        bear_equiv = pos.get(BEAR, 0) * (bear_mid if bear_mid else 0)
        gross = abs(bull_equiv) + abs(bear_equiv) + abs(ritc_equiv)
        # allowed gross after buffer
        allowed_gross = MAX_GROSS * INVENTORY_BUFFER
        if gross >= allowed_gross:
            return 0
        # scale desired_qty so that gross doesn't exceed allowed_gross
        # approximate added exposure of this qty (use ticker mid price)
        add_equiv = 0
        if ticker == RITC:
            add_equiv = desired_qty * ritc_mid_usd * (usd_mid if usd_mid else 1.0)
        elif ticker == BULL:
            add_equiv = desired_qty * (bull_mid if bull_mid else 0)
        elif ticker == BEAR:
            add_equiv = desired_qty * (bear_mid if bear_mid else 0)
        else:
            add_equiv = 0
        if gross + abs(add_equiv) > allowed_gross:
            # scale down proportionally
            max_add = max(0, allowed_gross - gross)
            if add_equiv <= 0:
                return 0
            scale = max_add / abs(add_equiv)
            q = max(0, int(desired_qty * scale))
            return q
        return int(desired_qty)
    except Exception:
        return int(desired_qty)

def place_mkt(ticker, action, qty): 
    # Sends Market orders; price param is ignored by most RIT cases when type=MARKET
    return s.post(f"{API}/orders",
                  params={"ticker": ticker, "type": "MARKET",
                          "quantity": int(qty), "action": action}).ok

def within_limits():
    # Simple gross/net guard using equity legs only
    try:
        pos = positions_map()
        # Convert RITC (USD) position to CAD equivalent using mid prices
        ritc_bid_usd, ritc_ask_usd = get_mid_prices(RITC)
        usd_bid, usd_ask = get_mid_prices(USD)
        # mid prices
        ritc_mid_usd = 0.5 * (ritc_bid_usd + ritc_ask_usd) if (ritc_bid_usd and ritc_ask_usd) else None
        usd_mid = 0.5 * (usd_bid + usd_ask) if (usd_bid and usd_ask) else None
        ritc_pos = pos.get(RITC, 0)
        if ritc_mid_usd is not None and usd_mid is not None:
            ritc_equiv_cad = ritc_pos * ritc_mid_usd * usd_mid
        else:
            ritc_equiv_cad = 0

        b_bid, b_ask = get_mid_prices(BULL)
        bo_bid, bo_ask = get_mid_prices(BEAR)
        bull_equiv = pos.get(BULL, 0) * (0.5 * (b_bid + b_ask))
        bear_equiv = pos.get(BEAR, 0) * (0.5 * (bo_bid + bo_ask))

        net = bull_equiv + bear_equiv + ritc_equiv_cad
        gross = abs(bull_equiv) + abs(bear_equiv) + abs(ritc_equiv_cad)
        return (gross < MAX_GROSS) and (MAX_SHORT_NET < net < MAX_LONG_NET)
    except Exception:
        # if anything goes wrong, be conservative and return False
        return False

def accept_active_tender_offers():
    # Retrieve active tender offers from the RIT API, and accept the offer
    r = s.get(f"{API}/tenders")  # replace with the correct endpoint
    r.raise_for_status()
    offers = r.json()
        
    if offers:
        tender_id = offers[0]['tender_id']
        price = offers[0]['price']
        if offers[0]['is_fixed_bid']:
            resp = s.post(f"{API}/tenders/{tender_id}")
        else:
            resp = s.post(f"{API}/tenders/{tender_id}", params={"price": price})
        return print("Tender Offer Accepted:", resp.ok)
    print("No active tenders")


def get_active_tenders():
    try:
        r = s.get(f"{API}/tenders")
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def tender_relative_value(bull_mid, bear_mid, ritc_mid_cad, usd_mid, current_tick):
    """Evaluate active tenders and execute relative-value trades when profitable.
    Strategy:
      - If tender is offering to buy ETF at a price > NAV - fees/slippage, sell ETF into tender (accept) and hedge by buying basket or appropriate legs.
      - If tender is offering to sell ETF at a price < NAV + fees, buy ETF and hedge by selling basket.
    This function is conservative: will only act when within_limits() and when micro_edge_adjustment shows clear buffer.
    """
    offers = get_active_tenders()
    if not offers:
        return False
    acted = False
    for off in offers:
        try:
            tender_id = off.get('tender_id')
            t_price = float(off.get('price', 0.0))
            is_bid = off.get('is_fixed_bid', True)
            # compare tender price (assume CAD per ETF share) to NAV per ETF share
            if bull_mid is None or bear_mid is None:
                continue
            nav_mid = bull_mid + bear_mid
            # if tender is buying ETF (we can tender into it)
            if is_bid:
                # If tender pays more than NAV plus buffer, create ETF (buy RITC) and accept tender
                diff = t_price - nav_mid
                adj = micro_edge_adjustment(diff)
                if adj > ARB_THRESHOLD_CAD and within_limits():
                    # buy basket to create ETF equivalent: buy BULL & BEAR then accept tender
                    q = min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb = inventory_aware_qty(q, ticker=BULL)
                    qs = inventory_aware_qty(q, ticker=BEAR)
                    if qb > 0 and qs > 0:
                        place_mkt(BULL, "BUY", qb)
                        place_mkt(BEAR, "BUY", qs)
                        # accept tender
                        try:
                            resp = s.post(f"{API}/tenders/{tender_id}")
                            log_trade("TENDER_ACCEPT", current_tick, 0, None, q, bull_mid, bear_mid, ritc_mid_cad, usd_mid, notes=f"accept bid {t_price}")
                            print(f"Tender RV: bought basket qb={qb},qs={qs} and accepted tender id={tender_id} price={t_price}")
                            acted = True
                        except Exception:
                            pass
            else:
                # tender selling ETF to market at t_price (we can buy); if t_price < NAV - buffer, buy tender and hedge
                diff = nav_mid - t_price
                adj = micro_edge_adjustment(diff)
                if adj > ARB_THRESHOLD_CAD and within_limits():
                    q = min(ORDER_QTY, MAX_SIZE_EQUITY)
                    # accept tender (buy) via API - depending on API semantics this may differ
                    try:
                        resp = s.post(f"{API}/tenders/{tender_id}")
                        # hedge by selling basket
                        qb = inventory_aware_qty(q, ticker=BULL)
                        qs = inventory_aware_qty(q, ticker=BEAR)
                        if qb > 0 and qs > 0:
                            place_mkt(BULL, "SELL", qb)
                            place_mkt(BEAR, "SELL", qs)
                        log_trade("TENDER_ACCEPT", current_tick, 0, None, q, bull_mid, bear_mid, ritc_mid_cad, usd_mid, notes=f"accept ask {t_price}")
                        print(f"Tender RV: accepted tender ask id={tender_id} price={t_price} and hedged by selling basket")
                        acted = True
                    except Exception:
                        pass
        except Exception:
            continue
    return acted

# --------- CORE LOGIC ----------
def step_once():
    global spreads, running_mean, ou_stats, trade_state, last_trade_tick, ou_entry_qty, ou_entry_side, first_valid_ou_tick, tick_price_cache
    # clear per-tick cache at start
    try:
        tick_price_cache.clear()
    except Exception:
        pass
    # Get executable prices
    bull_bid, bull_ask = best_bid_ask(BULL)
    bear_bid, bear_ask = best_bid_ask(BEAR)
    ritc_bid_usd, ritc_ask_usd = best_bid_ask(RITC)
    usd_bid, usd_ask = best_bid_ask(USD)   # USD quoted in CAD (USD/CAD)

    # populate cache with values we've already fetched this tick
    try:
        tick_price_cache[BULL] = (bull_bid, bull_ask)
        tick_price_cache[BEAR] = (bear_bid, bear_ask)
        tick_price_cache[RITC] = (ritc_bid_usd, ritc_ask_usd)
        tick_price_cache[USD] = (usd_bid, usd_ask)
    except Exception:
        pass

    # Convert RITC to CAD using USD book
    ritc_bid_cad = ritc_bid_usd * usd_bid
    ritc_ask_cad = ritc_ask_usd * usd_ask
    ritc_mid_usd = 0.5 * (ritc_bid_usd + ritc_ask_usd)
    usd_mid = 0.5 * (usd_bid + usd_ask)

    # Compute mids for lead/lag detection
    bull_mid = 0.5 * (bull_bid + bull_ask) if (bull_bid and bull_ask) else None
    bear_mid = 0.5 * (bear_bid + bear_ask) if (bear_bid and bear_ask) else None
    ritc_mid_cad = 0.5 * (ritc_bid_cad + ritc_ask_cad)

    stock_diff = None
    ritc_vs_basket = None
    if bull_mid is not None and bear_mid is not None:
        stock_diff = bull_mid - bear_mid
        # ritc compared to simple basket mid (sum of mids)
        ritc_vs_basket = ritc_mid_cad - (bull_mid + bear_mid)

    # --- compute NAV-ETF spread and update running stats ---
    try:
        if bull_mid is not None and bear_mid is not None and ritc_mid_cad is not None:
            nav_mid = bull_mid + bear_mid
            # implied USD/CAD from NAV and RITC USD price: CAD per USD
            implied_usd = None
            try:
                if ritc_mid_usd and ritc_mid_usd > 1e-12:
                    implied_usd = nav_mid / ritc_mid_usd
            except Exception:
                implied_usd = None
            spread = nav_mid - ritc_mid_cad
            # append to buffers
            spreads.append(spread)
            # running EWMA mean
            if running_mean is None:
                running_mean = float(spread)
            else:
                running_mean = (1.0 - EWMA_ALPHA) * running_mean + EWMA_ALPHA * float(spread)

            # fit OU on recent window
            recent = spreads[-OU_WINDOW:]
            fit = fit_ou(recent)
            if fit is not None:
                ou_stats = fit
                # print concise OU stats
                print(f"OU fit: mu={fit['mu']:.4f} half_life={fit['half_life']:.2f} resid_std={fit['resid_std']:.4f}")

            # update live plot
            if PLOT_ON:
                try:
                    if 'fig' not in globals():
                        init_spread_plot()
                    xs = list(range(len(spreads)))
                    line_spread.set_data(xs, spreads)
                    mean_series = [running_mean] * len(spreads) if running_mean is not None else [0]*len(spreads)
                    line_mean.set_data(xs, mean_series)
                    sigma = ou_stats['stationary_std'] if ou_stats and ou_stats.get('stationary_std') is not None else (np.std(spreads) if len(spreads)>1 else 0)
                    upper = [running_mean + sigma] * len(spreads)
                    lower = [running_mean - sigma] * len(spreads)
                    line_upper.set_data(xs, upper)
                    line_lower.set_data(xs, lower)
                    ax.relim(); ax.autoscale_view()
                    fig.canvas.draw(); fig.canvas.flush_events()
                except Exception:
                    pass
        else:
            spread = None
    except Exception:
        spread = None

    # ---------- FX-sided arbitrage ----------
    try:
        # implied_usd computed above if available
        if 'implied_usd' in locals() and implied_usd is not None and usd_mid is not None:
            # difference in CAD per USD
            usd_diff = implied_usd - usd_mid
            adj_usd_diff = micro_edge_adjustment(usd_diff, sigma=(ou_stats.get('stationary_std') if ou_stats else None), z_val=None)
            # if adjusted difference exceeds threshold, trade USD
            if adj_usd_diff >= FX_ARB_THRESHOLD and within_limits():
                q_fx = min(FX_ORDER_QTY, MAX_SIZE_FX)
                # implied says USD should be stronger => buy USD
                ok = place_mkt(USD, "BUY", q_fx)
                log_trade("FX_ENTRY", current_tick, 0, None, q_fx, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes=f"FX buy implied>market diff={usd_diff:.6f}")
                print(f"FX Arbitrage: implied_usd {implied_usd:.6f} > market {usd_mid:.6f} (diff={usd_diff:.6f}, adj={adj_usd_diff:.6f}) -> BUY USD qty={q_fx}")
            elif adj_usd_diff <= -FX_ARB_THRESHOLD and within_limits():
                q_fx = min(FX_ORDER_QTY, MAX_SIZE_FX)
                ok = place_mkt(USD, "SELL", q_fx)
                log_trade("FX_ENTRY", current_tick, 0, None, q_fx, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes=f"FX sell implied<market diff={usd_diff:.6f}")
                print(f"FX Arbitrage: implied_usd {implied_usd:.6f} < market {usd_mid:.6f} (diff={usd_diff:.6f}, adj={adj_usd_diff:.6f}) -> SELL USD qty={q_fx}")
    except Exception:
        pass

    # ---------- OU z-score trading rules ----------
    try:
        # try to get current tick for logging (non-fatal)
        try:
            current_tick, _ = get_tick_status()
        except Exception:
            current_tick = None

        if spread is not None and ou_stats and ou_stats.get('stationary_std'):
            sigma = ou_stats.get('stationary_std')
            mu = ou_stats.get('mu')
            if sigma and not np.isnan(sigma) and sigma > 1e-8:
                # update kalman filter for spread smoothing
                try:
                    kalman_spread.meas_var = max(1e-6, sigma**2)
                    kmean, kvar = kalman_spread.update(spread)
                except Exception:
                    kmean, kvar = kalman_spread.mean, kalman_spread.var
                # use kalman filtered mean for z-score centering (more stable)
                mu_used = kmean if kmean is not None else mu
                z = (spread - mu_used) / sigma
                # compute default scaled quantity based on z magnitude
                def scaled_qty_from_z(z_val, ticker=None, max_scale=MAX_SCALE):
                    scale = min(max_scale, max(1.0, abs(z_val) / ENTRY_Z))
                    qn = max(MIN_QTY, int(ORDER_QTY * scale))
                    qn = min(qn, MAX_SIZE_EQUITY)
                    # apply inventory-aware sizing
                    qn = inventory_aware_qty(qn, ticker=ticker)
                    return qn

                # OU warm-up / stability gating
                half_life = ou_stats.get('half_life') if ou_stats else None
                # record first valid OU tick when we have a finite half-life
                if first_valid_ou_tick is None:
                    if half_life is not None and not (np.isinf(half_life) or np.isnan(half_life)):
                        if current_tick is not None:
                            first_valid_ou_tick = current_tick
                            print(f"OU: first valid fit recorded at tick {current_tick}; waiting {MIN_OU_WARMUP_TICKS} ticks before entries")
                # determine if warmup is still active (we still collect data but do not open OU entries)
                ou_warmup_active = False
                can_enter_ou = False
                if first_valid_ou_tick is None:
                    ou_warmup_active = True
                else:
                    if current_tick is None:
                        ou_warmup_active = True
                    else:
                        ticks_since = current_tick - first_valid_ou_tick
                        if ticks_since < MIN_OU_WARMUP_TICKS:
                            ou_warmup_active = True
                            can_enter_ou = False
                        else:
                            ou_warmup_active = False
                            can_enter_ou = True
                # helpful debug about remaining warmup
                if ou_warmup_active and first_valid_ou_tick is not None and current_tick is not None:
                    ticks_left = max(0, MIN_OU_WARMUP_TICKS - (current_tick - first_valid_ou_tick))
                    print(f"OU warmup active: {ticks_left} ticks remaining before entries")

                # regime switching (z-score regime): prefer HMM when enough data
                regime = compute_regime(recent, ou_stats)
                try:
                    if len(recent) >= MIN_HMM_DATA:
                        hmm_model.fit(recent)
                        last_state = hmm_model.predict_last(recent)
                        if last_state is not None:
                            # extract observations assigned to that state to assess half-life/vol
                            states = hmm_model.viterbi(recent)
                            subseq = [recent[i] for i in range(len(recent)) if states[i] == last_state]
                            if len(subseq) >= 5:
                                sub_fit = fit_ou(subseq)
                                if sub_fit is not None:
                                    if sub_fit.get('half_life') is not None and not np.isinf(sub_fit.get('half_life')) and sub_fit.get('half_life') > HALF_LIFE_THRESH:
                                        regime = 'TRENDING'
                                    elif sub_fit.get('stationary_std') is not None and sub_fit.get('stationary_std') > VOL_HIGH_THRESH:
                                        regime = 'HIGH_VOL'
                                    else:
                                        regime = 'MEAN_REVERT'
                except Exception:
                    pass
                entry_z_eff = ENTRY_Z
                exit_z_eff = EXIT_Z
                max_scale_eff = MAX_SCALE
                if regime == 'HIGH_VOL':
                    entry_z_eff = ENTRY_Z * 1.5
                    exit_z_eff = EXIT_Z * 1.5
                    max_scale_eff = max(1.0, MAX_SCALE * 0.7)
                elif regime == 'TRENDING':
                    # trending / slow mean reversion: avoid OU mean-revert entries
                    can_enter_ou = False

                # entry logic (use scaled sizing)
                if trade_state == 0 and within_limits():
                    # enforce OU warmup gating
                    if not can_enter_ou:
                        # skip entries until warmup satisfied
                        pass
                    elif z > entry_z_eff:
                        q_entry = scaled_qty_from_z(z, ticker=RITC)
                        # spread high -> NAV > ETF -> ETF undervalued: SELL basket, BUY RITC
                        # sell basket (BULL/BEAR) sized by inventory for equities
                        q_b = inventory_aware_qty(q_entry, ticker=BULL)
                        q_s = inventory_aware_qty(q_entry, ticker=BEAR)
                        place_mkt(BULL, "SELL", q_b)
                        place_mkt(BEAR, "SELL", q_s)
                        place_mkt(RITC, "BUY", q_entry)
                        trade_state = 1
                        ou_entry_qty = q_entry
                        ou_entry_side = 1
                        last_trade_tick = current_tick
                        traded = True
                        print(f"OU Trade Entry: z={z:.3f} > {ENTRY_Z} -> SELL basket, BUY RITC qty={q_entry} (tick={current_tick})")
                        log_trade("ENTRY", current_tick, ou_entry_side, z, q_entry, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU entry")
                    elif z < -entry_z_eff:
                        q_entry = scaled_qty_from_z(z, ticker=RITC)
                        # spread low -> NAV < ETF -> ETF overvalued: BUY basket, SELL RITC
                        q_b = inventory_aware_qty(q_entry, ticker=BULL)
                        q_s = inventory_aware_qty(q_entry, ticker=BEAR)
                        place_mkt(BULL, "BUY", q_b)
                        place_mkt(BEAR, "BUY", q_s)
                        place_mkt(RITC, "SELL", q_entry)
                        trade_state = -1
                        ou_entry_qty = q_entry
                        ou_entry_side = -1
                        last_trade_tick = current_tick
                        traded = True
                        print(f"OU Trade Entry: z={z:.3f} < -{ENTRY_Z} -> BUY basket, SELL RITC qty={q_entry} (tick={current_tick})")
                        log_trade("ENTRY", current_tick, ou_entry_side, z, q_entry, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU entry")
                # normal exit when z returns within band
                elif trade_state == 1 and abs(z) < exit_z_eff and within_limits():
                    intended = ou_entry_qty if ou_entry_qty is not None else min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb, qs, qr = _compute_close_qtys(1, intended)
                    # execute safe closes based on actual positions
                    if qb > 0:
                        place_mkt(BULL, "BUY", qb)
                    if qs > 0:
                        place_mkt(BEAR, "BUY", qs)
                    if qr > 0:
                        place_mkt(RITC, "SELL", qr)
                    log_trade("EXIT", current_tick, 1, z, qr, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU normal exit")
                    trade_state = 0
                    ou_entry_qty = None
                    ou_entry_side = 0
                    last_trade_tick = current_tick
                    traded = True
                    print(f"OU Trade Exit: z={z:.3f} -> closing long_ritc qty={qr} (tick={current_tick})")
                elif trade_state == -1 and abs(z) < exit_z_eff and within_limits():
                    intended = ou_entry_qty if ou_entry_qty is not None else min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb, qs, qr = _compute_close_qtys(-1, intended)
                    if qb > 0:
                        place_mkt(BULL, "SELL", qb)
                    if qs > 0:
                        place_mkt(BEAR, "SELL", qs)
                    if qr > 0:
                        place_mkt(RITC, "BUY", qr)
                    log_trade("EXIT", current_tick, -1, z, qr, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU normal exit")
                    trade_state = 0
                    ou_entry_qty = None
                    ou_entry_side = 0
                    last_trade_tick = current_tick
                    traded = True
                    print(f"OU Trade Exit: z={z:.3f} -> closing short_ritc qty={qr} (tick={current_tick})")
                # STOP-LOSS: adverse move beyond STOP_Z triggers forced close
                elif trade_state == 1 and z < -STOP_Z and within_limits():
                    intended = ou_entry_qty if ou_entry_qty is not None else min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb, qs, qr = _compute_close_qtys(1, intended)
                    if qb > 0:
                        place_mkt(BULL, "BUY", qb)
                    if qs > 0:
                        place_mkt(BEAR, "BUY", qs)
                    if qr > 0:
                        place_mkt(RITC, "SELL", qr)
                    log_trade("STOP", current_tick, 1, z, qr, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU stop-loss")
                    trade_state = 0
                    ou_entry_qty = None
                    ou_entry_side = 0
                    last_trade_tick = current_tick
                    traded = True
                    print(f"OU Stop-Loss: z={z:.3f} < -{STOP_Z} -> stop loss close qty={qr} (tick={current_tick})")
                elif trade_state == -1 and z > STOP_Z and within_limits():
                    intended = ou_entry_qty if ou_entry_qty is not None else min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb, qs, qr = _compute_close_qtys(-1, intended)
                    if qb > 0:
                        place_mkt(BULL, "SELL", qb)
                    if qs > 0:
                        place_mkt(BEAR, "SELL", qs)
                    if qr > 0:
                        place_mkt(RITC, "BUY", qr)
                    log_trade("STOP", current_tick, -1, z, qr, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU stop-loss")
                    trade_state = 0
                    ou_entry_qty = None
                    ou_entry_side = 0
                    last_trade_tick = current_tick
                    traded = True
                    print(f"OU Stop-Loss: z={z:.3f} > {STOP_Z} -> stop loss close qty={qr} (tick={current_tick})")
                # TIME EXIT: force close after MAX_HOLD_TICKS
                elif trade_state != 0 and last_trade_tick is not None and current_tick is not None and (current_tick - last_trade_tick) > MAX_HOLD_TICKS and within_limits():
                    intended = ou_entry_qty if ou_entry_qty is not None else min(ORDER_QTY, MAX_SIZE_EQUITY)
                    qb, qs, qr = _compute_close_qtys(trade_state, intended)
                    if trade_state == 1:
                        if qb > 0:
                            place_mkt(BULL, "BUY", qb)
                        if qs > 0:
                            place_mkt(BEAR, "BUY", qs)
                        if qr > 0:
                            place_mkt(RITC, "SELL", qr)
                    else:
                        if qb > 0:
                            place_mkt(BULL, "SELL", qb)
                        if qs > 0:
                            place_mkt(BEAR, "SELL", qs)
                        if qr > 0:
                            place_mkt(RITC, "BUY", qr)
                    log_trade("TIME_EXIT", current_tick, ou_entry_side if ou_entry_side else trade_state, z, qr, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes="OU max hold exit")
                    trade_state = 0
                    ou_entry_qty = None
                    ou_entry_side = 0
                    last_trade_tick = current_tick
                    traded = True
                    print(f"OU Time Exit: held > {MAX_HOLD_TICKS} ticks -> force close qty={qr} (tick={current_tick})")
    except Exception:
        # keep running even if trading logic errors
        pass

    # Basket executable values in CAD
    basket_sell_value = bull_bid + bear_bid      # what we get if we SELL basket now
    basket_buy_cost   = bull_ask + bear_ask      # what we pay if we BUY basket now

    # Direction 1: Basket rich vs ETF
    # SELL basket (hit bids), BUY RITC in USD (lift ask) -> compare in CAD
    edge1 = basket_sell_value - ritc_ask_cad

    # Direction 2: ETF rich vs Basket
    # SELL RITC (hit bid in USD), BUY basket (lift asks) -> compare in CAD
    edge2 = ritc_bid_cad - basket_buy_cost
    
    # Tender-aware relative value: try to act on tenders first (creates/hedges ETF for tender capture)
    try:
        try:
            cur_tick, _ = get_tick_status()
        except Exception:
            cur_tick = None
        acted_tender = tender_relative_value(bull_mid, bear_mid, ritc_mid_cad, usd_mid, cur_tick)
        if not acted_tender:
            accept_active_tender_offers()
    except Exception:
        # fallback: still try accepting
        accept_active_tender_offers()

    traded = False
    # optional sigma used for micro-adjustments
    sigma_for_adjust = ou_stats.get('stationary_std') if ou_stats else None
    # ---------- Lead-Lag strategy ----------
    # If stocks show a positive spread (BULL > BEAR) but ETF is non-positive vs basket,
    # then ETF is lagging: SELL basket and BUY RITC (same as basket-rich arb).
    # Converse: if stocks show negative spread and ETF is >= basket, ETF is leading: BUY basket and SELL RITC.
    try:
        if stock_diff is not None and ritc_vs_basket is not None and within_limits():
            adjusted_ritc_vs_basket = micro_edge_adjustment(ritc_vs_basket, sigma=sigma_for_adjust)
            if stock_diff > LEADLAG_MIN_DIFF and adjusted_ritc_vs_basket <= 0:
                q = min(ORDER_QTY, MAX_SIZE_EQUITY)
                place_mkt(BULL, "SELL", q)
                place_mkt(BEAR, "SELL", q)
                place_mkt(RITC, "BUY", q)
                traded = True
                print(f"Lead-Lag: stocks positive (diff={stock_diff:.6f}); ETF lagging (ritc_vs_basket={ritc_vs_basket:.6f}, adj={adjusted_ritc_vs_basket:.6f}) -> SELL basket, BUY RITC")
            elif stock_diff < -LEADLAG_MIN_DIFF and adjusted_ritc_vs_basket >= 0:
                q = min(ORDER_QTY, MAX_SIZE_EQUITY)
                place_mkt(BULL, "BUY", q)
                place_mkt(BEAR, "BUY", q)
                place_mkt(RITC, "SELL", q)
                traded = True
                print(f"Lead-Lag: stocks negative (diff={stock_diff:.6f}); ETF leading (ritc_vs_basket={ritc_vs_basket:.6f}, adj={adjusted_ritc_vs_basket:.6f}) -> BUY basket, SELL RITC")
    except Exception:
        # if anything fails, continue to regular arb logic
        pass
    # ---------- Basket imbalance arbitrage ----------
    try:
        if stock_diff is not None and abs(stock_diff) > IMBALANCE_THRESHOLD and within_limits():
            # if BULL > BEAR by threshold, short BULL and long BEAR (sell expensive, buy cheap)
            mult = min(IMBALANCE_SCALE_MAX, abs(stock_diff) / IMBALANCE_DIV)
            q_base = max(MIN_QTY, int(ORDER_QTY * mult))
            if stock_diff > 0:
                # BULL expensive
                q_bull = inventory_aware_qty(q_base, ticker=BULL)
                q_bear = inventory_aware_qty(q_base, ticker=BEAR)
                if q_bull > 0 and q_bear > 0:
                    place_mkt(BULL, "SELL", q_bull)
                    place_mkt(BEAR, "BUY", q_bear)
                    log_trade("IMBALANCE", current_tick, 0, None, q_base, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes=f"imbalance bull>bear diff={stock_diff:.4f}")
                    print(f"Basket Imbalance: BULL>{BEAR} diff={stock_diff:.4f} -> SELL BULL qty={q_bull}, BUY BEAR qty={q_bear}")
            else:
                # BEAR expensive
                q_bull = inventory_aware_qty(q_base, ticker=BULL)
                q_bear = inventory_aware_qty(q_base, ticker=BEAR)
                if q_bull > 0 and q_bear > 0:
                    place_mkt(BULL, "BUY", q_bull)
                    place_mkt(BEAR, "SELL", q_bear)
                    log_trade("IMBALANCE", current_tick, 0, None, q_base, bull_mid, bear_mid, ritc_mid_usd, usd_mid, notes=f"imbalance bear>bull diff={stock_diff:.4f}")
                    print(f"Basket Imbalance: BEAR>{BULL} diff={stock_diff:.4f} -> BUY BULL qty={q_bull}, SELL BEAR qty={q_bear}")
    except Exception:
        pass
    
    if micro_edge_adjustment(edge1, sigma=sigma_for_adjust) >= ARB_THRESHOLD_CAD and within_limits():
        # Basket rich: sell BULL & BEAR, buy RITC
        q = min(ORDER_QTY, MAX_SIZE_EQUITY)
        place_mkt(BULL, "SELL", q)
        place_mkt(BEAR, "SELL", q)
        place_mkt(RITC, "BUY",  q)
        traded = True

    elif micro_edge_adjustment(edge2, sigma=sigma_for_adjust) >= ARB_THRESHOLD_CAD and within_limits():
        # ETF rich: buy BULL & BEAR, sell RITC
        q = min(ORDER_QTY, MAX_SIZE_EQUITY)
        place_mkt(BULL, "BUY",  q)
        place_mkt(BEAR, "BUY",  q)
        place_mkt(RITC, "SELL", q)
        traded = True

    return traded, edge1, edge2, {
        "bull_bid": bull_bid, "bull_ask": bull_ask,
        "bear_bid": bear_bid, "bear_ask": bear_ask,
        "ritc_bid_usd": ritc_bid_usd, "ritc_ask_usd": ritc_ask_usd,
        "usd_bid": usd_bid, "usd_ask": usd_ask,
        "ritc_bid_cad": ritc_bid_cad, "ritc_ask_cad": ritc_ask_cad
    }

def main():
    tick, status = get_tick_status()
    while status == "ACTIVE":
        traded, e1, e2, info = step_once()
        # Optional: print a lightweight heartbeat every 1s
        print(f"tick={tick} e1={e1:.4f} e2={e2:.4f} ritc_ask_cad={info['ritc_ask_cad']:.4f}")
        sleep(0.5)
        tick, status = get_tick_status()


def fetch_market_snapshot():
    """Fetch current book mids and compute spread without placing any orders.
    Returns a dict with mids and spread info.
    """
    try:
        # clear per-tick cache then populate
        try:
            tick_price_cache.clear()
        except Exception:
            pass
        bull_bid, bull_ask = best_bid_ask(BULL)
        bear_bid, bear_ask = best_bid_ask(BEAR)
        ritc_bid_usd, ritc_ask_usd = best_bid_ask(RITC)
        usd_bid, usd_ask = best_bid_ask(USD)
        # cache
        try:
            tick_price_cache[BULL] = (bull_bid, bull_ask)
            tick_price_cache[BEAR] = (bear_bid, bear_ask)
            tick_price_cache[RITC] = (ritc_bid_usd, ritc_ask_usd)
            tick_price_cache[USD] = (usd_bid, usd_ask)
        except Exception:
            pass

        bull_mid = 0.5 * (bull_bid + bull_ask) if (bull_bid and bull_ask) else None
        bear_mid = 0.5 * (bear_bid + bear_ask) if (bear_bid and bear_ask) else None
        ritc_mid_usd = 0.5 * (ritc_bid_usd + ritc_ask_usd) if (ritc_bid_usd and ritc_ask_usd) else None
        usd_mid = 0.5 * (usd_bid + usd_ask) if (usd_bid and usd_ask) else None
        ritc_mid_cad = (ritc_mid_usd * usd_mid) if (ritc_mid_usd is not None and usd_mid is not None) else None
        nav_mid = None
        spread = None
        if bull_mid is not None and bear_mid is not None and ritc_mid_cad is not None:
            nav_mid = bull_mid + bear_mid
            spread = nav_mid - ritc_mid_cad
        return {
            'bull_mid': bull_mid, 'bear_mid': bear_mid,
            'ritc_mid_usd': ritc_mid_usd, 'usd_mid': usd_mid,
            'ritc_mid_cad': ritc_mid_cad, 'nav_mid': nav_mid, 'spread': spread
        }
    except Exception:
        return {}


def wait_for_case_active(poll_interval=0.5, timeout=None):
    """Block until simulator case status becomes 'ACTIVE'. Returns current tick when active.
    If timeout (seconds) is provided, will raise TimeoutError after timeout.
    """
    import time
    start = time.time()
    while True:
        try:
            tick, status = get_tick_status()
            if status == 'ACTIVE':
                print(f"Case became ACTIVE at tick={tick}")
                return tick
        except Exception:
            # ignore transient connection issues and retry
            pass
        if timeout is not None and (time.time() - start) > timeout:
            raise TimeoutError('wait_for_case_active timed out')
        sleep(poll_interval)


def bootstrap_data_on_start(warmup_ticks=10, poll_interval=0.5):
    """Collect market snapshots for warmup_ticks while not placing orders.
    Appends spreads into global `spreads` and updates running_mean/ou_stats as normal.
    """
    global spreads, running_mean, ou_stats
    collected = 0
    print(f"Bootstrapping market data: collecting {warmup_ticks} snapshots before trading starts")
    while collected < warmup_ticks:
        try:
            # fetch snapshot (populates per-tick cache)
            snap = fetch_market_snapshot()
            # update spreads buffer and running mean similar to step_once
            sp = snap.get('spread')
            if sp is not None:
                spreads.append(sp)
                if running_mean is None:
                    running_mean = float(sp)
                else:
                    running_mean = (1.0 - EWMA_ALPHA) * running_mean + EWMA_ALPHA * float(sp)
                recent = spreads[-OU_WINDOW:]
                fit = fit_ou(recent)
                if fit is not None:
                    ou_stats = fit
            collected += 1
        except Exception:
            pass
        sleep(poll_interval)
    print("Bootstrap complete; entering trading loop")


def start_on_active(warmup_ticks=10, poll_interval=0.5):
    """Wait for case ACTIVE, collect warmup data, then run main trading loop."""
    try:
        wait_for_case_active(poll_interval=poll_interval)
        bootstrap_data_on_start(warmup_ticks=warmup_ticks, poll_interval=poll_interval)
        main()
    except Exception as e:
        print("start_on_active encountered error:", e)


if __name__ == "__main__":
    # auto-start behavior: wait for case ACTIVE, collect warmup snapshots, then run
    # warmup_ticks only applies to data collection; other strategies (FX, tenders, imbalance)
    # will still operate in step_once once main begins.
    start_on_active(warmup_ticks=10, poll_interval=0.5)


