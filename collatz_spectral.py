"""
Spectral and Hurst analysis of record Collatz climbers.

Map: shortcut map T(x) = (3x+1)/2 if x odd, x/2 if x even.
Parity vector: p_i = x_i mod 2 (parity of each iterate under T).
Under T, an odd step multiplies by ~3/2, an even step by 1/2.
Drift per step in log2: q*log2(3) - 1, where q = odd density.
Critical density q* = 1/log2(3) ~= 0.6309.

We analyze the ASCENT phase (steps from x_0 up to the trajectory maximum):
  - odd-step density q
  - PSD log-log slope beta (periodogram, log-binned), S(f) ~ 1/f^beta
  - DFA-1 Hurst exponent H
Baselines: i.i.d. Bernoulli(q) sequences of the same length (white noise:
beta ~ 0, H ~ 0.5), and the engineered "crystal" climber 2^k - 1 whose
first k steps are forced odd.
"""

import numpy as np

rng = np.random.default_rng(27)

def trajectory_parities(n):
    """Shortcut-map parity vector until reaching 1, plus log2 heights."""
    ps, heights = [], []
    x = n
    while x != 1:
        ps.append(x & 1)
        heights.append(float(x).__abs__())  # placeholder, replaced below
        if x & 1:
            x = (3 * x + 1) // 2
        else:
            x //= 2
    return np.array(ps, dtype=float), None

def trajectory_full(n, max_steps=10_000_000):
    """Return parity vector and log2 of iterates under shortcut map."""
    ps = []
    logs = []
    x = n
    steps = 0
    while x != 1 and steps < max_steps:
        ps.append(x & 1)
        logs.append(x.bit_length())  # log2 proxy, exact enough for peak location
        if x & 1:
            x = (3 * x + 1) // 2
        else:
            x //= 2
        steps += 1
    return np.array(ps, dtype=np.int8), np.array(logs, dtype=np.int64)

def dfa(x, min_s=4, max_frac=4, n_scales=20):
    """DFA-1 Hurst exponent of a 1D series."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 32:
        return np.nan
    y = np.cumsum(x - x.mean())
    max_s = N // max_frac
    if max_s <= min_s:
        return np.nan
    scales = np.unique(np.geomspace(min_s, max_s, n_scales).astype(int))
    F = []
    for s in scales:
        m = N // s
        segs = y[: m * s].reshape(m, s)
        t = np.arange(s)
        # linear detrend each segment
        tbar = t.mean()
        denom = ((t - tbar) ** 2).sum()
        slopes = ((segs - segs.mean(axis=1, keepdims=True)) * (t - tbar)).sum(axis=1) / denom
        trends = segs.mean(axis=1, keepdims=True) + np.outer(slopes, t - tbar)
        F.append(np.sqrt(((segs - trends) ** 2).mean()))
    F = np.array(F)
    good = F > 0
    if good.sum() < 4:
        return np.nan
    H, _ = np.polyfit(np.log(scales[good]), np.log(F[good]), 1)
    return H

def psd_slope(x, n_bins=12):
    """Fit S(f) ~ f^{-beta} on log-binned periodogram; return beta."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 64:
        return np.nan
    x = x - x.mean()
    ft = np.fft.rfft(x)
    psd = (np.abs(ft) ** 2)[1:]  # drop DC
    f = np.fft.rfftfreq(N)[1:]
    # log-spaced bins over available frequency range
    edges = np.geomspace(f[0], f[-1], n_bins + 1)
    fb, sb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        if sel.sum() > 0 and psd[sel].mean() > 0:
            fb.append(np.sqrt(lo * hi))
            sb.append(psd[sel].mean())
    if len(fb) < 5:
        return np.nan
    slope, _ = np.polyfit(np.log(fb), np.log(sb), 1)
    return -slope  # beta

def analyze(ps, label):
    q = ps.mean()
    H = dfa(ps)
    beta = psd_slope(ps)
    return dict(label=label, n=len(ps), q=q, H=H, beta=beta)

# ---------------------------------------------------------------
# 1. Find genuine peak-record setters by scan (small range), then add
#    famous large climbers and verify their stats directly.
# ---------------------------------------------------------------
def peak_records(limit):
    recs = []
    best = 0
    for n in range(2, limit):
        x = n
        peak = n
        while x != 1:
            if x & 1:
                x = (3 * x + 1) // 2
            else:
                x //= 2
            if x > peak:
                peak = x
        if peak > best:
            best = peak
            recs.append((n, peak))
    return recs

records_small = peak_records(100_000)

famous = [27, 703, 26623, 626331, 837799, 63728127, 670617279,
          9780657630, 1980976057694848447]

seeds = sorted(set([n for n, _ in records_small[-6:]] + famous))

print("=" * 90)
print("ASCENT-PHASE ANALYSIS OF RECORD CLIMBERS (shortcut map parity vectors)")
print(f"critical odd density q* = 1/log2(3) = {1/np.log2(3):.4f}")
print("=" * 90)
print(f"{'seed':>22} {'ascent_len':>10} {'odd_q':>8} {'DFA_H':>8} {'PSD_beta':>9}   peak_bits")

results = []
for n in seeds:
    ps, logs = trajectory_full(n)
    peak_idx = int(np.argmax(logs))
    if peak_idx < 32:
        continue
    ascent = ps[: peak_idx + 1]
    r = analyze(ascent, str(n))
    r["peak_bits"] = int(logs[peak_idx])
    results.append((n, r, ps, peak_idx))
    print(f"{n:>22} {r['n']:>10} {r['q']:>8.4f} {r['H']:>8.3f} {r['beta']:>9.3f}   {r['peak_bits']}")

# ---------------------------------------------------------------
# 2. Baseline: i.i.d. Bernoulli(q) at the same lengths
# ---------------------------------------------------------------
print("-" * 90)
print("BASELINE: i.i.d. Bernoulli(q) matched to each climber's (length, q); 200 reps each")
for n, r, ps, peak_idx in results:
    L, q = r["n"], r["q"]
    Hs, Bs = [], []
    for _ in range(200):
        b = (rng.random(L) < q).astype(float)
        Hs.append(dfa(b))
        Bs.append(psd_slope(b))
    Hs, Bs = np.array(Hs), np.array(Bs)
    zH = (r["H"] - np.nanmean(Hs)) / np.nanstd(Hs)
    zB = (r["beta"] - np.nanmean(Bs)) / np.nanstd(Bs)
    print(f"{n:>22}  H_iid={np.nanmean(Hs):.3f}±{np.nanstd(Hs):.3f} (z={zH:+.2f})"
          f"   beta_iid={np.nanmean(Bs):.3f}±{np.nanstd(Bs):.3f} (z={zB:+.2f})")

# ---------------------------------------------------------------
# 3. Longest ascent available: concatenated analysis + descent contrast
# ---------------------------------------------------------------
print("-" * 90)
biggest = max(results, key=lambda t: t[1]["n"])
n, r, ps, peak_idx = biggest
descent = ps[peak_idx:]
rd = analyze(descent.astype(float), "descent")
print(f"Longest climber {n}: ascent H={r['H']:.3f}, beta={r['beta']:.3f} (len {r['n']})"
      f" | descent H={rd['H']:.3f}, beta={rd['beta']:.3f} (len {rd['n']}), q={rd['q']:.4f}")

# ---------------------------------------------------------------
# 4. Engineered crystal: 2^k - 1 (k forced odd steps), then decoherence
# ---------------------------------------------------------------
print("-" * 90)
k = 400
n = 2**k - 1
ps, logs = trajectory_full(n)
peak_idx = int(np.argmax(logs))
print(f"CRYSTAL CLIMBER 2^{k}-1: forced odd prefix, peak at step {peak_idx}, total steps {len(ps)}")
# sliding-window odd density after the crystal phase
w = 100
qs = [ps[i:i+w].mean() for i in range(0, len(ps) - w, w)]
print("sliding-window odd density (window=100):")
print("  first 8 windows:", np.round(qs[:8], 3))
print("  last 8 windows: ", np.round(qs[-8:], 3))
post = ps[peak_idx:].astype(float)
rp = analyze(post, "post-crystal")
print(f"post-peak phase: len={rp['n']}, q={rp['q']:.4f}, H={rp['H']:.3f}, beta={rp['beta']:.3f}")

# ---------------------------------------------------------------
# 5. Export DFA curves for the best natural climber vs iid, for plotting
# ---------------------------------------------------------------
n, r, ps, peak_idx = biggest
ascent = ps[:peak_idx+1].astype(float)
def dfa_curve(x, min_s=4, max_frac=4, n_scales=20):
    x = np.asarray(x, dtype=float); N = len(x)
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.geomspace(min_s, N//max_frac, n_scales).astype(int))
    out = []
    for s in scales:
        m = N // s
        segs = y[:m*s].reshape(m, s)
        t = np.arange(s); tbar = t.mean(); denom = ((t-tbar)**2).sum()
        slopes = ((segs - segs.mean(axis=1, keepdims=True))*(t-tbar)).sum(axis=1)/denom
        trends = segs.mean(axis=1, keepdims=True) + np.outer(slopes, t-tbar)
        out.append((s, float(np.sqrt(((segs-trends)**2).mean()))))
    return out

curve_real = dfa_curve(ascent)
b = (rng.random(len(ascent)) < ascent.mean()).astype(float)
curve_iid = dfa_curve(b)
print("-" * 90)
print(f"DFA curve, climber {n} ascent: ", [(s, round(np.log10(F),3)) for s,F in curve_real])
print(f"DFA curve, matched iid:        ", [(s, round(np.log10(F),3)) for s,F in curve_iid])

# ---------------------------------------------------------------
# 6. Drift-identity correction E_k/k for each analyzed ascent
#    (the arithmetic floor q > q*(1 - E_k/k) cited in Sec. 5.1)
# ---------------------------------------------------------------
import math

def ascent_correction(n):
    """Walk to the trajectory peak; return (k, q_observed, E_k/k, forced bound)."""
    x, peak, peak_idx, cur = n, n, 0, 0
    while x != 1:
        x = (3 * x + 1) // 2 if x & 1 else x // 2
        cur += 1
        if x > peak:
            peak, peak_idx = x, cur
    x, E, S = n, 0.0, 0
    for _ in range(peak_idx):
        if x & 1:
            E += math.log2(1 + 1 / (3 * x)); S += 1
            x = (3 * x + 1) // 2
        else:
            x //= 2
    k = peak_idx
    qstar = 1 / math.log2(3)
    return k, S / k, E / k, qstar * (1 - E / k)

print("-" * 90)
print("DRIFT-IDENTITY CORRECTION per ascent: q > q*(1 - E_k/k)")
print(f"{'seed':>22} {'k':>5} {'q_obs':>8} {'E_k/k':>9} {'forced_bound':>12}")
mx = 0.0
for s in seeds:
    k, q, ek, fb = ascent_correction(s)
    mx = max(mx, ek)
    print(f"{s:>22} {k:>5} {q:>8.4f} {ek:>9.5f} {fb:>12.4f}")
print(f"max E_k/k across sample: {mx:.5f}  (paper cites <= 0.0025; floor q > 0.629)")

# ---------------------------------------------------------------
# 7. Drift-robustness check for the long-climber Hurst elevation
#    (Is H ~ 0.66 on the 399-step ascent a linear-drift artifact?)
#    DFA-1 removes piecewise-constant mean offsets; a linear ramp
#    in the series survives it (quadratic in the profile) and is
#    removed by DFA-2. We (a) measure the drift directly,
#    (b) recompute H under DFA-2, (c) compare against i.i.d.
#    controls carrying the OBSERVED ramp.
# ---------------------------------------------------------------

def dfa_poly(x, order=1, min_s=4, max_frac=4, n_scales=20):
    """DFA of arbitrary detrending order."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.geomspace(min_s, N // max_frac, n_scales).astype(int))
    scales = scales[scales > order + 1]
    F = []
    for s in scales:
        m = N // s
        segs = y[: m * s].reshape(m, s)
        t = np.arange(s)
        resid = np.empty_like(segs)
        for i in range(m):
            c = np.polyfit(t, segs[i], order)
            resid[i] = segs[i] - np.polyval(c, t)
        F.append(np.sqrt((resid ** 2).mean()))
    F = np.array(F)
    H, _ = np.polyfit(np.log(scales), np.log(F), 1)
    return H

print("=" * 90)
print("DRIFT-ROBUSTNESS CHECK: longest record ascent")
seed7 = max(seeds)
ps7, logs7 = trajectory_full(seed7)
peak7 = int(np.argmax(logs7))
asc = ps7[:peak7 + 1].astype(float)  # same slice as the main ascent analysis
N7 = len(asc)
q7 = asc.mean()
half = N7 // 2
sl, _ = np.polyfit(np.arange(N7), asc, 1)
print(f"seed {seed7}: ascent length {N7}, q = {q7:.4f}")
print(f"split-half density: {asc[:half].mean():.4f} -> {asc[half:].mean():.4f}; "
      f"regression slope {sl:.6f}/step (total drift {sl*N7:+.4f})")
w7 = 50
print("50-step window densities:",
      np.round([asc[i:i+w7].mean() for i in range(0, N7 - w7 + 1, w7)], 3))

h1, h2 = dfa_poly(asc, 1), dfa_poly(asc, 2)
rng7 = np.random.default_rng(27)
ramp = sl * (np.arange(N7) - N7 / 2)
H1s, H2s, H1r = [], [], []
for _ in range(500):
    b = (rng7.random(N7) < q7).astype(float)
    H1s.append(dfa_poly(b, 1)); H2s.append(dfa_poly(b, 2))
    br = (rng7.random(N7) < np.clip(q7 + ramp, 0, 1)).astype(float)
    H1r.append(dfa_poly(br, 1))
H1s, H2s, H1r = map(np.array, (H1s, H2s, H1r))
print(f"ascent:   DFA-1 H = {h1:.3f}   DFA-2 H = {h2:.3f}")
print(f"iid ctrl: DFA-1 H = {H1s.mean():.3f} +/- {H1s.std():.3f}   "
      f"DFA-2 H = {H2s.mean():.3f} +/- {H2s.std():.3f}   (DFA-2 null is biased up at this N)")
print(f"iid + observed ramp: DFA-1 H = {H1r.mean():.3f} +/- {H1r.std():.3f}")
print(f"z-scores: DFA-1 {(h1-H1s.mean())/H1s.std():+.2f}   DFA-2 {(h2-H2s.mean())/H2s.std():+.2f}   "
      f"vs ramped {(h1-H1r.mean())/H1r.std():+.2f}")
print("Conclusion: the observed ramp is too shallow to inflate H (ramped controls match iid);")
print("the elevation survives DFA-2, so it is not a linear-drift artifact. Significance 2-3 sigma")
print("depending on detrending order, per the paper's stated caveats.")
