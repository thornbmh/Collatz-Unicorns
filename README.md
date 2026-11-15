Collatz Unicorns — Analysis Code
Companion code for:
> M. H. Thornburg, \\\*Collatz Unicorns: Criticality at the Edge of Provability — Information Persistence, 1/f Noise, and Unprovable Divergence in the Collatz Map\\\* (2026). Preprint on the PhilSci-Archive.
The single script `collatz\\\_spectral.py` reproduces every empirical figure reported in Section 5.1 of the paper.
What it computes
All analysis uses the shortcut Collatz map T(x) = (3x+1)/2 for odd x, x/2 for even x, and the binary parity vector p_i = x_i mod 2 of each trajectory. The script:
Finds record climbers. Scans n < 100,000 for peak-excursion records and adds the well-known large climbers (27, 703, 626331, 837799, ..., 1980976057694848447). Every trajectory statistic is recomputed from scratch — nothing is taken from the literature.
Analyzes ascent phases (start to trajectory peak) for each seed: empirical odd-step density q, DFA-1 Hurst exponent H, and log-binned periodogram slope β (S(f) ∝ 1/f^β).
Benchmarks against controls. Each ascent is compared to 200 i.i.d. Bernoulli sequences matched in length and density parameter (fixed RNG seed 27), yielding the z-scores reported in the paper.
Analyzes post-peak (descent) phases — the "transient flattening" statistics (q ≈ 0.48–0.49, H ≈ 0.50, β ≈ 0).
Runs the crystal experiment. The engineered climber 2^400 − 1 (400 forced odd steps) is iterated through its peak; sliding-window densities show decoherence to white noise within a single 100-step window.
Computes the drift-identity correction E_k = Σ_odd log₂(1 + 1/(3x_i)) per ascent, giving the arithmetic floor q > q*(1 − E_k/k) cited in Section 5.1 (max E_k/k ≤ 0.0025 across the sample).
Key reference values the output should reproduce:
Quantity	Paper value
Critical density q* = 1/log₂3	0.6309
Ascent densities, all 13 seeds	q ∈ [0.65, 0.83]
Longest ascent (seed 1980976057694848447, 399 steps)	q = 0.7318, H ≈ 0.66 (controls 0.52 ± 0.04, z ≈ +3.1), β ≈ 0.05
Post-peak regimes	q ≈ 0.483–0.487, H ≈ 0.50, β ∈ [−0.16, −0.12]
Crystal 2^400 − 1 post-peak	white noise within one 100-step window
Requirements
Python ≥ 3.9
NumPy (any recent version; developed against 2.5)
No other dependencies. Arbitrary-precision integers are native Python.
Running
```
python3 collatz\\\_spectral.py
```
Runtime is on the order of one to two minutes on a laptop (the record scan to 10^5 dominates). All randomness is seeded (`numpy.random.default\\\_rng(27)`), so the control-ensemble statistics are exactly reproducible; tiny differences in the last reported digit may occur across NumPy versions.
Methods in brief
DFA-1: cumulative profile of the centered parity sequence; linear detrending in windows of geometrically spaced sizes s; H is the log–log slope of the RMS fluctuation F(s).
Spectral slope: periodogram of the centered sequence, averaged in log-spaced frequency bins; β is the negative log–log slope.
Ascent = steps from the seed to the trajectory's maximum; descent = steps from the maximum to 1.
Caveats carried in the paper apply here too: the H ≈ 0.66 result rests on a single trajectory, reflects selection across thirteen comparisons, and does not fully control for record-ascent conditioning.
Citation
If you use this code, please cite the paper above. The code itself may be cited as:
> Thornburg, M. H. (2026). \\\*collatz\\\_spectral.py\\\* — analysis code for "Collatz Unicorns: Criticality at the Edge of Provability."
Acknowledgment
Developed in collaboration with Claude (Anthropic); all results were verified by the author.
