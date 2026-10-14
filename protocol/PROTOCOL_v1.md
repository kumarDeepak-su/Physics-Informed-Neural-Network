# Poseidon 3D field example: inversion protocol, version 1

Written 10 October 2026, before any inversion of the prepared lines and before any score.
Everything in sections 1 to 8 is fixed by this document; the only quantities computed after it
are the ones it describes. A change to any rule after a score exists is a new protocol version,
and the earlier scores are reported with it.

Scope: Section VIII of the revised manuscript TGRS-2026-04997 (reviewer comment 6, field data).
The field example demonstrates the workflow of Section VII (classical inversions, networks with a
few well logs, 2-D coupled result as the network input) on real data with real wells and a
withheld-well test. It is an application with declared limits, not a powered method ranking:
the ranking of methods is established on the complete Marmousi-2 model.

## 1. Data (verification items v01 to v08 of `DATA_POST_STACK/POSEIDON_FIELD_INVERSION/verify/`)

- Seismic: Poseidon 3D near-angle stack (6 to 18 degrees), Kirchhoff PSTM, 4 ms, SEG reverse
  polarity (trough = impedance increase), zero-phased by the contractor, post-stack Q and
  time-variant scaling, no AGC. Ten 384 x 512 x 1501 blocks around the ten wells of the survey,
  read from the open AWS store (CC BY 4.0), hashed in `near_stack_blocks_record.json`.
- Wells: the five wells with a digital VSP time-depth, measured density and sonic over a usable
  window and a verification tie that beats both null tests: Kronos 1, Pharos 1, Poseidon 2,
  Proteus 1, Torosa 1 (v06, v08). Poseidon 1 is excluded (18 rows, tie not significant);
  Boreas 1, Crown 1, Poseidon North 1 and Zephyros 1 have no time-depth on disk. The selection
  is by these data criteria only and was made during verification, before this protocol.
- Prepared files (`8_field_poseidon/prepared/`, `manifest.json` with SHA-256 of every input and
  output, `PREPARE_RECORD.md`): for each well the inline and the crossline through the well
  (512 and 384 traces), window 1.5 to 4.0 s (625 samples at 4 ms) extended by 64 samples on each
  side for the wavelet; log-impedance in two-way time on the window axis (bin averages of
  ln(density x velocity) from the delivered time-domain logs, Torosa 1 from the checkshot-
  calibrated LAS restricted to its measured-density interval 3577 to 4666 m MD); the low-passed
  log; the data and synthetic RMS over the tie rows at the well.
- Impedance unit: kg m^-2 s^-1 (SI). Only ratios and logarithms enter the inversion; the unit
  affects nothing but the display.

## 2. Forward model, wavelet, polarity, gain

- Forward convention of the submitted code: r[q] = (Z[q+1] - Z[q]) / (Z[q+1] + Z[q]),
  synthetic = full convolution with the wavelet trimmed by (n_w - 1)/2 at both ends, i.e. the
  wavelet centre at zero lag. The field line S_ext carries 64 samples before and 63 after the
  window so that the trace-wise and 2-D solvers see the same data length as in the synthetic
  experiments (nt + n_w - 2 = 752 samples).
- Survey wavelet: the mean of the five tied blocks' zero-phase statistical wavelets of v05
  (seismic only, no well information), unit peak, 129 taps at 4 ms, centre 64; half-amplitude
  band 12.1 to 58.3 Hz (`prepared/survey_wavelet.json`). The declared polarity is applied by
  negating the wavelet. The per-block wavelets are a declared sensitivity for the classical
  methods only.
- Weight transfer: the paper's ADMM weights (mu = 5e-6, alpha = 5e-5, lambda = 1e-4) are
  multiplied by s_n = (sigma_op(survey wavelet) / sigma_op(paper Ricker))^2 = 1.0805, the rule of
  the earlier field protocols (sigma_op = max over frequency of pi f |W(f)|); the soft threshold
  mu/lambda, the iteration counts and the tolerances are unchanged; the reweighting floor is
  epsilon = 1e-3, the value of Section VII (declared). The 2-D coupled solver keeps
  mu_x = 100 mu, rho = 40 alpha, 40 outer and 10 inner iterations. The physics weight of the
  networks is W_PHYSICS / s_n.
- Gain: the data are divided by g_case, the geometric mean over the four calibration wells of
  the case of (data RMS / synthetic RMS) over each well's tie rows, with the synthetic from the
  rendered log and the survey wavelet (`prepared/wells/*.npz`, `gain_well`). The evaluation well
  contributes nothing to its own case's gain.

## 3. Alignment and its budget

- Primary alignment of every well: its delivered VSP time-depth, no shift, no stretch, no phase
  fit. The verification lags of the tied wells with the survey wavelet (sub-sample, v06 rule)
  are +7.3, +2.0, -3.8, -0.7 and +10.0 ms for Kronos 1, Pharos 1, Poseidon 2, Proteus 1 and
  Torosa 1; under the primary alignment these are the timing errors of the targets.
- Gates (v05, v08): relative-band metric 2.3 ms, envelope and low-pass metrics 18.9 ms. The
  relative-band correlation is therefore timing-limited at four of the five wells and is reported
  as such; the envelope and low-pass correlations and the log-impedance RMSE are inside the
  budget at every well and carry the comparison.
- Declared sensitivity: each evaluation well's own fitted lag (in-sample, an upper bound of what
  a tie would give), reported next to the primary numbers and never used for a decision.

## 4. Cases

Leave-one-well-out, five cases, in the fixed order Proteus 1, Pharos 1, Poseidon 2, Kronos 1,
Torosa 1. In each case the evaluation well E is withheld: its log is not read in phase A (ledger
with a must-not-read entry), its lines are inverted and used label-free, its log enters only
phase B. The other four wells are calibration wells: they supply the gain (section 2), the
starting model (section 5) and the sparse labels on their own two lines (section 6). Each case
is scored on the two lines through E, at the well trace of each line (integer trace, no
interpolation).

## 5. Starting model

Flat (laterally constant) low-frequency model: the mean over the calibration wells of the
low-passed log-impedance (Gaussian, 16 samples = 2.9 Hz half amplitude) on each sample where at
least one calibration log is defined, held at the end values outside, identical for every trace
of every line of the case. No horizon interpretation is used. This is the weakest structural
prior of Section VII-B (the well-derived model), chosen because the release carries no
interpreted horizons and because it gives every method the same, fully declared start.

## 6. Methods

On d / g_case with the survey wavelet, for every line of the case:
- M1 trace-wise reweighted-L1 ADMM (banded solver of Section VII);
- M2 2-D coupled ADMM (lateral TV + reweighted L1), the same solver as Section VII;
- M3 2D U-Net and M4 ResCNN with the 2-D coupled result as input (the practical configuration of
  Section VII), trained once per case on the patches of all ten lines: on the eight calibration
  lines the supervised and vertical-gradient terms act on the calibration well's column over its
  log rows (the sparse setting of Section VII); on the two evaluation lines no label term acts.
  Patch geometry 64 x 128, stride 32 x 64, 80 epochs, the loss and schedule of Section III,
  warm start from the synthetic noisy checkpoints, five seeds, reported as the seed mean and
  spread. Declared sensitivity: the same networks with the trace-wise result as input.
- B1 starting model alone (the floor).
No method, weight or seed is changed after a score exists. The 2-D coupled inversion is also the
information-matched classical reference: on the evaluation line no method sees the evaluation
well, and the calibration wells reach the classical methods only through the gain and the
starting model.

## 7. Scores (phase B, once per case)

Target: the rendered ln(impedance) of the evaluation well on its tie rows (log rows minus 64 ms
at each end). Prediction: ln Z of each method at the well trace. Metrics, identical rows for
every method: (i) correlation of the band-passed series (12 to 58 Hz, zero-phase), reported with
the timing caveat of section 3; (ii) correlation of the envelopes of the band-passed series;
(iii) correlation of the low-passed series (below 12 Hz) after removing the starting model;
(iv) relative RMSE of ln Z, i.e. RMS error over the standard deviation of the target; (v) full-
band correlation. Spatial null for (i), (ii) and (v): the same metric between the target and the
prediction at every trace at least 80 traces (1 km) from the well on the same line; p = fraction
of such traces scoring at least as well as the well trace. Pooled result: the row-weighted mean
of each metric over the ten evaluation lines, with the five-well spread; effective sample size
429 by the bandwidth rule (v07). Networks: metrics per seed, then mean and standard deviation.

## 8. What the result can and cannot support

Differences between methods smaller than about 0.1 in correlation are not resolvable with 429
effective samples (Zou minimum detectable difference 0.06 to 0.14 at the frozen power, v08).
The section reports the numbers as they come, including a null result, and makes no ranking
claim beyond what the pooled spread and the nulls support. The angle range of the stack (6 to
18 degrees) is a declared departure from normal incidence; its AVO bias is not corrected.

## 9. Reproducibility

Code: `3_code/poseidon/prepare_poseidon.py` (data), `3_code/poseidon/poseidon_runner.py`
(phases A and B, records with SHA-256 of inputs and outputs, must-not-read enforcement in
phase A). Cluster: Athena, one GPU per case. Every record carries the protocol hash
(`PROTOCOL_v1.sha256`) and the software versions.

## Addendum 1 (10 October 2026, after the first scores; no change to sections 1 to 8)

- A derived row is added to the score tables: the mean of the U-Net and ResCNN seed-mean sections with the
  2-D input (M5). It is computed from the sealed phase-A outputs, involves no new inversion, weight, seed or
  selection, and is reported as a derived variant next to M3 and M4.
- The sensitivity of section 3 (each well's own verification lag) is reported in the tables as "band, own lag".

## Addendum 2 (10 October 2026, after the first scores; declared alignment sensitivity)

- VSP-derived alignment: for every well with a VSP corridor stack on disk (Kronos 1, Pharos 1, Poseidon 2;
  Torosa 1 has a deconvolved upgoing wavefield instead), the shift between the surface near-stack trace at
  the well and the VSP trace is the lag of maximum absolute correlation over the log window extended by
  100 ms, lags -24 to +24 ms on the 1 ms grid of the VSP, refined by parabolic interpolation
  (`3_code/poseidon/vsp_alignment.py`, `prepared/vsp_alignment.json`). Proteus 1 has no VSP trace on disk and
  keeps a zero shift. The shift uses two seismic measurements and no impedance log; it is applied to the
  target of the withheld well as a second alignment ("VSP lag") next to the primary one (no shift) and the
  in-sample one (own lag). The rule was fixed before the shifts were applied to any score; the exploratory
  correlation curves that motivated it were inspected once, on the same four wells, before this addendum.
- Nothing in phase A changes; phase B is recomputed from the sealed outputs with the three alignments.
