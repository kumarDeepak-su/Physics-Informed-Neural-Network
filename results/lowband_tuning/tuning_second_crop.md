# Tuning on the second crop (U-Net, seed 42, epsilon 1e-3)


Tuning of the two new settings of the practical variants on the SECOND crop only
(4500:4900 x 1000:1300, epsilon = 1e-3 data of run crop_second_e3):
  network input prior   : trace-wise RW-L1 (paper) or 2-D coupled ADMM/TV result
  low-band anchor       : weight W in {0, 0.3, 1, 3}, low-pass sigma in {12, 24} samples
U-Net, seed 42, 80 epochs, synthetic pretraining; label-free and two simulated wells
(crop traces 60 and 340); clean and noisy data.
Selection rule, fixed before running: the configuration with the highest mean SNR over the
four cases (clean/noisy x label-free/wells); the wells-only mean is reported as well.
The fault crop and the full model are not touched here.

Priors: RW-L1 clean 34.63 / noisy 34.41; 2-D TV clean 34.59 / noisy 34.75.

| prior | W_lowband | sigma | clean none | clean wells | noisy none | noisy wells | mean | wells mean |
|---|---|---|---|---|---|---|---|---|
| rw | 0 | 12 | 34.60 | 34.91 | 34.76 | 34.37 | 34.66 | 34.64 |
| tv | 0 | 12 | 34.59 | 34.83 | 34.72 | 35.02 | 34.79 | 34.92 |
| rw | 0.3 | 12 | 34.57 | 34.86 | 34.70 | 34.34 | 34.62 | 34.60 |
| rw | 0.3 | 24 | 34.58 | 34.87 | 34.70 | 34.43 | 34.65 | 34.65 |
| rw | 1 | 12 | 34.58 | 34.83 | 34.60 | 34.40 | 34.60 | 34.62 |
| rw | 1 | 24 | 34.59 | 34.86 | 34.63 | 34.49 | 34.64 | 34.67 |
| rw | 3 | 12 | 34.57 | 34.79 | 34.50 | 34.46 | 34.58 | 34.63 |
| rw | 3 | 24 | 34.60 | 34.82 | 34.49 | 34.47 | 34.60 | 34.65 |
| tv | 0.3 | 12 | 34.56 | 34.84 | 34.70 | 34.98 | 34.77 | 34.91 |
| tv | 0.3 | 24 | 34.56 | 34.81 | 34.71 | 35.01 | 34.77 | 34.91 |
| tv | 1 | 12 | 34.56 | 34.82 | 34.72 | 35.00 | 34.77 | 34.91 |
| tv | 1 | 24 | 34.56 | 34.81 | 34.72 | 35.02 | 34.78 | 34.92 |
| tv | 3 | 12 | 34.54 | 34.81 | 34.71 | 34.99 | 34.76 | 34.90 |
| tv | 3 | 24 | 34.58 | 34.79 | 34.74 | 34.98 | 34.77 | 34.89 |

Selected: prior tv, W_lowband 0, sigma 12 (mean 34.79 dB).
