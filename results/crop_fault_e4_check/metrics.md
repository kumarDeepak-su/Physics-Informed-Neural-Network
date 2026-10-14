# Complete Marmousi-2 model: metrics

Model 400 x 300 samples at 1.250 ms, noise SNR ratio 8.0, traces [8100, 8500], samples [1500, 1800].

## Classical methods, SNR (dB) per region

| Method | whole | crop_fault | shallow | middle | deep | left | centre | right |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| initial_model | 21.94 | 21.94 | 28.27 | 22.29 | 20.40 | 21.72 | 22.58 | 21.43 |
| admm_clean | 33.51 | 33.51 | 29.27 | 37.35 | 35.04 | 33.71 | 35.81 | 31.38 |
| admm_noisy | 29.01 | 29.01 | 27.86 | 30.61 | 28.55 | 29.13 | 29.45 | 28.33 |
| tv2d_clean | 31.53 | 31.53 | 29.44 | 37.49 | 30.30 | 32.25 | 33.46 | 29.26 |
| tv2d_noisy | 31.48 | 31.48 | 31.87 | 33.25 | 30.27 | 31.84 | 32.76 | 29.85 |

## Networks, training set `all` (48 patches, 0.0 % of samples held out)

| Network | Data | Supervision | Seeds | SNR whole | SNR held out | prior whole | prior held out | crop_fault | shallow | middle | deep | left | centre | right | ms/patch-step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rescnn | clean | dense | 1 | 38.94 |  | 33.51 |  | 38.94 | 38.79 | 40.11 | 38.23 | 38.26 | 41.00 | 37.71 | 21.09 |
| rescnn | noisy | dense | 1 | 33.32 |  | 29.01 |  | 33.32 | 34.34 | 33.75 | 32.61 | 32.85 | 34.05 | 32.99 | 21.09 |
| unet | clean | dense | 1 | 40.47 |  | 33.51 |  | 40.47 | 43.45 | 40.93 | 39.27 | 39.09 | 42.43 | 40.20 | 21.11 |
| unet | noisy | dense | 1 | 33.96 |  | 29.01 |  | 33.96 | 35.55 | 34.35 | 33.10 | 33.30 | 34.68 | 33.86 | 21.10 |
