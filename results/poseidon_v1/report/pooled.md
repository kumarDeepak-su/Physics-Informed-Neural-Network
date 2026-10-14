# Poseidon field example: pooled scores (protocol v1)

Row-weighted means over the evaluation lines (two per withheld well); networks: seed means.

| Method | lines | r band | r envelope | r low | rel. RMSE | r full | p(band) median | p(env) median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Starting model | 10 | -0.031 | 0.150 | -- | 0.893 | 0.677 | 1.000 | 1.000 |
| Trace-wise RW-L1 ADMM | 10 | 0.443 | 0.556 | 0.306 | 0.993 | 0.641 | 0.000 | 0.027 |
| 2-D coupled ADMM | 10 | 0.445 | 0.562 | 0.255 | 0.959 | 0.651 | 0.000 | 0.023 |
| 2D U-Net, 2-D input | 10 | 0.425 | 0.546 | 0.391 | 0.898 | 0.689 | 0.000 | 0.013 |
| ResCNN, 2-D input | 10 | 0.419 | 0.555 | 0.430 | 0.884 | 0.699 | 0.000 | 0.038 |
| Ensemble U-Net + ResCNN, 2-D input (derived) | 10 | 0.432 | 0.570 | 0.432 | 0.875 | 0.704 | 0.000 | 0.029 |
| 2D U-Net, trace-wise input | 10 | 0.418 | 0.546 | 0.277 | 0.984 | 0.643 | 0.000 | 0.013 |
| ResCNN, trace-wise input | 10 | 0.425 | 0.547 | 0.335 | 0.968 | 0.656 | 0.000 | 0.038 |

## Per line

| Line | Method | r band | r envelope | r low | rel. RMSE | r full | p(band) | p(env) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Kronos_1_inline | Starting model | -0.095 | -0.067 | -- | 0.887 | 0.703 | 1.000 | 1.000 |
| Kronos_1_crossline | Starting model | -0.095 | -0.067 | -- | 0.887 | 0.703 | 1.000 | 1.000 |
| Pharos_1_inline | Starting model | -0.522 | 0.515 | -- | 1.023 | 0.846 | 1.000 | 1.000 |
| Pharos_1_crossline | Starting model | -0.522 | 0.515 | -- | 1.023 | 0.846 | 1.000 | 1.000 |
| Poseidon_2_inline | Starting model | -0.099 | -0.111 | -- | 0.674 | 0.779 | 1.000 | 1.000 |
| Poseidon_2_crossline | Starting model | -0.099 | -0.111 | -- | 0.674 | 0.779 | 1.000 | 1.000 |
| Proteus_1_inline | Starting model | 0.162 | 0.293 | -- | 0.935 | 0.745 | 1.000 | 1.000 |
| Proteus_1_crossline | Starting model | 0.162 | 0.293 | -- | 0.935 | 0.745 | 1.000 | 1.000 |
| Torosa_1_inline | Starting model | 0.471 | 0.669 | -- | 1.293 | -0.115 | 1.000 | 1.000 |
| Torosa_1_crossline | Starting model | 0.471 | 0.669 | -- | 1.293 | -0.115 | 1.000 | 1.000 |
| Kronos_1_inline | Trace-wise RW-L1 ADMM | 0.303 | 0.509 | 0.245 | 0.922 | 0.680 | 0.000 | 0.000 |
| Kronos_1_crossline | Trace-wise RW-L1 ADMM | 0.303 | 0.509 | 0.245 | 0.922 | 0.680 | 0.000 | 0.084 |
| Pharos_1_inline | Trace-wise RW-L1 ADMM | 0.575 | 0.527 | 0.606 | 0.893 | 0.864 | 0.000 | 0.272 |
| Pharos_1_crossline | Trace-wise RW-L1 ADMM | 0.575 | 0.527 | 0.606 | 0.893 | 0.864 | 0.009 | 0.373 |
| Poseidon_2_inline | Trace-wise RW-L1 ADMM | 0.440 | 0.546 | 0.693 | 0.817 | 0.758 | 0.000 | 0.000 |
| Poseidon_2_crossline | Trace-wise RW-L1 ADMM | 0.440 | 0.546 | 0.693 | 0.817 | 0.758 | 0.000 | 0.000 |
| Proteus_1_inline | Trace-wise RW-L1 ADMM | 0.742 | 0.596 | 0.114 | 1.044 | 0.709 | 0.000 | 0.000 |
| Proteus_1_crossline | Trace-wise RW-L1 ADMM | 0.742 | 0.596 | 0.114 | 1.044 | 0.709 | 0.000 | 0.000 |
| Torosa_1_inline | Trace-wise RW-L1 ADMM | -0.154 | 0.639 | -0.582 | 1.722 | -0.308 | 0.799 | 0.054 |
| Torosa_1_crossline | Trace-wise RW-L1 ADMM | -0.154 | 0.639 | -0.582 | 1.722 | -0.308 | 0.689 | 0.364 |
| Kronos_1_inline | 2-D coupled ADMM | 0.307 | 0.513 | 0.099 | 0.957 | 0.649 | 0.000 | 0.000 |
| Kronos_1_crossline | 2-D coupled ADMM | 0.305 | 0.519 | 0.006 | 0.962 | 0.635 | 0.000 | 0.062 |
| Pharos_1_inline | 2-D coupled ADMM | 0.584 | 0.562 | 0.237 | 1.008 | 0.854 | 0.000 | 0.238 |
| Pharos_1_crossline | 2-D coupled ADMM | 0.577 | 0.555 | 0.329 | 0.993 | 0.865 | 0.009 | 0.302 |
| Poseidon_2_inline | 2-D coupled ADMM | 0.440 | 0.547 | 0.427 | 0.813 | 0.705 | 0.000 | 0.000 |
| Poseidon_2_crossline | 2-D coupled ADMM | 0.440 | 0.550 | 0.488 | 0.781 | 0.736 | 0.000 | 0.000 |
| Proteus_1_inline | 2-D coupled ADMM | 0.741 | 0.593 | 0.409 | 0.931 | 0.779 | 0.000 | 0.000 |
| Proteus_1_crossline | 2-D coupled ADMM | 0.741 | 0.596 | 0.353 | 0.930 | 0.769 | 0.000 | 0.000 |
| Torosa_1_inline | 2-D coupled ADMM | -0.154 | 0.647 | -0.157 | 1.486 | -0.142 | 0.799 | 0.045 |
| Torosa_1_crossline | 2-D coupled ADMM | -0.148 | 0.639 | -0.225 | 1.481 | -0.155 | 0.689 | 0.378 |
| Kronos_1_inline | ResCNN, trace-wise input | 0.295 | 0.562 | 0.192 | 0.931 | 0.675 | 0.000 | 0.000 |
| Kronos_1_crossline | ResCNN, trace-wise input | 0.262 | 0.480 | 0.406 | 0.855 | 0.743 | 0.000 | 0.116 |
| Pharos_1_inline | ResCNN, trace-wise input | 0.578 | 0.466 | 0.595 | 1.009 | 0.863 | 0.000 | 0.266 |
| Pharos_1_crossline | ResCNN, trace-wise input | 0.587 | 0.516 | 0.568 | 1.166 | 0.865 | 0.004 | 0.240 |
| Poseidon_2_inline | ResCNN, trace-wise input | 0.399 | 0.564 | 0.670 | 0.756 | 0.745 | 0.000 | 0.000 |
| Poseidon_2_crossline | ResCNN, trace-wise input | 0.395 | 0.488 | 0.679 | 0.745 | 0.756 | 0.000 | 0.000 |
| Proteus_1_inline | ResCNN, trace-wise input | 0.739 | 0.587 | 0.143 | 1.020 | 0.731 | 0.000 | 0.000 |
| Proteus_1_crossline | ResCNN, trace-wise input | 0.734 | 0.583 | 0.217 | 0.978 | 0.746 | 0.000 | 0.009 |
| Torosa_1_inline | ResCNN, trace-wise input | -0.141 | 0.657 | -0.458 | 1.610 | -0.254 | 0.756 | 0.068 |
| Torosa_1_crossline | ResCNN, trace-wise input | -0.156 | 0.650 | -0.546 | 1.593 | -0.306 | 0.693 | 0.360 |
| Kronos_1_inline | ResCNN, 2-D input | 0.288 | 0.559 | 0.427 | 0.949 | 0.731 | 0.000 | 0.000 |
| Kronos_1_crossline | ResCNN, 2-D input | 0.276 | 0.533 | 0.162 | 0.942 | 0.679 | 0.000 | 0.044 |
| Pharos_1_inline | ResCNN, 2-D input | 0.570 | 0.484 | 0.480 | 0.882 | 0.848 | 0.000 | 0.255 |
| Pharos_1_crossline | ResCNN, 2-D input | 0.591 | 0.592 | 0.499 | 0.935 | 0.885 | 0.009 | 0.036 |
| Poseidon_2_inline | ResCNN, 2-D input | 0.374 | 0.531 | 0.547 | 0.718 | 0.754 | 0.000 | 0.000 |
| Poseidon_2_crossline | ResCNN, 2-D input | 0.382 | 0.522 | 0.543 | 0.703 | 0.770 | 0.000 | 0.000 |
| Proteus_1_inline | ResCNN, 2-D input | 0.732 | 0.564 | 0.511 | 0.871 | 0.807 | 0.000 | 0.000 |
| Proteus_1_crossline | ResCNN, 2-D input | 0.736 | 0.571 | 0.624 | 0.770 | 0.836 | 0.000 | 0.040 |
| Torosa_1_inline | ResCNN, 2-D input | -0.159 | 0.660 | 0.085 | 1.369 | -0.039 | 0.785 | 0.057 |
| Torosa_1_crossline | ResCNN, 2-D input | -0.163 | 0.647 | -0.112 | 1.415 | -0.089 | 0.702 | 0.382 |
| Kronos_1_inline | 2D U-Net, trace-wise input | 0.299 | 0.537 | 0.041 | 0.961 | 0.619 | 0.003 | 0.000 |
| Kronos_1_crossline | 2D U-Net, trace-wise input | 0.271 | 0.552 | 0.278 | 0.894 | 0.704 | 0.000 | 0.009 |
| Pharos_1_inline | 2D U-Net, trace-wise input | 0.576 | 0.458 | 0.551 | 1.034 | 0.841 | 0.000 | 0.295 |
| Pharos_1_crossline | 2D U-Net, trace-wise input | 0.562 | 0.438 | 0.649 | 0.959 | 0.870 | 0.013 | 0.378 |
| Poseidon_2_inline | 2D U-Net, trace-wise input | 0.380 | 0.530 | 0.622 | 0.796 | 0.757 | 0.000 | 0.000 |
| Poseidon_2_crossline | 2D U-Net, trace-wise input | 0.400 | 0.509 | 0.638 | 0.781 | 0.760 | 0.000 | 0.000 |
| Proteus_1_inline | 2D U-Net, trace-wise input | 0.735 | 0.612 | 0.210 | 1.013 | 0.744 | 0.000 | 0.000 |
| Proteus_1_crossline | 2D U-Net, trace-wise input | 0.720 | 0.566 | 0.055 | 1.054 | 0.702 | 0.000 | 0.018 |
| Torosa_1_inline | 2D U-Net, trace-wise input | -0.200 | 0.647 | -0.502 | 1.606 | -0.290 | 0.853 | 0.074 |
| Torosa_1_crossline | 2D U-Net, trace-wise input | -0.172 | 0.642 | -0.516 | 1.562 | -0.230 | 0.684 | 0.347 |
| Kronos_1_inline | 2D U-Net, 2-D input | 0.327 | 0.579 | 0.314 | 0.962 | 0.704 | 0.000 | 0.000 |
| Kronos_1_crossline | 2D U-Net, 2-D input | 0.304 | 0.547 | 0.184 | 0.870 | 0.684 | 0.000 | 0.018 |
| Pharos_1_inline | 2D U-Net, 2-D input | 0.569 | 0.434 | 0.502 | 0.915 | 0.872 | 0.000 | 0.354 |
| Pharos_1_crossline | 2D U-Net, 2-D input | 0.584 | 0.521 | 0.519 | 0.972 | 0.884 | 0.004 | 0.196 |
| Poseidon_2_inline | 2D U-Net, 2-D input | 0.424 | 0.532 | 0.496 | 0.764 | 0.753 | 0.000 | 0.000 |
| Poseidon_2_crossline | 2D U-Net, 2-D input | 0.381 | 0.464 | 0.516 | 0.721 | 0.750 | 0.000 | 0.000 |
| Proteus_1_inline | 2D U-Net, 2-D input | 0.729 | 0.575 | 0.464 | 0.840 | 0.792 | 0.000 | 0.000 |
| Proteus_1_crossline | 2D U-Net, 2-D input | 0.731 | 0.568 | 0.461 | 0.841 | 0.794 | 0.000 | 0.009 |
| Torosa_1_inline | 2D U-Net, 2-D input | -0.262 | 0.673 | 0.100 | 1.428 | -0.054 | 0.881 | 0.054 |
| Torosa_1_crossline | 2D U-Net, 2-D input | -0.228 | 0.653 | 0.048 | 1.419 | -0.042 | 0.742 | 0.351 |
| Kronos_1_inline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.316 | 0.589 | 0.404 | 0.943 | 0.729 | 0.000 | 0.000 |
| Kronos_1_crossline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.301 | 0.574 | 0.192 | 0.888 | 0.695 | 0.000 | 0.031 |
| Pharos_1_inline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.608 | 0.516 | 0.520 | 0.874 | 0.875 | 0.000 | 0.306 |
| Pharos_1_crossline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.605 | 0.586 | 0.544 | 0.945 | 0.893 | 0.004 | 0.107 |
| Poseidon_2_inline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.407 | 0.542 | 0.536 | 0.714 | 0.763 | 0.000 | 0.000 |
| Poseidon_2_crossline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.390 | 0.512 | 0.541 | 0.691 | 0.769 | 0.000 | 0.000 |
| Proteus_1_inline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.737 | 0.576 | 0.504 | 0.847 | 0.807 | 0.000 | 0.000 |
| Proteus_1_crossline | Ensemble U-Net + ResCNN, 2-D input (derived) | 0.744 | 0.580 | 0.587 | 0.791 | 0.829 | 0.000 | 0.027 |
| Torosa_1_inline | Ensemble U-Net + ResCNN, 2-D input (derived) | -0.213 | 0.679 | 0.093 | 1.393 | -0.047 | 0.850 | 0.051 |
| Torosa_1_crossline | Ensemble U-Net + ResCNN, 2-D input (derived) | -0.198 | 0.661 | -0.033 | 1.412 | -0.066 | 0.720 | 0.364 |
