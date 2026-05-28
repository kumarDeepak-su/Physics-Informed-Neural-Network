# Fault-Focused Marmousi Transfer Benchmark

This folder contains the reproducibility files for the cropped Marmousi transfer-learning benchmark used in the revised manuscript.

Files:

- `02_2D_Laptop_ADMM_PINN_ResCNN_Attention.py`: the executed Python script.
- `2D_results.npz`: saved numerical arrays for the benchmark.
- `2D_results_metadata.json`: run metadata, crop coordinates, software versions, parameters, metrics, timings, and output-file records.

Benchmark crop:

- Traces: `8100:8500`
- Samples: `1500:1800`
- Shape: `400 x 300`

Command used:

```bash
python 02_2D_Laptop_ADMM_PINN_ResCNN_Attention.py \
  --benchmark marmousi-crop \
  --run-tag marmousi_fault_crop_400x300 \
  --epochs 80 \
  --device cpu
```

The script expects local `vp.segy` and `density.segy` Marmousi files. Their paths and file sizes from the original run are recorded in `2D_results_metadata.json`.
