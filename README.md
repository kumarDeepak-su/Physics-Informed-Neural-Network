# ADMM-guided physics-informed deep learning for 2D acoustic impedance inversion

Code, run records and numerical results of

> D. Kumar and J. N. Tripathi, "ADMM-Guided Physics-Informed Deep Learning for 2D Acoustic
> Impedance Inversion with Reweighted l1 Sparse Regularization", IEEE Transactions on Geoscience
> and Remote Sensing (under revision, manuscript TGRS-2026-04997).

This release accompanies the revised manuscript. It contains the submitted code unchanged in its
algorithms, the additions made for the revision (banded solvers, the complete-Marmousi-2 runner,
the sparse-well supervision, the field-data preparation and runner with its frozen protocol) and
the records of every run reported in the paper. The state of the repository at the revised submission is
tagged `tgrs-revision-1`.

## Layout

| Path | Content |
|---|---|
| `2D_ADMM_PINN_ResCNN_Attention.py` | Main script: forward model, reweighted-l1 ADMM, 2-D coupled ADMM, the three networks, the physics-informed loss, training and the crop experiments (Sections II to VI). |
| `admm_banded.py` | Banded (exact, parallel) versions of the trace-wise and 2-D coupled solvers used for the complete model and the field data; reproduce the dense solvers to 1e-6 dB. |
| `run_marmousi_full.py` | Stages `prepare`, `train`, `baseline`, `sweep`, `report` for the complete Marmousi-2 model: starting models, noise rule, supervision settings (`dense`, `none`, `wells`), input prior (`--prior rw|tv`), well-conditioned comparator (`baseline --wells`), regions, records (Section VII). |
| `test_marmousi_full.py` | Tests T1 to T8: equivalence to the published crop results, forward model, masked supervision, well penalty, end-to-end slice. |
| `summarise_runs.py`, `tables_full_marmousi.py`, `figures_full_marmousi.py`, `ensemble_full_marmousi.py` | Cross-run tables, LaTeX rows, manuscript figures and ensemble rows of Section VII from the stored outputs. |
| `diagnose_objective.py`, `tune_second_crop.py` | Diagnosis of the label-free objective and the tuning of the network input on the development crop (disclosed in Section VII-A). |
| `run_2d_tv_baseline.py`, `run_noise_sweep.py`, `run_noise_sweep_matched.py`, `run_wavelet_mismatch.py`, `run_multiseed_ci.py`, `_replot_*.py`, `generate_publication_architecture_figures_v5.py` | Crop-experiment scripts of the submitted version and the figure regeneration from cached results (Sections V and VI, Fig. 2). |
| `poseidon/` | Field example (Section VIII): `prepare_poseidon.py` (lines, logs, wavelets, calibration), `vsp_alignment.py` (VSP-derived alignment), `poseidon_runner.py` (phases A and B with the read ledger and seals), `figures_poseidon.py`, `tables_poseidon.py`, `tables_poseidon_seeds.py` (per-seed field statistics, Table S12); `verify/` holds the verification scripts v01 to v08, their evidence, the data provenance notes and the NOPIMS cross-check. |
| `protocol/PROTOCOL_v1.md` | The field protocol fixed before any inversion, with its SHA-256 and two dated addenda. |
| `athena_jobs/` | Slurm job scripts used on the Athena cluster (one node: 8 A100, 128 cores). |
| `SYNTHETIC_PRETRAIN/` | The six checkpoints trained on the synthetic benchmark (Section V), with hashes; the warm start of every Marmousi-2 and field training. |
| `results/` | Reports (`metrics.md`, `metrics.json`, `prepared.json`, `ensemble.json`) of every complete-model run, the field scores per case and the pooled report, the prepared field lines and logs with their manifest, the tuning grid and the diagnosis records. |

## Environment

Python 3.11; NumPy 2.4.6; SciPy 1.17.1; PyTorch 2.10.0 (CUDA 12.6 on the cluster, CPU or MPS on a
laptop); Matplotlib 3.10.9; threadpoolctl; for the field data lasio 0.32, segyio and openpyxl.
`requirements.txt` pins the versions. The crop experiments run on a laptop CPU in minutes; the
complete-model trainings need a GPU (about one hour each on a shared A100).

## Reproducing the paper

Crop experiments (Sections IV to VI): run `2D_ADMM_PINN_ResCNN_Attention.py` with the presets
documented in its header, then the `run_*.py` scripts; the figures are regenerated from the cached
`.npz` results by the `_replot_*.py` scripts.

Complete Marmousi-2 model (Section VII), on a node with the Marmousi-2 SEG-Y files in
`data/marmousi2/`:

```
python run_marmousi_full.py prepare --out runs/full --marmousi-dir data/marmousi2 \
    --smooth-sigma-x 22 --smooth-sigma-t 33 --noise-reference fault --epsilon 1e-3
python run_marmousi_full.py sweep --out runs/full --networks unet rescnn --datas clean noisy \
    --supervisions dense none wells --n-wells 5 --seeds 42 123 7 2024 2025 --train-set all \
    --transfer-dir SYNTHETIC_PRETRAIN --device cuda --with-baseline --prior tv
python run_marmousi_full.py baseline --out runs/full --wells --n-wells 5 --well-weight 1
python run_marmousi_full.py report --out runs/full
```

`athena_jobs/marmousi_full_node.sbatch`, `starting_model_sweep.sbatch`, `full_confirm_tv.sbatch` and
`sm_sweep_tv.sbatch` (the five-well, 2-D-input variant on the other seven starting models) are the exact
jobs that produced the reported runs; `results/*/prepared.json` and `metrics.json`
carry every parameter.

Field example (Section VIII): the near-stack blocks and well files are read from the public sources
named in the paper (AWS bucket `s3://tgs-opendata-poseidon`; the NOPIMS well packages, obtained from the
public ConocoPhillips-supported mirror linked from the TerraNubis dataset page); `poseidon/verify/` documents
their verification, `poseidon/prepare_poseidon.py` writes the lines, logs and wavelets (released in
`results/poseidon_prepared/` with their hashes), and `athena_jobs/poseidon_cases.sbatch` runs the
five leave-one-well-out cases, the seals, the scores and the report.

## Data

Marmousi-2 (Martin, Wiley and Marfurt, 2006) is the public elastic model. The Poseidon 3D angle
stacks are distributed by TGS through the Registry of Open Data on AWS under CC BY 4.0; the original
data were acquired by ConocoPhillips and are held by Geoscience Australia, whose NOPIMS repository
provides the well data; the well files used here were obtained from the public ConocoPhillips-supported
mirror linked from the TerraNubis dataset page (https://terranubis.com/datainfo/NW-Shelf-Australia-Poseidon-3D,
CC BY 3.0 AU), and the AWS dataset page is https://registry.opendata.aws/tgs-opendata-poseidon/.
All 75 mirror well files used were checked against the NOPIMS package listings and match in name and byte
size (`poseidon/verify/NOPIMS_CROSSCHECK_2026-10-12.md`; the provenance notes are in `poseidon/verify/SOURCE.md`). Neither data set is redistributed here beyond the extracted field lines and
rendered logs needed to reproduce the field example.

## Licence

The code is released under the MIT licence. The data keep the licences of their providers.
