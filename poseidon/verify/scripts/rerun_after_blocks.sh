#!/bin/zsh
# waits for the block runner to finish, then reruns v05 -> v06 -> v07 -> v08 -> v01 -> v02 with all ten blocks
cd "/Users/deepakkumar/Desktop/ONGOING_RESEARCH/MARMOUSI/Reweighted_L1_Sparse/2D_ADMM_PINN_MANUSCRIPT/DATA_POST_STACK/POSEIDON_FIELD_INVERSION/verify/scripts"
while ! grep -q "ALL DONE" "/Users/deepakkumar/Desktop/ONGOING_RESEARCH/MARMOUSI/Reweighted_L1_Sparse/2D_ADMM_PINN_MANUSCRIPT/Poseidon3D/seismic/extract_remaining_sync.log"; do sleep 20; done
n=$(ls /Users/deepakkumar/Desktop/ONGOING_RESEARCH/MARMOUSI/Reweighted_L1_Sparse/2D_ADMM_PINN_MANUSCRIPT/Poseidon3D/seismic/near_stack_*.npy | wc -l | tr -d ' ')
echo "blocks on disk: $n  $(date +%H:%M:%S)"
for s in v05_seismic_band.py "v06_ties.py --n-null 500" v07_rows_and_power.py v08_summary.py v02_grid_and_bins.py v01_inventory.py; do
  echo "=== $s $(date +%H:%M:%S)"
  ~/.venvs/deepfwi/bin/python ${=s} 2>&1 | grep -v -i "warn" | tail -16 | cut -c1-220
done
echo "CHAIN DONE $(date +%H:%M:%S)"
