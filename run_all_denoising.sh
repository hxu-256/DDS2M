#!/bin/bash
# Launch all four VST-CARS denoising runs across 2 GPUs.
# GPU0: abr14_cond (448) -> glycerol (64)
# GPU1: abr14_ctrl (448) -> ygbead (256)
cd "/mnt/d/GaTech Dropbox/Haoyu Xu/workspace/crikit3_denoise/DDS2M"
PY=/home/hxu256/miniconda3/envs/DDS2M/bin/python
mkdir -p run_logs

gpu0_jobs() {
  DDS2M_GPU=0 $PY main_denoising.py --config msi_denoising_abr14_cond.yml --deg denoising0.11 --rank 10 > run_logs/abr14_cond.out 2>&1
  DDS2M_GPU=0 $PY main_denoising.py --config msi_denoising_glycerol.yml   --deg denoising0.07 --rank 10 > run_logs/glycerol.out   2>&1
}
gpu1_jobs() {
  DDS2M_GPU=1 $PY main_denoising.py --config msi_denoising_abr14_ctrl.yml --deg denoising0.12 --rank 10 > run_logs/abr14_ctrl.out 2>&1
  DDS2M_GPU=1 $PY main_denoising.py --config msi_denoising_ygbead.yml     --deg denoising0.13 --rank 10 > run_logs/ygbead.out     2>&1
}

gpu0_jobs &
P0=$!
gpu1_jobs &
P1=$!
wait $P0 $P1
echo "ALL DONE"
