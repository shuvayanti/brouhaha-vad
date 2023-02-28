#!/bin/bash
#SBATCH --gres=gpu:gtx1080:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00

# load conda environment
source /shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate brouhaha

python main.py apply \
      --data_dir /scratch2/mlavechin/VoiceTypeClassifierPaper/DATA/BabyTrain_full_resplitted2/test/ \
      --out_dir /home/sdas/brouhaha-vad/predictions \
      --model_path models/best/checkpoints/best.ckpt \
      --ext "wav"
