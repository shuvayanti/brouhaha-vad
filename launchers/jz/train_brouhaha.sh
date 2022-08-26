#!/bin/bash
#SBATCH --account=xdz@v100
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread


python main.py runs/brouhaha/ train \
-p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
--classes brouhaha \
--model_type pyannet \
--epoch 10 \
--data_dir "/gpfsscratch/rech/xdz/uzm31mf/data_new/mini_data"