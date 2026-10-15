#!/bin/bash
#SBATCH --job-name='ligand_multiclass'
#SBATCH --chdir=/datapool/home/info_wang/tongyanting/GCL/model/0522/ligand_multiclass_task
#SBATCH --partition=info-wang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=100-00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0   # 三个任务，数组ID 0,1,2

echo job start time is `date`
echo `hostname`

source ~/.bashrc
conda activate tyt_ae

# 根据 SLURM_ARRAY_TASK_ID 选择不同任务
case $SLURM_ARRAY_TASK_ID in
    0)
        python -u train_ligand_multiclass.py \
  --split_dir /datapool/home/info_wang/tongyanting/GCL/model/0522/ligand_multiclass_task/splits_selected_classes \
  --output_dir /datapool/home/info_wang/tongyanting/GCL/model/0522/ligand_multiclass_task/output/without_con \
  --stats_path /datapool/home/info_wang/tongyanting/GCL/model/0510/pretrain/output/norm_stats.pt \
  --pretrained_model_path /datapool/home/info_wang/tongyanting/GCL/model/0510/pretrain/binary_no_contrast/output/best_model.pt \
  --gvp_hidden_v 32 \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --patience 10 \
  --seed 42 \
  --num_workers 10 \
  --classifier_dropout 0.3 \
  --weighted_sampler \
  --device cuda
        ;;
esac

echo job end time is `date`
