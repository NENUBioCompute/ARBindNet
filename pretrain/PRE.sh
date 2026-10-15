#!/bin/bash
#SBATCH --job-name='PRE'
#SBATCH --chdir=/datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain
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
         python main_PRE.py \
    --input_dir /datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/input \
    --output_dir /datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/output \
    --batch_size 128 \
    --lr 1e-4 \
    --epochs 500 \
    --patience 10 \
        ;;
esac

echo job end time is `date`
