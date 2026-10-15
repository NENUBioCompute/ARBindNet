#!/bin/bash
#SBATCH --job-name='ALL_PR_RECOG'
#SBATCH --chdir=/datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/domain_recognition
#SBATCH --partition=info-wang
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=100-00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-4   # 四个任务，对应种子 0,1,2,3,4,5

echo job start time is `date`
echo `hostname`

source ~/.bashrc
conda activate tyt_ae

# 定义需要使用的四个种子，按数组索引对应
seeds=(0,1,2,3,4,5)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

# 运行训练脚本，输出目录中会包含具体种子值以便区分
python main.py \
    --input_dir /datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/domain_recognition/input \
    --output_dir /datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/domain_recognition/output01 \
    --pretrained_model_path /datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/output_mask_0.10/best_model.pt \
    --seed ${seed} \
    --batch_size 128 \
    --lr 1e-4 \
    --finetune_epochs 500
