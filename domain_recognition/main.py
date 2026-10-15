#!/usr/bin/env python
# main.py
import argparse
import os
import logging
import torch
import numpy as np
import random
from config import Config
from finetune import train_finetune  # 假设 finetune.py 中已定义 train_finetune(config, logger)


def main():
    parser = argparse.ArgumentParser(description="Protein binary classification fine-tuning")

    # ========== 路径参数：只需两个 ==========
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Root directory of input data (contains train/val/test lists, labels, norm_stats.pt)")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save outputs (models, logs, predictions)")
    # parser.add_argument('--pretrained_model_path', type=str, default=None,
    #                     help="Path to pretrained model; if not given, defaults to <input_dir>/best_model.pt")
    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        default=None,
        help='Path to pretrained model. If omitted, use Config.pretrained_model_path.'
    )

    # ========== 训练超参数 ==========
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=112)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--freeze_encoders', action='store_true', default=False)
    
    # ========== 模型结构参数（一般保持默认即可） ==========
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--egnn_layers', type=int, default=3)
    parser.add_argument('--egnn_heads', type=int, default=4)
    parser.add_argument('--use_attention', type=bool, default=True)
    parser.add_argument('--gvp_hidden_s', type=int, default=128)
    parser.add_argument('--gvp_hidden_v', type=int, default=32)
    parser.add_argument('--gvp_layers', type=int, default=3)
    parser.add_argument('--gvp_dropout', type=float, default=0.1)
    parser.add_argument('--classifier_dropout', type=float, default=0.3)
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='Skip training, only run testing with best_finetuned_model.pt')
    args = parser.parse_args()

    # ========== 根据 input_dir 自动构造所有数据路径 ==========
    input_dir = args.input_dir
    config = Config()  # 原有默认值会被覆盖

    # 列表文件
    config.train_list_path = os.path.join(input_dir, "train_list.txt")
    config.val_list_path = os.path.join(input_dir, "val_list.txt")
    config.test_list_path = os.path.join(input_dir, "test_list.txt")
    # 标签文件
    config.train_label_file = os.path.join(input_dir, "train_labels.txt")
    config.val_label_file = os.path.join(input_dir, "val_labels.txt")
    config.test_label_file = os.path.join(input_dir, "test_labels.txt")
    # # 归一化统计量（预训练时生成的）
    # config.norm_stats_path = os.path.join(input_dir, "norm_stats.pt")
    # # # 预训练权重
    # if args.pretrained_model_path is not None:
    #     config.pretrained_model_path = args.pretrained_model_path
    # else:
    #     config.pretrained_model_path = os.path.join(input_dir, "best_model.pt")
    # 设置预训练权重路径
    if args.pretrained_model_path is not None:
        config.pretrained_model_path = os.path.abspath(
            args.pretrained_model_path
        )

    # 训练模式下检查文件是否存在
    if not args.test_only:
        if not os.path.isfile(config.pretrained_model_path):
            raise FileNotFoundError(
                f"Pretrained model not found: "
                f"{config.pretrained_model_path}"
            )

    # 输出目录
    config.output_dir = args.output_dir
    os.makedirs(config.output_dir, exist_ok=True)

    # 将命令行超参覆盖到 config
    for key in ['batch_size', 'finetune_epochs', 'lr', 'weight_decay', 'patience',
                'seed', 'num_workers', 'freeze_encoders', 'hidden_dim',
                'egnn_layers', 'egnn_heads', 'use_attention', 'gvp_hidden_s',
                'gvp_hidden_v', 'gvp_layers', 'gvp_dropout', 'classifier_dropout']:
        if hasattr(config, key):
            setattr(config, key, getattr(args, key))
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== 日志设置 ==========
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(os.path.join(config.output_dir, 'finetune.log'), mode='a')
        ]
    )
    logger = logging.getLogger(__name__)

    # 固定随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 启动训练
    if args.test_only:
        from finetune import test_only  # 需要新增这个函数
        test_only(config, logger)
    else:
        train_finetune(config, logger)


if __name__ == "__main__":
    main()