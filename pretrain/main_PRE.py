# main.py
import os
import argparse
from config import Config
from train_PRE import train, setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="ARBindNet protein pretraining")
    # 输入/输出路径
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing train_list.txt, val_list.txt, test_list.txt')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for logs, models, and normalization stats')
    # 可选超参数覆盖
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (default: 10)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 创建配置对象并动态设置路径与超参数
    config = Config()
    config.input_dir = args.input_dir
    config.output_dir = args.output_dir
    # 根据输入目录自动拼接三个列表文件路径
    config.train_list_path = os.path.join(args.input_dir, 'train_list.txt')
    config.val_list_path = os.path.join(args.input_dir, 'val_list.txt')
    config.test_list_path = os.path.join(args.input_dir, 'test_list.txt')
    # 可覆盖的超参数
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.epochs = args.epochs
    config.patience = args.patience

    # 设置日志（输出目录会自动创建，若不存在）
    logger = setup_logging(config.output_dir)
    logger.info(f"Input directory: {config.input_dir}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Hyperparameters: batch_size={config.batch_size}, lr={config.lr}, "
                f"epochs={config.epochs}, patience={config.patience}")

    # 开始训练
    train(config, logger)