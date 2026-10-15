import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix
)
from config import Config
from model_finetune import ARBindNet_Classifier
from dataset_finetune import ProteinBinaryDataset

# ========== 辅助函数 ==========
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'finetune.log'), mode='a')
        ]
    )
    return logging.getLogger(__name__)

#把一个 batch 里的多个蛋白样本，拼成一个“大图 + 对齐索引 + 标签”
def collate_fn_with_label(batch):
    atom_graphs, res_graphs, atom_to_resid_list, labels, file_paths= zip(*batch)
    batched_atom_g = dgl.batch(atom_graphs)
    batched_res_g = dgl.batch(res_graphs)

    new_atom_to_resid = []
    res_offset = 0
    for atom_g, res_g, atom_to_resid in zip(atom_graphs, res_graphs, atom_to_resid_list):
        num_res = res_g.num_nodes()
        global_res_id = atom_to_resid + res_offset
        new_atom_to_resid.append(global_res_id)
        res_offset += num_res
    batched_atom_to_resid = torch.cat(new_atom_to_resid, dim=0)

    # labels = torch.stack(labels)
    labels = torch.stack(labels).float()
    file_paths = [os.path.basename(p) for p in file_paths]
    return batched_atom_g, batched_res_g, batched_atom_to_resid, labels, file_paths

def load_label_dict(label_file):
    label_dict = {}
    with open(label_file, 'r') as f:
        for line in f:
            path, label = line.strip().split()
            label_dict[path] = int(label)
    return label_dict

def save_checkpoint(state, filepath):
    torch.save(state, filepath)

def load_checkpoint(filepath, model, optimizer, scheduler, device):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_auc = checkpoint.get('best_val_auc', 0.0)
    patience_counter = checkpoint.get('patience_counter', 0)
    # 恢复随机状态（可选，但强烈建议，保证可复现）
    random.setstate(checkpoint['random_state'])
    np.random.set_state(checkpoint['numpy_state'])
    torch.set_rng_state(checkpoint['torch_state'].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_state'])
    return start_epoch, best_val_auc, patience_counter


# ========== 训练函数 ==========
def train_finetune(config, logger):
    device = config.device
    # logger = setup_logging(config.output_dir)

    # 随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # 加载归一化统计量
    stats = torch.load("/datapool/home/info_wang/tongyanting/GCL/ARBindNet/model/pretrain/output/norm_stats.pt")

    # 读取文件列表
    with open(config.train_list_path, 'r') as f:
        train_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    with open(config.val_list_path, 'r') as f:
        val_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    with open(config.test_list_path, 'r') as f:
        test_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]

    # 加载标签
    train_labels = load_label_dict(config.train_label_file)
    val_labels = load_label_dict(config.val_label_file)
    test_labels = load_label_dict(config.test_label_file)

    # 数据集
    train_dataset = ProteinBinaryDataset(train_files, stats, train_labels)
    val_dataset = ProteinBinaryDataset(val_files, stats, val_labels)
    test_dataset = ProteinBinaryDataset(test_files, stats, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn_with_label, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate_fn_with_label, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_fn_with_label, num_workers=config.num_workers)

    train_labels_list = list(train_labels.values())
    num_pos = sum(train_labels_list)
    num_neg = len(train_labels_list) - num_pos
    logger.info(f"Training set statistics: Positive = {num_pos}, Negative = {num_neg}")
    # 模型
    model = ARBindNet_Classifier(
        config,
        pretrained_path=config.pretrained_model_path,
        freeze_encoders=config.freeze_encoders
    ).to(device)

    # 损失函数与优化器
    pos_weight = torch.tensor([2.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    #只训练需要梯度的参数
    params_to_train = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params_to_train, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    # ---------- 断点续训 ----------
    checkpoint_path = os.path.join(config.output_dir, 'checkpoint.pt')
    start_epoch = 1


    if os.path.exists(checkpoint_path):
        logger.info(f"Found checkpoint: {checkpoint_path}, resuming training...")
        start_epoch, best_val_auc, patience_counter = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, device
        )
        logger.info("=" * 50)
        logger.info("Resuming from checkpoint...")
        logger.info("=" * 50)
    else:
        logger.info("No checkpoint found, starting from scratch.")

    best_val_auc = 0.0
    best_model_path = os.path.join(config.output_dir, 'best_finetuned_model.pt')
    patience_counter = 0

    for epoch in range(start_epoch, config.finetune_epochs + 1):
        # ---------- 训练 ----------
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        for atom_g, res_g, atom_to_resid, labels, file_paths in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            atom_g = atom_g.to(device)
            res_g = res_g.to(device)
            atom_to_resid = atom_to_resid.to(device)
            labels = labels.to(device)

            logits = model(atom_g, res_g, atom_to_resid)
            # loss = criterion(logits, labels)
            logits = logits.squeeze(-1)
            loss = criterion(logits, labels)

            #反向传播,更新参数
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            #计算预测结果
            # preds = torch.argmax(logits, dim=1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            #收集预测 & 标签,用来最后算 accuracy
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        #计算训练集的accuracy和平均 loss
        train_acc = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)

        # ---------- 验证 ----------
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for atom_g, res_g, atom_to_resid, labels, file_paths in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                atom_g = atom_g.to(device)
                res_g = res_g.to(device)
                atom_to_resid = atom_to_resid.to(device)
                labels = labels.to(device)

                logits = model(atom_g, res_g, atom_to_resid)
                labels = labels.float().unsqueeze(1)
                loss = criterion(logits, labels)

                # loss = criterion(logits, labels)
                val_loss += loss.item()

                logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                all_val_probs.extend(probs.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(val_auc)

        logger.info(f"Epoch {epoch:3d} | Train Loss: {avg_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}")
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'patience_counter': patience_counter,
            'random_state': random.getstate(),
            'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state().cpu(),
            'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        save_checkpoint(checkpoint_data, checkpoint_path)
        logger.info(f"  Checkpoint saved at epoch {epoch}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  --> Saved best model with AUC {val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # ---------- 测试 ----------
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    test_preds, test_labels, test_probs, test_files = [], [], [], []

    with torch.no_grad():
        for atom_g, res_g, atom_to_resid, labels, file_paths in tqdm(test_loader, desc="Testing"):
            atom_g = atom_g.to(device)
            res_g = res_g.to(device)
            atom_to_resid = atom_to_resid.to(device)
            # labels = labels.to(device)
            labels = labels.float().to(device)
            logits = model(atom_g, res_g, atom_to_resid)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_files.extend(file_paths)
    # Metrics
    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_mcc = matthews_corrcoef(test_labels, test_preds)

    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info(f"Test Results:")
    logger.info(f"  Accuracy:    {test_acc:.4f}")
    logger.info(f"  AUC:         {test_auc:.4f}")
    logger.info(f"  Precision:   {test_precision:.4f}")
    logger.info(f"  Recall:      {test_recall:.4f}")
    logger.info(f"  F1 Score:    {test_f1:.4f}")
    logger.info(f"  MCC:         {test_mcc:.4f}")
    logger.info(f"  Specificity: {test_specificity:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # CSV 保存
    import pandas as pd

    df = pd.DataFrame({
        "file": test_files,
        "label": test_labels,
        "pred": test_preds,
        "prob": test_probs
    })

    csv_path = os.path.join(config.output_dir, "test_predictions(sample).csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Predictions saved to: {csv_path}")


def test_only(config, logger):
    """
    跳过训练和验证，直接加载 best_finetuned_model.pt 进行测试，
    并打印所有评估指标（包括 PR AUC），最后保存预测 CSV。
    """
    device = config.device
    logger.info("=" * 50)
    logger.info("Running in TEST-ONLY mode ...")
    logger.info("=" * 50)

    # 1. 加载归一化统计量（注意：路径与你训练时保持一致）
    stats_path = "/datapool/home/info_wang/tongyanting/GCL/model/0510/pretrain/output/norm_stats.pt"
    if not os.path.exists(stats_path):
        logger.error(f"Norm stats file not found: {stats_path}")
        return
    stats = torch.load(stats_path, map_location='cpu')

    # 2. 读取测试文件列表
    if not os.path.exists(config.test_list_path):
        logger.error(f"Test list file not found: {config.test_list_path}")
        return
    with open(config.test_list_path, 'r') as f:
        test_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]

    # 3. 加载测试标签
    if not os.path.exists(config.test_label_file):
        logger.error(f"Test label file not found: {config.test_label_file}")
        return

    test_label_dict = load_label_dict(config.test_label_file)

    # 4. 构建 DataLoader
    test_dataset = ProteinBinaryDataset(test_files, stats, test_label_dict)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_label,
        num_workers=config.num_workers
    )

    # 5. 初始化模型（结构参数必须与训练时完全一致）
    model = ARBindNet_Classifier(
        config,
        pretrained_path=None,
        freeze_encoders=config.freeze_encoders
    ).to(device)

    # 6. 加载训练好的最佳权重
    best_model_path = os.path.join(config.output_dir, 'best_finetuned_model.pt')
    # best_model_path = config.checkpoint_path
    if not os.path.exists(best_model_path):
        logger.error(f"Best model checkpoint not found: {best_model_path}")
        return
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    logger.info(f"Loaded best model weights from {best_model_path}")

    # 7. 开始测试
    test_preds = []
    test_labels_list = []
    test_probs = []
    test_files_list = []

    with torch.no_grad():
        for atom_g, res_g, atom_to_resid, labels, file_paths in tqdm(test_loader, desc="Testing"):
            atom_g = atom_g.to(device)
            res_g = res_g.to(device)
            atom_to_resid = atom_to_resid.to(device)
            labels = labels.float().to(device)

            logits = model(atom_g, res_g, atom_to_resid)
            logits = logits.squeeze(-1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
            test_files_list.extend(file_paths)

    # 8. 计算并打印所有指标
    from sklearn.metrics import average_precision_score

    test_acc = accuracy_score(test_labels_list, test_preds)
    test_auc = roc_auc_score(test_labels_list, test_probs)
    test_auprc = average_precision_score(test_labels_list, test_probs)
    test_precision = precision_score(test_labels_list, test_preds, zero_division=0)
    test_recall = recall_score(test_labels_list, test_preds, zero_division=0)
    test_f1 = f1_score(test_labels_list, test_preds, zero_division=0)
    test_mcc = matthews_corrcoef(test_labels_list, test_preds)

    tn, fp, fn, tp = confusion_matrix(test_labels_list, test_preds).ravel()
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info("Test Results:")
    logger.info(f"  Accuracy:    {test_acc:.4f}")
    logger.info(f"  AUC:         {test_auc:.4f}")
    logger.info(f"  PR AUC:      {test_auprc:.4f}")
    logger.info(f"  Precision:   {test_precision:.4f}")
    logger.info(f"  Recall:      {test_recall:.4f}")
    logger.info(f"  F1 Score:    {test_f1:.4f}")
    logger.info(f"  MCC:         {test_mcc:.4f}")
    logger.info(f"  Specificity: {test_specificity:.4f}")
    logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 9. 保存预测结果到 CSV
    import pandas as pd
    df = pd.DataFrame({
        "file": test_files_list,
        "label": test_labels_list,
        "pred": test_preds,
        "prob": test_probs
    })
    csv_path = os.path.join(config.output_dir, "test_predictions.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved to {csv_path}")
if __name__ == "__main__":
    config = Config()
    train_finetune(config)