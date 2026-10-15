# train.py
import os
import logging
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import compute_global_stats, ProteinGraphDataset, collate_fn
from model import ARBindNet_Pretrain
from loss import contrastive_loss, reconstruction_loss


def setup_logging(output_dir):
    """创建日志记录器，输出到控制台和文件"""
    os.makedirs(output_dir, exist_ok=True)
    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'train.log'), mode='a')
        ]
    )
    return logging.getLogger(__name__)


def train(config, logger):
    """完整训练循环，包含验证与早停"""

    # Set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ---------- 读取文件列表 ----------
    with open(config.train_list_path, 'r') as f:
        train_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    logger.info(f"Number of training files: {len(train_files)}")

    with open(config.val_list_path, 'r') as f:
        val_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    logger.info(f"Number of validation files: {len(val_files)}")

    with open(config.test_list_path, 'r') as f:
        test_files = [line.strip() for line in f if line.strip().endswith('.hdf5')]
    logger.info(f"Number of test files: {len(test_files)}")

    # ===== 计算全局归一化统计量 =====
    stats = torch.load("/datapool/home/info_wang/tongyanting/GCL/model/0618/pretrain/output/norm_stats.pt")
    # stats = compute_global_stats(train_files)
    # torch.save(stats, os.path.join(config.output_dir, "norm_stats.pt"))
    # logger.info("Global normalization stats computed and saved.")
    # logger.info(f"Atom feature stats - Mean: {stats['atom_mean'].tolist()}, Std: {stats['atom_std'].tolist()}")
    # logger.info(f"Residue feature stats - Mean: {stats['res_mean'].tolist()}, Std: {stats['res_std'].tolist()}")

    # ---------- 构建数据集 ----------
    train_dataset = ProteinGraphDataset(train_files, stats)
    val_dataset = ProteinGraphDataset(val_files, stats)
    test_dataset = ProteinGraphDataset(test_files, stats)
    logger.info(f"Train dataset size: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=True
    )

    # ---------- 模型、优化器、学习率调度 ----------
    model = ARBindNet_Pretrain(config).to(config.device)
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ---------- 早停与最佳模型保存 ----------
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(config.output_dir, 'best_model.pt')

    for epoch in range(1, config.epochs + 1):
        # ===== 训练 =====
        model.train()
        total_train_loss = 0.0
        train_contrast_loss = 0.0
        train_recon_loss = 0.0
        train_type_loss = 0.0
        train_coord_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        #遍历batch
        for atom_g, res_g, atom_to_resid in pbar:
            atom_g = atom_g.to(config.device)
            res_g = res_g.to(config.device)
            atom_to_resid = atom_to_resid.to(config.device)
            #模型前向传播
            z_atom, z_res, type_logits, x_atom_pred, mask = model(atom_g, res_g, atom_to_resid)
            #loss计算
            loss_contrast = contrastive_loss(z_atom, z_res, config.temperature)
            loss_recon, loss_type, loss_coord = reconstruction_loss(
                type_logits, x_atom_pred, atom_g, mask, config.lambda_coord
            )
            loss = config.lambda_contrast * loss_contrast + config.lambda_recon * loss_recon
            #反向传播+参数更新
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #统计loss
            total_train_loss += loss.item()
            train_contrast_loss += loss_contrast.item()
            train_recon_loss += loss_recon.item()
            train_type_loss += loss_type.item()
            train_coord_loss += loss_coord.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'c': f"{loss_contrast.item():.4f}",
                't': f"{loss_type.item():.4f}",
                'x': f"{loss_coord.item():.4f}",
                'R': f"{loss_recon.item():.4f}"
            })
        #epoch平均loss
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_contrast = train_contrast_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_type = train_type_loss / len(train_loader)
        avg_train_coord = train_coord_loss / len(train_loader)

        # ===== 验证 =====
        model.eval()
        total_val_loss = 0.0
        val_contrast_loss = 0.0
        val_recon_loss = 0.0
        val_type_loss = 0.0
        val_coord_loss = 0.0

        with torch.no_grad():
            for atom_g, res_g, atom_to_resid in tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]"):
                atom_g = atom_g.to(config.device)
                res_g = res_g.to(config.device)
                atom_to_resid = atom_to_resid.to(config.device)

                z_atom, z_res, type_logits, x_atom_pred, mask = model(atom_g, res_g, atom_to_resid)

                loss_contrast = contrastive_loss(z_atom, z_res, config.temperature)
                loss_recon, loss_type, loss_coord = reconstruction_loss(
                    type_logits, x_atom_pred, atom_g, mask, config.lambda_coord
                )
                loss = config.lambda_contrast * loss_contrast + config.lambda_recon * loss_recon

                total_val_loss += loss.item()
                val_contrast_loss += loss_contrast.item()
                val_recon_loss += loss_recon.item()
                val_type_loss += loss_type.item()
                val_coord_loss += loss_coord.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_contrast = val_contrast_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_type = val_type_loss / len(val_loader)
        avg_val_coord = val_coord_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)

        # ===== 日志输出 =====
        logger.info(
            f"Epoch {epoch:3d} | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} "
            f"(C: {avg_train_contrast:.4f}, "
            f"T: {avg_train_type:.4f}, "
            f"X: {avg_train_coord:.4f}, "
            f"R: {avg_train_recon:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(C: {avg_val_contrast:.4f}, "
            f"T: {avg_val_type:.4f}, "
            f"X: {avg_val_coord:.4f}, "
            f"R: {avg_val_recon:.4f})"
        )

        # ===== 早停与保存最佳模型 =====
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, best_model_path)
            logger.info(f"  --> Validation loss improved. Best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"  --> No improvement. Early stopping counter: {patience_counter}/{config.patience}")
            if patience_counter >= config.patience:
                logger.info(f"Early stopping triggered. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break

    logger.info(f"Training finished. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
    logger.info("Loading best model for test evaluation...")
    checkpoint = torch.load(best_model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # ===== 测试 =====
    test(model, test_loader, config, logger)


def test(model, test_loader, config, logger):
    """在测试集上评估模型"""
    model.eval()

    total_test_loss = 0.0
    test_contrast_loss = 0.0
    test_recon_loss = 0.0
    test_type_loss = 0.0
    test_coord_loss = 0.0

    with torch.no_grad():
        for atom_g, res_g, atom_to_resid in tqdm(test_loader, desc="Testing"):
            atom_g = atom_g.to(config.device)
            res_g = res_g.to(config.device)
            atom_to_resid = atom_to_resid.to(config.device)

            z_atom, z_res, type_logits, x_atom_pred, mask = model(atom_g, res_g, atom_to_resid)

            loss_contrast = contrastive_loss(z_atom, z_res, config.temperature)
            loss_recon, loss_type, loss_coord = reconstruction_loss(
                type_logits, x_atom_pred, atom_g, mask, config.lambda_coord
            )
            loss = config.lambda_contrast * loss_contrast + config.lambda_recon * loss_recon

            total_test_loss += loss.item()
            test_contrast_loss += loss_contrast.item()
            test_recon_loss += loss_recon.item()
            test_type_loss += loss_type.item()
            test_coord_loss += loss_coord.item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_contrast = test_contrast_loss / len(test_loader)
    avg_test_recon = test_recon_loss / len(test_loader)
    avg_test_type = test_type_loss / len(test_loader)
    avg_test_coord = test_coord_loss / len(test_loader)

    logger.info(
        f"[TEST] Loss: {avg_test_loss:.4f} "
        f"(C: {avg_test_contrast:.4f}, "
        f"T: {avg_test_type:.4f}, "
        f"X: {avg_test_coord:.4f}, "
        f"R: {avg_test_recon:.4f})"
    )

    return avg_test_loss