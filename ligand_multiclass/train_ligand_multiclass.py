#!/usr/bin/env python
import argparse
import csv
import json
import logging
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import Config
from dataset_ligand_multiclass import LigandMulticlassDataset, collate_ligand_multiclass
from model_ligand_multiclass import PLiSAGE_LigandMulticlass


def setup_logger(output_dir):
    logger = logging.getLogger("ligand_multiclass")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = os.path.join(output_dir, "training.log")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_labels(split_file):
    labels = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labels.append(int(line.split()[1]))
    return labels


def build_class_weights(labels, num_classes, device):
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for cls in range(num_classes):
        weights.append(total / (num_classes * max(counts.get(cls, 0), 1)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loader(dataset, batch_size, num_workers, shuffle=False, weighted=False):
    if weighted:
        labels = [label for _, label in dataset.samples]
        counts = Counter(labels)
        sample_weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_ligand_multiclass,
            num_workers=num_workers,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_ligand_multiclass,
        num_workers=num_workers,
    )


def run_epoch(model, loader, criterion, optimizer, device, train, desc=None, show_progress=False):
    model.train(train)
    total_loss = 0.0
    all_labels, all_preds, all_probs, all_files = [], [], [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        loader_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            disable=not show_progress,
            dynamic_ncols=True,
        )
        for atom_g, res_g, atom_to_resid, labels, file_paths in loader_iter:
            atom_g = atom_g.to(device)
            res_g = res_g.to(device)
            atom_to_resid = atom_to_resid.to(device)
            labels = labels.to(device)

            logits = model(atom_g, res_g, atom_to_resid)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_files.extend(file_paths)

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, np.asarray(all_probs), all_files


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }


def save_predictions(path, files, labels, preds, probs, class_names):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "pred", "label_name", "pred_name", *[f"prob_{name}" for name in class_names]])
        for file_path, label, pred, prob in zip(files, labels, preds, probs):
            writer.writerow([file_path, label, pred, class_names[label], class_names[pred], *prob])


def make_best_model_path(output_dir, epoch, score):
    best_dir = os.path.join(output_dir, "best_models")
    os.makedirs(best_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"best_ligand_multiclass_model_epoch_{epoch:03d}_val_bal_acc_{score:.6f}_{timestamp}.pt"
    return os.path.join(best_dir, filename)


def make_epoch_model_path(output_dir, epoch, score):
    epoch_dir = os.path.join(output_dir, "epoch_models")
    os.makedirs(epoch_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"epoch_{epoch:03d}_val_bal_acc_{score:.6f}_{timestamp}.pt"
    return os.path.join(epoch_dir, filename)


def evaluate_and_save(model, test_loader, criterion, device, test_file, class_names, output_dir, best_epoch=None, best_score=None, logger=None):
    num_classes = len(class_names)
    test_metrics, test_probs, test_files = run_epoch(model, test_loader, criterion, None, device, train=False)
    test_labels = read_labels(test_file)
    test_preds = np.argmax(test_probs, axis=1).tolist()

    report = classification_report(test_labels, test_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))

    metrics_path = os.path.join(output_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        if best_epoch is not None:
            f.write(f"best_epoch: {best_epoch}\n")
        if best_score is not None:
            f.write(f"best_val_balanced_accuracy: {best_score:.6f}\n")
        for key, value in test_metrics.items():
            f.write(f"test_{key}: {value:.6f}\n")
        f.write("\nclassification_report:\n")
        f.write(report)
        f.write("\nconfusion_matrix:\n")
        f.write(str(cm))
        f.write("\n")

    np.savetxt(os.path.join(output_dir, "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
    predictions_path = os.path.join(output_dir, "test_predictions.csv")
    save_predictions(predictions_path, test_files, test_labels, test_preds, test_probs, class_names)

    if logger is not None:
        logger.info("Test metrics")
        for key, value in test_metrics.items():
            logger.info("  %s: %.4f", key, value)
        logger.info("\n%s", report)
        logger.info("saved metrics: %s", metrics_path)
        logger.info("saved predictions: %s", predictions_path)


def main():
    parser = argparse.ArgumentParser(description="Train ligand-category multiclass classifier.")
    parser.add_argument("--split_dir", required=True, help="Directory from make_ligand_multiclass_split.py.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--pretrained_model_path", help="Required for training. Pretrained backbone checkpoint path.")
    parser.add_argument("--test_only", action="store_true", help="Skip training and evaluate --checkpoint_path on the test split.")
    parser.add_argument("--checkpoint_path", help="Trained ligand multiclass checkpoint path. Required with --test_only.")
    parser.add_argument("--use_train_val", action="store_true", help="Train on train_val.txt instead of train.txt.")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gvp_hidden_v", type=int, default=32)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    args = parser.parse_args()

    if args.test_only and not args.checkpoint_path:
        parser.error("--checkpoint_path is required when --test_only is set.")
    if not args.test_only and not args.pretrained_model_path:
        parser.error("--pretrained_model_path is required for training.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    with open(os.path.join(args.split_dir, "class_mapping.json"), "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [mapping[str(i)] for i in range(len(mapping))]
    num_classes = len(class_names)

    config = Config()
    config.hidden_dim = args.hidden_dim
    config.gvp_hidden_v = args.gvp_hidden_v
    config.classifier_dropout = args.classifier_dropout
    config.device = str(device)

    stats = torch.load(args.stats_path, map_location="cpu")
    train_file = os.path.join(args.split_dir, "train_val.txt" if args.use_train_val else "train.txt")
    val_file = os.path.join(args.split_dir, "val.txt")
    test_file = os.path.join(args.split_dir, "test.txt")

    test_dataset = LigandMulticlassDataset(test_file, stats)

    logger.info("test samples: %d", len(test_dataset))
    logger.info("test class counts: %s", Counter(label for _, label in test_dataset.samples))

    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers, shuffle=False)

    model = PLiSAGE_LigandMulticlass(config, num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(device)

    if args.test_only:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        logger.info("loaded checkpoint for test only: %s", args.checkpoint_path)
        evaluate_and_save(model, test_loader, nn.CrossEntropyLoss(), device, test_file, class_names, args.output_dir, logger=logger)
        return

    train_dataset = LigandMulticlassDataset(train_file, stats)
    val_dataset = LigandMulticlassDataset(val_file, stats)
    logger.info("train samples: %d", len(train_dataset))
    logger.info("val samples: %d", len(val_dataset))
    logger.info("train class counts: %s", Counter(label for _, label in train_dataset.samples))

    train_loader = make_loader(train_dataset, args.batch_size, args.num_workers, shuffle=True, weighted=args.weighted_sampler)
    val_loader = make_loader(val_dataset, args.batch_size, args.num_workers, shuffle=False)

    model.load_pretrained_backbone(args.pretrained_model_path, map_location=device, logger=logger)

    class_weights = build_class_weights(read_labels(train_file), num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    best_score = -1.0
    best_epoch = 0
    stale = 0
    best_path = None
    history_path = os.path.join(args.output_dir, "training_history.csv")

    with open(history_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_bal_acc", "train_macro_f1", "val_loss", "val_bal_acc", "val_macro_f1", "val_weighted_f1"])

        epoch_iter = tqdm(
            range(1, args.epochs + 1),
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in epoch_iter:
            train_metrics, _, _ = run_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                train=True,
                desc=f"Train {epoch:03d}",
                show_progress=True,
            )
            val_metrics, _, _ = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                device,
                train=False,
                desc=f"Val {epoch:03d}",
                show_progress=True,
            )
            scheduler.step(val_metrics["balanced_accuracy"])

            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["balanced_accuracy"],
                val_metrics["macro_f1"],
                val_metrics["weighted_f1"],
            ])
            f.flush()

            score = val_metrics["balanced_accuracy"]
            epoch_path = make_epoch_model_path(args.output_dir, epoch, score)
            torch.save(model.state_dict(), epoch_path)
            logger.info(
                "Epoch %03d | train loss %.4f bal_acc %.4f macroF1 %.4f | val loss %.4f bal_acc %.4f macroF1 %.4f | saved epoch model: %s",
                epoch,
                train_metrics["loss"],
                train_metrics["balanced_accuracy"],
                train_metrics["macro_f1"],
                val_metrics["loss"],
                val_metrics["balanced_accuracy"],
                val_metrics["macro_f1"],
                epoch_path,
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale = 0
                best_path = make_best_model_path(args.output_dir, epoch, score)
                torch.save(model.state_dict(), best_path)
                logger.info("saved non-overwriting best model: %s", best_path)
            else:
                stale += 1
                if stale >= args.patience:
                    logger.info("early stopping at epoch %d", epoch)
                    break

    model.load_state_dict(torch.load(best_path, map_location=device))
    evaluate_and_save(model, test_loader, criterion, device, test_file, class_names, args.output_dir, best_epoch, best_score, logger=logger)


if __name__ == "__main__":
    main()
