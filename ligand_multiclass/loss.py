# losses.py
import torch
import torch.nn.functional as F


def random_mask_atom(num_atoms, mask_ratio, device):
    """
    生成长度为 num_atoms 的布尔掩码，True 表示被遮蔽。
    """
    num_mask = int(num_atoms * mask_ratio)
    mask = torch.zeros(num_atoms, device=device)
    perm = torch.randperm(num_atoms, device=device)
    mask[perm[:num_mask]] = 1
    return mask.bool()


def contrastive_loss(z_atom, z_res, temperature=0.07):
    """
    图级对比损失 (对称 NT-Xent)。
    z_atom, z_res: (B, d) 已经过投影头的图表示。
    """
    z_atom = F.normalize(z_atom, dim=-1)
    z_res = F.normalize(z_res, dim=-1)
    logits = torch.mm(z_atom, z_res.t()) / temperature
    labels = torch.arange(z_atom.size(0), device=z_atom.device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.t(), labels)
    return (loss_1 + loss_2) / 2


def reconstruction_loss(type_logits, x_pred, atom_g, mask, lambda_coord=0.1):
    """
    掩码重建损失：被遮蔽原子的类型交叉熵 + 坐标 MSE。
    type_logits: (N_atom, num_types)
    x_pred:      (N_atom, 3)
    atom_g:      DGL 原子图（包含 ground truth 坐标和特征）
    mask:        (N_atom,) 布尔型，True 表示需重建的位置
    """
    # 从原始特征的前5维取 one-hot 索引
    true_types = atom_g.ndata['feat'][:, :5].argmax(dim=-1)
    loss_type = F.cross_entropy(type_logits[mask], true_types[mask])
    loss_coord = F.mse_loss(x_pred[mask], atom_g.ndata['x'][mask])
    loss_total = loss_type + lambda_coord * loss_coord
    return loss_total, loss_type, loss_coord