# dataset.py
import os
import torch
import h5py
import dgl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ProteinGraphDataset(Dataset):
    """
    蛋白质图数据集：
    每个样本包含原子图、残基图以及原子到残基的映射索引。
    特征归一化依据全局统计量 stats。
    """

    def __init__(self, file_list, stats):
        self.file_list = file_list
        self.stats = stats

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        with h5py.File(path, 'r') as f:
            # ========== 原子图 ==========
            atom_grp = f['protein/atom_graph']
            atom_feat = torch.tensor(atom_grp['nodes/feat'][:]).float()
            # 连续特征归一化（第5-7维为坐标相关连续量）
            atom_cont = atom_feat[:, 5:8]
            atom_feat[:, 5:8] = (atom_cont - self.stats["atom_mean"]) / self.stats["atom_std"]
            atom_x = torch.tensor(atom_grp['nodes/x'][:]).float()
            atom_src = torch.tensor(atom_grp['edges/src'][:]).long()
            atom_dst = torch.tensor(atom_grp['edges/dst'][:]).long()
            atom_edge_attr = torch.tensor(atom_grp['edges/edge_attr'][:]).float()
            num_atoms = atom_feat.size(0)
            atom_g = dgl.graph((atom_src, atom_dst), num_nodes=num_atoms)
            atom_g.ndata['feat'] = atom_feat
            atom_g.ndata['x'] = atom_x
            atom_g.edata['edge_attr'] = atom_edge_attr

            # ========== 残基图 ==========
            res_grp = f['protein/residue_graph']
            res_feat = torch.tensor(res_grp['nodes/feat'][:]).float()
            res_feat = res_feat[:, :23]   # 仅使用前23维
            hp = res_feat[:, 20:22]       # 疏水性,极性
            rsasa = res_feat[:, 22:23]    # rsasa
            hp = (hp - self.stats["res_mean"]) / self.stats["res_std"]
            rsasa = torch.clamp(rsasa, 0, 1.5)
            res_feat[:, 20:23] = torch.cat([hp, rsasa], dim=1)
            res_x = torch.tensor(res_grp['nodes/x'][:]).float()
            res_vec = torch.tensor(res_grp['nodes/vec'][:]).float()
            res_frame = torch.tensor(res_grp['nodes/frame'][:]).float()
            res_src = torch.tensor(res_grp['edges/src'][:]).long()
            res_dst = torch.tensor(res_grp['edges/dst'][:]).long()
            res_edge_attr = torch.tensor(res_grp['edges/edge_attr'][:]).float()
            res_edge_vec = torch.tensor(res_grp['edges/edge_vec'][:]).float()
            num_res = res_feat.size(0)
            res_g = dgl.graph((res_src, res_dst), num_nodes=num_res)
            res_g.ndata['feat'] = res_feat
            res_g.ndata['x'] = res_x
            # 向量特征确保三维
            res_g.ndata['vec'] = res_vec.unsqueeze(-1) if res_vec.dim() == 2 else res_vec
            res_g.ndata['frame'] = res_frame
            res_g.edata['edge_attr'] = res_edge_attr
            res_g.edata['edge_vec'] = res_edge_vec.unsqueeze(-1) if res_edge_vec.dim() == 2 else res_edge_vec

            # ========== 原子到残基映射 ==========
            atom_to_resid = torch.tensor(f['protein/atom_residue_id'][:]).long()
            atom_to_resid = torch.clamp(atom_to_resid, 0, num_res - 1)

        return atom_g, res_g, atom_to_resid


def collate_fn(batch):
    """
    将多个蛋白质图合并为一个大图，同时修正原子→残基的索引偏移。
    """
    atom_graphs, res_graphs, atom_to_resid_list = zip(*batch)
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

    return batched_atom_g, batched_res_g, batched_atom_to_resid


def compute_global_stats(file_list):
    """
    遍历训练集所有文件，计算原子连续特征（第5-7维）的均值/标准差，
    以及残基疏水性特征（第20-21维）的均值/标准差，用于归一化。
    """
    atom_feats = []
    res_feats = []

    for path in tqdm(file_list, desc="Computing global stats"):
        with h5py.File(path, 'r') as f:
            atom_feat = torch.tensor(f['protein/atom_graph/nodes/feat'][:]).float()
            res_feat = torch.tensor(f['protein/residue_graph/nodes/feat'][:]).float()

            atom_feats.append(atom_feat[:, 5:8])
            res_feats.append(res_feat[:, 20:22])

    atom_feats = torch.cat(atom_feats, dim=0)
    res_feats = torch.cat(res_feats, dim=0)
    stats = {
        "atom_mean": atom_feats.mean(0),
        "atom_std": atom_feats.std(0) + 1e-8,
        "res_mean": res_feats.mean(0),
        "res_std": res_feats.std(0) + 1e-8,
    }
    return stats