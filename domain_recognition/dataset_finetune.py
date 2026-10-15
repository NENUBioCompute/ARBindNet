import h5py
import torch
import dgl
from torch.utils.data import Dataset

class ProteinBinaryDataset(Dataset):
    def __init__(self, file_list, stats, label_dict):
        """
        file_list: list of hdf5 paths
        stats: dict with 'atom_mean', 'atom_std', 'res_mean', 'res_std'
        label_dict: dict mapping path -> int label (0/1)
        """
        self.file_list = file_list
        self.stats = stats
        self.label_dict = label_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.label_dict[path]

        with h5py.File(path, 'r') as f:
            # 原子图
            atom_grp = f['protein/atom_graph']
            atom_feat = torch.tensor(atom_grp['nodes/feat'][:]).float()
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

            # 残基图
            res_grp = f['protein/residue_graph']
            res_feat = torch.tensor(res_grp['nodes/feat'][:]).float()[:, :23]
            hp = res_feat[:, 20:22]
            rsasa = res_feat[:, 22:23]
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
            res_g.ndata['vec'] = res_vec.unsqueeze(-1) if res_vec.dim()==2 else res_vec
            res_g.ndata['frame'] = res_frame
            res_g.edata['edge_attr'] = res_edge_attr
            res_g.edata['edge_vec'] = res_edge_vec.unsqueeze(-1) if res_edge_vec.dim()==2 else res_edge_vec

            # 映射
            atom_to_resid = torch.tensor(f['protein/atom_residue_id'][:]).long()
            atom_to_resid = torch.clamp(atom_to_resid, 0, num_res-1)

        return atom_g, res_g, atom_to_resid, torch.tensor(label, dtype=torch.long), path