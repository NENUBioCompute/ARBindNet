import h5py
import torch
import dgl
from torch.utils.data import Dataset


class LigandMulticlassDataset(Dataset):
    def __init__(self, split_file, stats):
        self.samples = []
        self.stats = stats
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split()[:2]
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with h5py.File(path, "r") as f:
            atom_grp = f["protein/atom_graph"]
            atom_feat = torch.tensor(atom_grp["nodes/feat"][:]).float()
            atom_cont = atom_feat[:, 5:8]
            atom_feat[:, 5:8] = (atom_cont - self.stats["atom_mean"]) / self.stats["atom_std"]
            atom_x = torch.tensor(atom_grp["nodes/x"][:]).float()
            atom_src = torch.tensor(atom_grp["edges/src"][:]).long()
            atom_dst = torch.tensor(atom_grp["edges/dst"][:]).long()
            atom_edge_attr = torch.tensor(atom_grp["edges/edge_attr"][:]).float()
            atom_g = dgl.graph((atom_src, atom_dst), num_nodes=atom_feat.size(0))
            atom_g.ndata["feat"] = atom_feat
            atom_g.ndata["x"] = atom_x
            atom_g.edata["edge_attr"] = atom_edge_attr

            res_grp = f["protein/residue_graph"]
            res_feat = torch.tensor(res_grp["nodes/feat"][:]).float()[:, :23]
            hp = res_feat[:, 20:22]
            rsasa = res_feat[:, 22:23]
            hp = (hp - self.stats["res_mean"]) / self.stats["res_std"]
            rsasa = torch.clamp(rsasa, 0, 1.5)
            res_feat[:, 20:23] = torch.cat([hp, rsasa], dim=1)
            res_x = torch.tensor(res_grp["nodes/x"][:]).float()
            res_vec = torch.tensor(res_grp["nodes/vec"][:]).float()
            res_frame = torch.tensor(res_grp["nodes/frame"][:]).float()
            res_src = torch.tensor(res_grp["edges/src"][:]).long()
            res_dst = torch.tensor(res_grp["edges/dst"][:]).long()
            res_edge_attr = torch.tensor(res_grp["edges/edge_attr"][:]).float()
            res_edge_vec = torch.tensor(res_grp["edges/edge_vec"][:]).float()
            res_g = dgl.graph((res_src, res_dst), num_nodes=res_feat.size(0))
            res_g.ndata["feat"] = res_feat
            res_g.ndata["x"] = res_x
            res_g.ndata["vec"] = res_vec.unsqueeze(-1) if res_vec.dim() == 2 else res_vec
            res_g.ndata["frame"] = res_frame
            res_g.edata["edge_attr"] = res_edge_attr
            res_g.edata["edge_vec"] = res_edge_vec.unsqueeze(-1) if res_edge_vec.dim() == 2 else res_edge_vec

            atom_to_resid = torch.tensor(f["protein/atom_residue_id"][:]).long()
            atom_to_resid = torch.clamp(atom_to_resid, 0, res_feat.size(0) - 1)

        return atom_g, res_g, atom_to_resid, torch.tensor(label, dtype=torch.long), path


def collate_ligand_multiclass(batch):
    atom_graphs, res_graphs, atom_to_resid_list, labels, file_paths = zip(*batch)
    batched_atom_g = dgl.batch(atom_graphs)
    batched_res_g = dgl.batch(res_graphs)

    new_atom_to_resid = []
    res_offset = 0
    for res_g, atom_to_resid in zip(res_graphs, atom_to_resid_list):
        new_atom_to_resid.append(atom_to_resid + res_offset)
        res_offset += res_g.num_nodes()

    return (
        batched_atom_g,
        batched_res_g,
        torch.cat(new_atom_to_resid, dim=0),
        torch.stack(labels),
        list(file_paths),
    )
