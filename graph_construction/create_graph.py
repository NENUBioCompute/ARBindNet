import h5py
from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein
import torch.nn.functional as F
import numpy as np
import torch
import warnings
import os
import dgl
import math
from Bio .PDB import is_aa
import torch
from Bio.PDB import Polypeptide
from scipy.spatial import distance
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit.Chem import MolFromPDBFile
from Bio.PDB import ShrakeRupley
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from multiprocessing import Pool, cpu_count
from functools import partial
import os
mass_ = np.array([1.008, 4.002602, 6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.998403163, 20.1797,
        22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948, 39.0983, 40.078,
        44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934, 63.546, 65.38,
        69.723, 72.63, 74.921595, 78.971, 79.904, 83.798, 85.4678, 87.62, 88.90584, 91.224, 92.90637,
        95.95, 97.0, 101.07, 102.9055, 106.42, 107.8682, 112.414, 114.818, 118.71, 121.76, 127.6,
        126.90447, 131.293, 132.90545196, 137.327, 138.90547, 140.116, 140.90766, 144.242, 145.0,
        150.36, 151.964, 157.25, 158.92535, 162.5, 164.93033, 167.259, 168.93422, 173.054, 174.9668,
        178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569, 200.592, 204.38,
        207.2, 208.9804, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.0377, 231.03588, 238.02891,
        237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0, 257.0, 258.0, 259.0, 262.0, 267.0, 270.0,
        271.0, 270.0, 277.0, 276.0, 281.0, 282.0, 285.0, 285.0, 289.0, 288.0, 293.0, 294.0, 294.0])

covRadius_ = np.array([0.32, 0.28, 1.34, 0.96, 0.84, 0.72, 0.68, 0.68, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11,
        1.036, 1.02, 1.02, 1.06, 2.03, 0.992, 1.70, 1.60, 1.53, 1.39, 1.19, 1.42, 1.11, 1.24, 1.32,
        1.448, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46,
        1.42, 1.39, 1.45, 1.668, 1.42, 1.39, 1.39, 1.38, 1.4, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
        1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, 1.87, 1.75, 1.70, 1.62, 1.51, 1.44,
        1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06, 2.00, 1.96,
        1.90, 1.87, 1.80, 1.69, 1.66, 1.68, 1.65, 1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34,
        1.29, 1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57])

vdwRadius_ = np.array([1.2, 1.4, 1.82, 2.0, 2.0, 1.7, 1.55, 1.52, 1.47, 1.54, 2.27, 1.73, 2.0, 2.1, 1.8, 1.8,
        1.75, 1.88, 2.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.4, 1.39, 1.87, 2.0, 1.85, 1.9,
        1.85, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.72, 1.58, 1.93, 2.17, 2.0, 2.06,
        1.98, 2.16, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.75, 1.66, 1.55, 1.96, 2.02, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 1.86, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])


hydrophobicity_index = {
        0: -3.2, 1: -3.5, 2: -4.5, 3: 2.8, 4: 1.8,
        5: 2.5, 6: -0.4, 7: -3.5, 8: -3.5, 9: -3.9,
        10: 3.8, 11: 1.9, 12: -3.5, 13: -0.8, 14: -1.3,
        15: -0.7, 16: 4.5, 17: -0.9, 18: -1.6, 19: 4.2}

amino_acid_polarity = {
    0: 0.5,   # HIS/HID/HIE/HIP (polar, partial +)
    1: -1.0,  # ASP/ASH
    2: 1.0,   # ARG/ARN
    3: 0.0,   # PHE
    4: 0.0,   # ALA
    5: 0.0,   # CYS/CYX
    6: 0.0,   # GLY
    7: 0.5,   # GLN
    8: -1.0,  # GLU/GLH
    9: 1.0,   # LYS/LYN
    10: 0.0,  # LEU
    11: 0.0,  # MET
    12: 0.5,  # ASN
    13: 0.5,  # SER
    14: 0.5,  # TYR
    15: 0.5,  # THR
    16: 0.0,  # ILE
    17: 0.0,  # TRP
    18: 0.0,  # PRO
    19: 0.0,  # VAL
    20: 0.0,  # SEC
    21: 0.0,  # PYL
    22: 0.0,  # ASX
    23: 0.0,  # XLE
    24: 0.0,  # GLX
    25: 0.0   # XXX
}

ATOM_MAP = {
        0: 0,  # H (Hydrogen)
        5: 1,  # C (Carbon)
        6: 2,  # N (Nitrogen)
        7: 3,  # O (Oxygen)
        15: 4  # S (Sulfur)
    }

ATOM_EDGE_TYPE = {
    'cov': 0,
    'hb': 1,
    'spatial': 2
}

RES_EDGE_TYPE = {
    'backbone': 0,
    'hb': 1,
    'spatial': 2
}
MAX_ASA = {
    0: 216,  # HIS
    1: 187,  # ASP
    2: 230,  # LYS
    3: 228,  # PHE
    4: 121,  # ALA
    5: 148,  # CYS
    6: 97,   # GLY
    7: 214,  # GLN
    8: 214,  # GLU
    9: 230,  # LYS
    10: 191, # LEU
    11: 203, # MET
    12: 187, # ASN
    13: 143, # SER
    14: 255, # TYR
    15: 163, # THR
    16: 195, # ILE
    17: 264, # TRP
    18: 154, # PRO
    19: 165  # VAL
}
# 工具：PDB计算
biopython_parser = PDBParser()
sr = ShrakeRupley(probe_radius=1.4, n_points=100)

#RBF编码本质上是把一个变量距离映射成一个向量
def rbf(dist, d_max, num_bins=16, d_min=0.0):
    dist = torch.clamp(dist, d_min, d_max)
    centers = torch.linspace(d_min, d_max, num_bins, device=dist.device)
    width = ((d_max - d_min) / num_bins) * 1.2

    return torch.exp(-((dist.unsqueeze(-1) - centers) ** 2) / (width ** 2))

def compute_gasteiger_charge(pdb_file):
    mol = MolFromPDBFile(pdb_file, removeHs=False)
    if mol is None:
        return None
    ComputeGasteigerCharges(mol)
    charges = []
    for atom in mol.GetAtoms():
        try:
            charges.append(float(atom.GetProp('_GasteigerCharge')))
        except:
            charges.append(0.0)

    return np.array(charges)

def build_radius_edges(pos, cutoff):
    dist_matrix = distance.cdist(pos.numpy(), pos.numpy())

    src, dst = [], []
    N = len(pos)

    for i in range(N):
        for j in range(N):
            if i != j and dist_matrix[i, j] < cutoff:
                src.append(i)
                dst.append(j)

    return torch.tensor(src).long(), torch.tensor(dst).long()


# 原子特征
def compute_atom_feat(hdf5_file):

    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    curProtein.load_hdf5(hdf5_file)

    if 'positive' in hdf5_file:
        pdb_file = '/datapool/home/info_wang/tongyanting/GCL/ARBindNet/data_sample/sample_output/domain_pdb/pos/' + hdf5_file.split('/')[-1].replace('hdf5', 'pdb')
    else:
        pdb_file = '/datapool/home/info_wang/tongyanting/GCL/ARBindNet/data_sample/sample_output/domain_pdb/neg/' + hdf5_file.split('/')[-1].replace('hdf5', 'pdb')

    atom_types = curProtein.atomTypes_
    atom_pos = curProtein.atomPos_.reshape(-1, 3)
    N = len(atom_types)
    num_classes = 5
    # 安全映射索引
    mapped_types = []
    for t in atom_types:
        mapped_types.append(ATOM_MAP.get(int(t), 5))
    mapped_types = torch.tensor(mapped_types, dtype=torch.long)

    # ===== 元素 =====
    atom_onehot = F.one_hot(mapped_types, num_classes=num_classes).float()
    # ===== 物理 =====
    atom_mass = torch.tensor(mass_[atom_types]).view(-1,1)
    atom_cov = torch.tensor(covRadius_[atom_types]).view(-1,1)
    atom_vdw = torch.tensor(vdwRadius_[atom_types]).view(-1,1)
    # ===== 拼接 =====
    h_feat = torch.cat([
        atom_onehot,
        atom_mass,
        atom_cov,
        atom_vdw
    ], dim=1)

    x_feat = torch.tensor(atom_pos).float()

    return h_feat, x_feat

def compute_residue_feat(hdf5_file):

    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    curProtein.load_hdf5(hdf5_file)

    if 'positive' in hdf5_file:
        pdb_file = '/datapool/home/info_wang/tongyanting/GCL/ARBindNet/data_sample/sample_output/domain_pdb/pos/' + hdf5_file.split('/')[-1].replace('hdf5', 'pdb')
    else:
        pdb_file = '/datapool/home/info_wang/tongyanting/GCL/ARBindNet/data_sample/sample_output/domain_pdb/neg/' + hdf5_file.split('/')[-1].replace('hdf5', 'pdb')

    res_type = curProtein.aminoType_

    # ===== one-hot =====
    res_onehot = F.one_hot(torch.tensor(res_type), num_classes=20).float()

    # ===== hydrophobicity / polarity =====
    hyd = torch.tensor([
        hydrophobicity_index.get(r, 0) for r in res_type
    ]).view(-1,1)

    polar = torch.tensor([
        amino_acid_polarity.get(r, 0) for r in res_type
    ]).view(-1,1)

    # # DIHEDRAL ANGLES (phi / psi)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('X', pdb_file)
        model = structure[0]

    # Build residue list + index map
    residues = []
    res_index_map = {}

    for chain in model:
        for res in chain:
            if not is_aa(res, standard=True):
                continue

            key = (chain.id, res.id[1])
            res_index_map[key] = len(residues)
            residues.append(res)

    N = len(residues)

    # init
    phi = torch.zeros(N)
    psi = torch.zeros(N)

    phi_mask = torch.zeros(N)
    psi_mask = torch.zeros(N)

    # dihedral function
    def calc_dihedral(p0, p1, p2, p3):
        import numpy as np

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1 /= np.linalg.norm(b1) + 1e-8

        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)

        return np.arctan2(y, x)

    # compute phi / psi
    for i, res in enumerate(residues):

        try:
            N_i = res["N"].get_vector().get_array()
            CA_i = res["CA"].get_vector().get_array()
            C_i = res["C"].get_vector().get_array()
        except:
            continue
        # ===== phi =====
        if i > 0:
            prev_res = residues[i - 1]
            # 同链 + 连续残基
            same_chain = (prev_res.get_parent().id == res.get_parent().id)
            continuous = (abs(prev_res.id[1] - res.id[1]) == 1)
            if same_chain and continuous:
                try:
                    C_prev = prev_res["C"].get_vector().get_array()
                    phi[i] = calc_dihedral(C_prev, N_i, CA_i, C_i)
                    phi_mask[i] = 1.0
                except:
                    pass

        # ===== psi =====
        if i < N - 1:
            next_res = residues[i + 1]

            same_chain = (next_res.get_parent().id == res.get_parent().id)
            continuous = (abs(next_res.id[1] - res.id[1]) == 1)

            if same_chain and continuous:
                try:
                    N_next = next_res["N"].get_vector().get_array()
                    psi[i] = calc_dihedral(N_i, CA_i, C_i, N_next)
                    psi_mask[i] = 1.0
                except:
                    pass
    # encoding
    phi_sin = torch.sin(phi)
    phi_cos = torch.cos(phi)
    psi_sin = torch.sin(psi)
    psi_cos = torch.cos(psi)

    mask_phi = phi_mask.unsqueeze(-1)
    mask_psi = psi_mask.unsqueeze(-1)

    # ===== RSASA =====
    sr = ShrakeRupley()
    sr.compute(model, level="R")
    res_rsasa_list = []
    for i, residue in enumerate(model.get_residues()):
        sasa = getattr(residue, "sasa", 0.0)
        res_idx = res_type[i] if i < len(res_type) else 0
        max_asa = MAX_ASA.get(res_idx, 200.0)
        rsasa = sasa / (max_asa + 1e-8)
        res_rsasa_list.append(rsasa)
    res_rsasa = torch.tensor(res_rsasa_list, dtype=torch.float32).view(-1, 1)

    # FEATURE CONCAT
    res_feat = torch.cat([
        res_onehot,
        hyd,
        polar,
        res_rsasa,
        phi_sin.view(-1,1),
        phi_cos.view(-1,1),
        psi_sin.view(-1,1),
        psi_cos.view(-1,1),
        mask_phi.view(-1,1),
        mask_psi.view(-1,1),
    ], dim=1)
    res_pos = torch.tensor(curProtein.aminoPos_[0]).float()

    # ===== 计算 Cα -> Cβ 方向 =====
    vec_list = []

    for res in residues:
        try:
            CA = res["CA"].get_vector().get_array()

            # backbone atoms
            N = res["N"].get_vector().get_array()
            C = res["C"].get_vector().get_array()

            # ===== pseudo CB construction =====
            b1 = CA - N
            b2 = C - CA

            # normal of peptide plane
            n = np.cross(b1, b2)
            n_norm = np.linalg.norm(n) + 1e-8
            n = n / n_norm

            b1_norm = b1 / (np.linalg.norm(b1) + 1e-8)

            # PLiSAGE-style constants (common approximation)
            CB = CA + 0.5 * b1_norm - 0.5 * n + 0.2 * np.cross(b1_norm, n)

            # direction
            v = CB - CA
            v = v / (np.linalg.norm(v) + 1e-8)

        except:
            # fallback：绝对不要直接全0（建议随机或 backbone方向）
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        vec_list.append(v)

    res_vec = torch.from_numpy(np.stack(vec_list)).float()

    # ===== 计算每个残基的局部标准正交框架 R_res =====
    frame_list = []
    for res in residues:
        try:
            N = res["N"].get_vector().get_array()
            CA = res["CA"].get_vector().get_array()
            C = res["C"].get_vector().get_array()

            # 主链方向 e1 = C - CA
            e1 = C - CA
            e1 = e1 / (np.linalg.norm(e1) + 1e-8)

            # 辅助向量 v2 = N - CA
            v2 = N - CA
            # Gram-Schmidt 正交化：e2 = v2 - (v2·e1)*e1，再归一化
            e2 = v2 - np.dot(v2, e1) * e1
            norm_e2 = np.linalg.norm(e2)
            if norm_e2 > 1e-6:
                e2 = e2 / norm_e2
            else:
                # 回退：若 N 和 C 共线（理论上不可能），构造任意垂直向量
                e2 = np.array([1.0, 0.0, 0.0]) if abs(e1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                e2 = e2 - np.dot(e2, e1) * e1
                e2 = e2 / (np.linalg.norm(e2) + 1e-8)

            # e3 = e1 × e2
            e3 = np.cross(e1, e2)
            e3 = e3 / (np.linalg.norm(e3) + 1e-8)

            # 旋转矩阵：列向量为基 (3,3)
            R = np.stack([e1, e2, e3], axis=1)  # 每一列是一个基向量

        except KeyError:
            # 如果缺少主链原子，使用单位矩阵
            R = np.eye(3, dtype=np.float32)

        frame_list.append(R)

    res_frame = torch.tensor(np.stack(frame_list, axis=0), dtype=torch.float32)  # (N_res, 3, 3)
    return res_feat, res_pos, res_vec, res_frame

def build_unified_edges(atom_pos, adj_cov, adj_hb, cutoff, edge_type_dict):
    src_sp, dst_sp = build_radius_edges(atom_pos, cutoff)

    edge_dict = {}

    def add_edges(src, dst, type_id):
        for i, j in zip(src.tolist(), dst.tolist()):
            key = (i, j)
            if key not in edge_dict:
                edge_dict[key] = torch.zeros(len(edge_type_dict))
            edge_dict[key][type_id] = 1.0

    add_edges(src_sp, dst_sp, edge_type_dict['spatial'])
    add_edges(adj_cov[:, 0], adj_cov[:, 1], edge_type_dict['cov'])
    add_edges(adj_hb[:, 0], adj_hb[:, 1], edge_type_dict['hb'])

    src, dst, type_feat = [], [], []

    for (i, j), t in edge_dict.items():
        src.append(i)
        dst.append(j)
        type_feat.append(t)

    src = torch.tensor(src)
    dst = torch.tensor(dst)
    type_feat = torch.stack(type_feat)

    vec = atom_pos[dst] - atom_pos[src]
    dist = torch.norm(vec, dim=-1, keepdim=True)
    # dir_vec = get_direction(vec)

    dist_feat = rbf(dist.squeeze(-1), d_max=5.0)

    edge_attr = torch.cat([dist_feat, type_feat], dim=-1)

    return src, dst, edge_attr

def create_atom_graph(hdf5_file):
    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    curProtein.load_hdf5(hdf5_file)

    atom_pos = torch.tensor(curProtein.atomPos_.reshape(-1, 3)).float()
    adj_cov = torch.tensor(curProtein.covBondList_)
    adj_hb  = torch.tensor(curProtein.covBondListHB_)

    src, dst, edge_attr = build_unified_edges(
        atom_pos,
        adj_cov,
        adj_hb,
        cutoff=5.0,
        edge_type_dict=ATOM_EDGE_TYPE
    )

    g = dgl.graph((src, dst), num_nodes=len(atom_pos))

    h_feat, x_feat = compute_atom_feat(hdf5_file)

    g.ndata['feat'] = h_feat
    g.ndata['x'] = x_feat
    g.edata['edge_attr'] = edge_attr

    return g

def build_unified_res_edges(res_pos, src_bb, dst_bb, src_hb, dst_hb, cutoff, edge_type_dict):
    src_sp, dst_sp = build_radius_edges(res_pos, cutoff)

    edge_dict = {}

    def add_edges(src, dst, type_id):
        for i, j in zip(src.tolist(), dst.tolist()):
            key = (i, j)
            if key not in edge_dict:
                edge_dict[key] = torch.zeros(len(edge_type_dict))
            edge_dict[key][type_id] = 1.0

    add_edges(src_bb, dst_bb, edge_type_dict['backbone'])
    add_edges(src_hb, dst_hb, edge_type_dict['hb'])
    add_edges(src_sp, dst_sp, edge_type_dict['spatial'])

    src, dst, type_feat = [], [], []

    for (i, j), t in edge_dict.items():
        src.append(i)
        dst.append(j)
        type_feat.append(t)

    src = torch.tensor(src)
    dst = torch.tensor(dst)
    type_feat = torch.stack(type_feat)

    vec = res_pos[dst] - res_pos[src]
    dist = torch.norm(vec, dim=-1, keepdim=True)
    # dir_vec = get_direction(vec)
    edge_vec = vec / (dist + 1e-8)

    dist_feat = rbf(dist.squeeze(-1), d_max=10.0)

    edge_attr = torch.cat([dist_feat, type_feat], dim=-1)

    return src, dst, edge_attr, edge_vec


def create_residue_graph(hdf5_file):

    periodicTable_ = PyPeriodicTable()
    curProtein = PyProtein(periodicTable_)
    curProtein.load_hdf5(hdf5_file)

    res_pos = torch.tensor(curProtein.aminoPos_[0]).float()
    N = len(res_pos)

    src_bb, dst_bb = zip(*curProtein.aminoNeighs_)
    src_hb, dst_hb = zip(*curProtein.aminoNeighsHB_)


    src_bb = torch.tensor(src_bb)
    dst_bb = torch.tensor(dst_bb)
    src_hb = torch.tensor(src_hb)
    dst_hb = torch.tensor(dst_hb)

    src, dst, edge_attr, edge_vec = build_unified_res_edges(
        res_pos,
        src_bb, dst_bb,
        src_hb, dst_hb,
        cutoff=10.0,
        edge_type_dict=RES_EDGE_TYPE
    )
    g = dgl.graph((src, dst), num_nodes=N)


    feat, x ,vec, frame = compute_residue_feat(hdf5_file)
    g.ndata['feat'] = feat
    g.ndata['x'] = x
    g.ndata['vec'] = vec
    g.ndata['frame'] = frame
    g.edata['edge_attr'] = edge_attr
    g.edata['edge_vec'] = edge_vec

    return g
def save_graph_to_hdf5(atom_graph, res_graph, filename, hdf5_path):

    with h5py.File(filename, 'w') as f:
        protein = f.create_group("protein")

        atom = protein.create_group("atom_graph")
        node = atom.create_group("nodes")

        for k, v in atom_graph.ndata.items():
            node.create_dataset(k, data=v.cpu().numpy())

        edge = atom.create_group("edges")
        edge.create_dataset("src", data=atom_graph.edges()[0].cpu().numpy())
        edge.create_dataset("dst", data=atom_graph.edges()[1].cpu().numpy())
        edge.create_dataset("edge_attr", data=atom_graph.edata['edge_attr'].cpu().numpy())

        res = protein.create_group("residue_graph")
        node = res.create_group("nodes")

        for k, v in res_graph.ndata.items():
            node.create_dataset(k, data=v.cpu().numpy())

        edge = res.create_group("edges")
        edge.create_dataset("src", data=res_graph.edges()[0].cpu().numpy())
        edge.create_dataset("dst", data=res_graph.edges()[1].cpu().numpy())
        edge.create_dataset("edge_attr", data=res_graph.edata['edge_attr'].cpu().numpy())
        edge.create_dataset("edge_vec",data=res_graph.edata['edge_vec'].cpu().numpy())
        # atom--residue
        periodicTable_ = PyPeriodicTable()
        curProtein = PyProtein(periodicTable_)

        # 读取原始数据
        curProtein.load_hdf5(hdf5_path)

        atom_residue_id = np.array(curProtein.atomResidueIds_, dtype=np.int32)
        atom_residue_names = np.array(curProtein.atomResidueNames_, dtype="S")

        # ===== 保存 =====
        protein.create_dataset(
            "atom_residue_id",
            data=atom_residue_id,
            compression="gzip"
        )

        if atom_residue_names is not None:
            protein.create_dataset(
                "atom_residue_names",
                data=atom_residue_names,
                compression="gzip"
            )

# ===== 单文件处理函数 =====
def process_single_file(hdf5_path, save_dir):
    try:
        filename = os.path.basename(hdf5_path)

        print(f"[PROCESS] {filename}")

        atom_graph = create_atom_graph(hdf5_path)
        res_graph = create_residue_graph(hdf5_path)

        save_path = os.path.join(save_dir, filename)

        save_graph_to_hdf5(atom_graph, res_graph, save_path, hdf5_path)

        return f"[OK] {filename}"

    except Exception as e:
        return f"[ERROR] {hdf5_path} | {str(e)}"

# ===== 从 txt 读取 =====
def process_from_txt(txt_path, save_dir, num_workers=8):

    os.makedirs(save_dir, exist_ok=True)
    with open(txt_path, "r") as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"Total files: {len(files)}")
    with Pool(processes=num_workers) as pool:
        results = pool.map(
            partial(process_single_file, save_dir=save_dir),
            files
        )
    for r in results:
        print(r)
# ===== 主函数 =====
if __name__ == "__main__":

    txt_path = "/datapool/home/info_wang/tongyanting/GCL/ARBindNet/graph_construction/hdf5_pos_list.txt"
    output_dir = "./domain_graph/pos"

    process_from_txt(
        txt_path,
        output_dir,
        num_workers=cpu_count() // 2
    )

# if __name__ == "__main__":
#     filename = '1aig_5_positive.hdf5'
#     hdf5_path = os.path.join('/datapool/home/info_wang/tongyanting/GCL/alpha_data/domain_hdf5/pos/',filename)
#     print(hdf5_path)
#     atom_graph = create_atom_graph(hdf5_path)
#     res_graph = create_residue_graph(hdf5_path)
#     print(atom_graph)
#     print(res_graph)
#     save_graph_to_hdf5(
#         atom_graph,
#         res_graph,
#         f'/datapool/home/info_wang/tongyanting/GCL/alpha_data/create_graph_test/{filename}',
#         hdf5_path
#     )




