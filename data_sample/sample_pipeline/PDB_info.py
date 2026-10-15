# @Time    : 2022/8/24 13:11
# @Author  : Yihang Bao
# @FileName: PDB_info.py.py
# @Description:


import Bio.PDB
import numpy as np
import math
import os
# import Bio.PDBparser.ResidueDepth as rd
from Bio.PDB.ResidueDepth import *
from Bio.PDB.Polypeptide import is_aa
import time
import PDB.PDBparser.ParserStructure
import pickle
import threading
import queue
import multiprocessing as mp
import PDB
import warnings
import sys
sys.path.append('../')

def takeSecond(elem):
    return elem[1]

#用来读取pdb结构的文件
def parse(file):
    #创建一个PDBParser对象,Bio.PDB 模块中用于解析PDB文件的解析器,PERMISSIVE=1 表示采用宽松模式
    parser = Bio.PDB.PDBParser(PERMISSIVE=1)
    #调用解析器的get_structure方法，并传入两个参数，test为结构对象指定的标识符
    structure = parser.get_structure("test", file)
    #解析后返回一个 Structure 对象，它包含该 PDB 文件中的所有模型、链、残基和原子信息
    return structure

# PDBinfo类负责PDB的数据解析，通过调用PDBparser和Biopython完成数据解析并以pickle形式存储
class PDBInfo:
    #类属性
    structure = None
    site_info = None
    PDB_id = None
    def __init__(self, file, MSMS='msms'):
        #解析PDB文件
        structure = parse(file)
        #用MSMS计算分子表面
        surface = get_surface(structure[0], MSMS=MSMS)
        #获取所有残基
        residues = structure[0].get_residues()
        #为每个残基添加额外信息
        for residue in residues:
            residue.xtra["coord_center"] = self.res_coord_center(residue)
            residue.xtra["res_depth"] = residue_depth(residue, surface)
            residue.xtra["ca_depth"] = ca_depth(residue, surface)
            residue.xtra["ca_coord"] = self.ca_coord(residue)
        #保存结构对象
        self.structure = structure
        #获取活性位点信息
        self.site_info = self.get_site(file)
        #提取PDB_id
        self.PDB_id = file[-8:-4]


    def get_site(self, file):
        #创建一个自定义解析器ParserStructure，调用其parse方法解析pdb文件，返回一个字典dic
        ps = PDB.PDBparser.ParserStructure.ParserStructure()
        dic = ps.parse(file)
        #初始化site_info列表用于存储所有位点
        site_info = []
        #遍历dic['SITE']中的每个活性位点，PDB文件中每个SITE记录定义了一个活性位点
        for site in dic['SITE']:
            #对每个位点，初始化一个空列表site_res用于存放该位点对应的残基对象
            site_res = []
            count = 0
            #遍历该位点下的每个残基信息，将PDB文件中SITE记录里描述的活性位点信息残基与Bio.pdb结构对象中的实际残基进行匹配
            for res_info in site['resName']:
                #获取该链上的所有残基
                res_list = Bio.PDB.Selection.unfold_entities(self.structure[0][res_info['chainID']], 'R')
                #遍历该链上的所有残基对象
                for res in res_list:
                    #满足三个匹配条件：残基名，残基序号，插入码
                    if res_info['resName'] == res.get_resname().strip() and int(res_info['seq']) == res.get_id()[1] and res_info['iCode'] == res.get_id()[2].strip():
                        #判断是否为标准氨基酸残基
                        if is_aa(res, standard=True):
                            site_res.append(res)
            site['site_res'] = site_res
            #将匹配到的、且是标准氨基酸的残基对象添加到 site_res 列表中
            site_info.append(site)
        return site_info
        # site_dict = DataProcess.parse_site.get_site_info(file)
        # site_info = {}
        # for site, seqs in site_dict.items():
        #     site_res_list = []
        #     for seq in seqs:
        #         res_list = Bio.PDBparser.Selection.unfold_entities(self.structure[0][seq[1]], 'R')
        #         for res in res_list:
        #             if seq[0] == res.get_id()[1] and seq[2] == res.get_id()[2]:
        #                 seq[0] = (res.get_id()[0], res.get_id()[1], seq[2])
        #                 if is_aa(res):
        #                     site_res_list.append(res)
        #     site_info[site] = site_res_list
        # return site_info

    # def get_radius(self):
    #     radius_list = []
    #     for site, res_list in self.site_info.items():
    #         if len(res_list) == 0:
    #             continue
    #         site_center, radius = self.site_coord_center(res_list)
    #         # radius = math.ceil(radius)
    #         radius_list.append(radius)
    #     return radius_list
    #
    # def get_distance(self, coord1, coord2):
    #     diff = coord1 - coord2
    #     return numpy.sqrt(numpy.dot(diff, diff))
    #
    # def site_coord_center(self, res_list):
    #     site_center = np.array([0.0, 0.0, 0.0])
    #     coord_list = []
    #     for res in res_list:
    #         coord = res.xtra["coord_center"]
    #         coord_list.append(coord)
    #         site_center += coord
    #     site_center = site_center/len(res_list)
    #     radius = 0.0
    #     for coord in coord_list:
    #         distance = self.get_distance(coord, site_center)
    #         if distance > radius:
    #             radius = distance
    #     return site_center, radius
    #
#遍历残基中的所有原子，将所有原子的坐标在x,y,z上进行累加，将累加后的坐标除以原子总数，得到平均坐标，即几何中心
    def res_coord_center(self, res):
        coord = np.array([0.0, 0.0, 0.0])
        len = 0
        atom_list = res.get_atoms()
        for atom in atom_list:
            len += 1
            coord += atom.get_coord()
        coord /= len
        return coord
#获取残基中CA原子的坐标，如果有，就返回CA原子的三维坐标
    def ca_coord(self, res):
        if res.has_id("CA"):
            ca = res["CA"]
            return ca.get_coord()
        else:
            return None

    # def get_depth(self):
    #     depth_list = []
    #     for site, res_list in self.site_info.items():
    #         depth = 0
    #         for res in res_list:
    #             if depth < res.xtra["res_depth"]:
    #                 depth = res.xtra["res_depth"]
    #         depth_list.append(depth)
    #     return depth_list
    #
    # def get_close_res(self, n, max_depth, coord_center):
    #     res_list = []
    #     distance_list = []
    #     for res in Bio.PDBparser.Selection.unfold_entities(self.structure[0], "R"):
    #         if is_aa(res):
    #             res_center = self.res_coord_center(res)
    #             distance = self.get_distance(res_center, coord_center)
    #             distance_list.append((res, distance))
    #     distance_list.sort(key=takeSecond)
    #     for res in distance_list:
    #         print(res, file=data)
    #         print(res[0].xtra["res_depth"],file=data)
    #     i = 0
    #     while n > 0:
    #         if distance_list[i][0].xtra["res_depth"] <= max_depth:
    #             res_list.append(distance_list[i][0])
    #             n -= 1
    #         i += 1
    #     return res_list




if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    file_path = '/root/ServerTest/3941.pdb'
    res_info = PDBInfo(file_path, MSMS='msms')
    picklePath = file_path + '.pickle'
    f = open(picklePath, 'wb')
    pickle.dump(res_info, f)
    f.close()





# data = open(r"F:githubmy_codepdb118l.txt", "w")
# res_info = PDBInfo(r"H:iodataPDBfilepdbtest28pdb118l.ent")
# for site, res_list in res_info.site_info.items():
#     print(res_list, file=data)
#     center, radius = res_info.site_coord_center(res_list)
#     print(radius, file=data)
#     result_list = res_info.get_close_res(10, 6, center)
# res_info.print_center()