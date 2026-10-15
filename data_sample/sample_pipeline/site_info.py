# @Time    : 2022/8/24 11:02
# @Author  : Yihang Bao
# @FileName: site_info.py
# @Description:


import Bio.PDB
import numpy as np
from Bio.PDB.ResidueDepth import *
from Bio.PDB.Polypeptide import is_aa
import PDB.PDBparser.ParserStructure
from PDB.PDBparser.CallParser import callParser
import pickle
import threading
import queue
import numpy
import multiprocessing as mp
from PDB.DataProcess.PDB_info import PDBInfo
import random
import re
import csv

# site_info类用于处理单个数据，包含各种处理单个数据的方法
class Site_info:
    def __init__(self, PDB_info, pdb_path):
        #保存传入的 PDBInfo 对象，包含完整的结构信息和活性位点信息
        self.file_path = None
        self.PDB_info = PDB_info
        self.pdb_path = pdb_path
        self.add_site_binding()
        self.site_res_check()
        self.get_site_center_res_coord()


#用于清理活性位点中的无效残基，通过检查每个残基的 ca_coord是否存在，来判断残基是否有效，如果某个位点所有残基均被移除，则该位点整体也被删除。
    def site_res_check(self):
        del_site_list = []
        for site in self.PDB_info.site_info:
            del_res_list = []
            for res in site['site_res']:
                if res.xtra['ca_coord'] is None:
                    del_res_list.append(res)
            for res in del_res_list:
                site['site_res'].remove(res)
            if len(site['site_res']) == 0:
                del_site_list.append(site)
        for site in del_site_list:
            self.PDB_info.site_info.remove(site)

#从 PDB 文件的 REMARK 800 记录中提取活性位点的补充信息，并将这些信息添加到self.PDB_info.site_info 中对应的位点字典中
    # def add_site_binding(self):
    #     #自定义解析器callparser，调用parser方法ParserStructure解析pdb文件
    #     callparser = callParser()
    #     ps = callparser.parser(self.pdb_path)
    #     content = None
    #     #遍历 ps['REMARK'] 列表，找到 remarkNum == 800 的分组，将其文本行列表存入 content
    #     for remark in ps['REMARK']:
    #         if remark['remarkNum'] == 800:
    #             content = remark['empty']
    #     if content is not None:
    #         for i, line in enumerate(content):
    #             #当某行以SITE_IDENTIFIER（前15个字符）开头，提取位点标识符（第 17 到 20 个字符），三字母代码
    #             if line[0:15] == 'SITE_IDENTIFIER':
    #                 SITE_IDENTIFIER = line[17:20]
    #                 #然后根据下一行（content[i+1]）的内容判断证据代码和位点描述的位置
    #                 if content[i+1][:13] == 'EVIDENCE_CODE':
    #                     EVIDENCE_CODE = content[i+1][15::]
    #                     if content[i + 2][:16] == 'SITE_DESCRIPTION':
    #                         SITE_DESCRIPTION = content[i + 2][18::]
    #                     else:
    #                         SITE_DESCRIPTION
    #                 elif content[i + 1][:16] == 'SITE_DESCRIPTION':
    #                     SITE_DESCRIPTION = content[i + 1][18::]
    #                     EVIDENCE_CODE = ''
    #                 else:
    #                     EVIDENCE_CODE = ''
    #                     SITE_DESCRIPTION = ''
    #
    #                 #遍历self.PDB_info.site_info列表，找到 siteID 与提取的 SITE_IDENTIFIER 匹配的位点，为其添加 'EVIDENCE_CODE' 和 'SITE_DESCRIPTION' 键值对。
    #                 for site in self.PDB_info.site_info:
    #                     if site['siteID'] == SITE_IDENTIFIER:
    #                         site['EVIDENCE_CODE'] = EVIDENCE_CODE
    #                         site['SITE_DESCRIPTION'] = SITE_DESCRIPTION
    import re

    def add_site_binding(self):
        """
        解析 REMARK 800，更新 self.PDB_info.site_info：
        - 添加 'EVIDENCE_CODE'
        - 添加 'SITE_DESCRIPTION'
        - 提取 'binding_ligand'（结合物质三字母代码）
        """
        callparser = callParser()
        ps = callparser.parser(self.pdb_path)
        content = None

        # 找到 REMARK 800
        for remark in ps['REMARK']:
            if remark['remarkNum'] == 800:
                content = remark['empty']
                break

        if content is None:
            return

        for i, line in enumerate(content):
            if line[0:15] == 'SITE_IDENTIFIER':
                SITE_IDENTIFIER = line[17:20]

                # 解析 EVIDENCE_CODE 和 SITE_DESCRIPTION
                EVIDENCE_CODE = ''
                SITE_DESCRIPTION = ''
                if i + 1 < len(content):
                    next_line = content[i + 1]
                    if next_line.startswith('EVIDENCE_CODE'):
                        EVIDENCE_CODE = next_line[15:].strip()
                        if i + 2 < len(content) and content[i + 2].startswith('SITE_DESCRIPTION'):
                            SITE_DESCRIPTION = content[i + 2][18:].strip()
                    elif next_line.startswith('SITE_DESCRIPTION'):
                        SITE_DESCRIPTION = next_line[18:].strip()

                # 提取结合物质（第一个三字母代码）
                binding_ligand = ''
                SITE_DESCRIPTION_UPPER = SITE_DESCRIPTION.upper()
                match = re.search(r'FOR RESIDUE\s+([A-Z0-9]{2,3})', SITE_DESCRIPTION_UPPER)
                if match:
                    binding_ligand = match.group(1)
                else:
                    # 如果没有匹配到三字母代码，就用原始描述（建议用原始大小写更好）
                    binding_ligand = SITE_DESCRIPTION.strip()
                # match = re.search(r'FOR RESIDUE\s+([A-Z0-9]{2,3})', SITE_DESCRIPTION.upper())
                # if match:
                #     binding_ligand = match.group(1)

                # 更新 site_info
                for site in self.PDB_info.site_info:
                    if site['siteID'] == SITE_IDENTIFIER:
                        site['EVIDENCE_CODE'] = EVIDENCE_CODE
                        site['SITE_DESCRIPTION'] = SITE_DESCRIPTION
                        site['binding_ligand'] = binding_ligand  # 新增字段

    def save_binding_ligands_csv(self, dataset, csv_path):
        fieldnames = ['pdb_file', 'site_id', 'binding_ligand']

        file_exists = os.path.exists(csv_path)

        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # 只有第一次写表头
            if not file_exists:
                writer.writeheader()

            for domain in dataset:
                if domain.get('isPositive', 0) != 1:
                    continue

                writer.writerow({
                    'pdb_file': domain.get('domain_id', ''),
                    'site_id': domain.get('siteID', ''),
                    'binding_ligand': domain.get('binding_ligand', '')
                })
    # def save_binding_ligands_csv(self, dataset, csv_path):
    #     """
    #     :param dataset: get_dataset() 返回的列表（正负样本）
    #     :param csv_path: 输出 CSV 路径
    #     """
    #     # print(type(dataset[0]), dataset[0])
    #     fieldnames = ['pdb_file','site_id', 'binding_ligand']
    #
    #     with open(csv_path, 'w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=fieldnames)
    #         writer.writeheader()
    #
    #         for domain in dataset:
    #             # 只保留正样本
    #             if domain.get('isPositive', 0) != 1:
    #                 continue
    #
    #             # # 结合域文件名（你在保存pdb时应该有记录）
    #             # # pdb_file_name = domain.get('domain_pdb_name', '')
    #             # pdb_file_name = domain.get('domain_id', '')
    #             pdb_file_name = domain.get('domain_id','')
    #             site_id = domain.get('siteID', '')
    #             # 结合配体
    #             ligand = domain.get('binding_ligand', '')
    #
    #             writer.writerow({
    #                 'pdb_file': pdb_file_name,
    #                 'binding_ligand': ligand,
    #                 'site_id': site_id
    #             })
    # def get_site_num(self):
    #     return len(self.PDB_info.site_info)
    #
    # def get_site_res_num(self):
    #     site_res_list = []
    #     for site in self.PDB_info.site_info:
    #         site_res_list.append(len(site['site_res']))
    #     return site_res_list
    #
    # def get_site_info(self):
    #     site_list = []
    #     for i, site in enumerate(self.PDB_info.site_info):
    #         if len(site['site_res']) == 0:
    #             continue
    #         site_res_coord = self.get_site_res_coord(site['site_res'])
    #         site_res_coord = self.get_relative_dis(site_res_coord, site['center_res_ca_coord'])
    #         site_res_coord = self.get_relative_depth(site_res_coord, site['center_res_ca_depth'])
    #         site_res_coord = self.get_res_distances(site_res_coord)
    #         site_list.append(site_res_coord)
    #     # for site in site_list:
    #     #     for res in site:
    #     #         print(res)
    #     return site_list

#将残基列表转化为可供机器学习模型使用的数值特征矩阵
    def get_site_res_info(self, res_list, center_res_ca_coord, center_res_ca_depth):
        site_res_coord = self.get_site_res_coord(res_list)
        site_res_coord = self.get_relative_dis(site_res_coord, center_res_ca_coord)
        site_res_coord = self.get_relative_depth(site_res_coord, center_res_ca_depth)
        site_res_coord = self.get_res_distances(site_res_coord)
        return site_res_coord
#从残基对象列表中提取基础特征，并将每个残基的特征封装为一个字典
    def get_site_res_coord(self, res_list):
        site_res_coord = []
        for res in res_list:
            res_info_dict = {}
            #残基几何中心坐标
            res_info_dict['coord_center'] = res.xtra['coord_center']
            #Cα 原子坐标
            res_info_dict['ca_coord'] = res.xtra['ca_coord']
            #残基深度（残基中心到表面距离）
            res_info_dict['res_depth'] = res.xtra['res_depth']
            # Cα 原子深度
            res_info_dict['ca_depth'] = res.xtra['ca_depth']
            #残基三字母名称（如 'ALA'）
            res_info_dict['res_name'] = res.get_resname()
            #将当前残基的特征字典添加到列表中
            site_res_coord.append(res_info_dict)
        return site_res_coord
#计算每个残基的 Cα 坐标相对于活性位点中心 Cα 坐标的偏移向量
    def get_relative_dis(self, site_res_coord, center_res_ca_coord):
        for res in site_res_coord:
            res['relative_ca_coord'] = res['ca_coord'] - center_res_ca_coord
        return site_res_coord
#计算每个残基的 Cα 深度相对于活性位点中心残基 Cα 深度的绝对差值
    def get_relative_depth(self, site_res_coord, center_res_ca_depth):
        for res in site_res_coord:
            res['relative_ca_depth'] = abs(res['ca_depth'] - center_res_ca_depth)
        return site_res_coord
#计算每个残基到活性位点中心的欧氏距离
    def get_res_distances(self, site_res_coord):
        center_coord = np.array([0.0, 0.0, 0.0])
        res_dis_list = []
        ca_dis_list = []
        for res in site_res_coord:
            # res_dis = self.get_distance(res['relative_coord_center'], center_coord)
            ca_dis = self.get_distance(res['relative_ca_coord'], center_coord)
            # res_dis_list.append(res_dis)
            # res['res_dis'] = res_dis
            res['ca_dis'] = ca_dis
            ca_dis_list.append(ca_dis)
        # site['res_dis'] = res_dis_list
        # site['ca_dis'] = ca_dis_list
        return site_res_coord
#计算每个活性位点的几何中心（基于 Cα 坐标的平均值），找出距离该中心最近的残基，并保存该残基的 Cα 坐标和深度信息
    def get_site_center_res_coord(self):
        #遍历 PDB_info 对象中存储的每个活性位点，跳过空位点
        for site in self.PDB_info.site_info:
            if len(site['site_res']) == 0:
                continue
            #初始化ca_center
            ca_center = np.array([0.0, 0.0, 0.0])
            #遍历位点中的所有残基，累加每个残基的 Cα 坐标，累加后除以残基数，得到平均坐标，即该位点的中心点
            for res in site['site_res']:
                # print(type(ca_center))
                # print(type(res.xtra['ca_coord']))
                ca_center += res.xtra['ca_coord']
            ca_center /= len(site['site_res'])
            #计算每个残基的ca到ca_center的欧式距离，将 (res, dis) 元组添加到列表
            res_dis_list = []
            for res in site['site_res']:
                dis = self.get_distance(ca_center, res.xtra['ca_coord'])
                res_dis_list.append((res, dis))
            #按距离从小到大排序
            res_dis_list.sort(key=lambda x: x[1])
            #存储最近残基的信息（ca_coord和ca_depth）
            center_res_ca_coord = res_dis_list[0][0].xtra['ca_coord']
            site['center_res_ca_coord'] = center_res_ca_coord
            center_res_ca_depth = res_dis_list[0][0].xtra['ca_depth']
            site['center_res_ca_depth'] = center_res_ca_depth
            # print(center_res_ca_coord)

    # input  res_list return res_center_coord res_center_depth
    def _get_center_depth(self, res_list):
        ca_center = np.array([0.0, 0.0, 0.0])
        for res in res_list:
            # print(type(ca_center))
            # print(type(res.xtra['ca_coord']))
            ca_center += res.xtra['ca_coord']
        ca_center /= len(res_list)
        res_dis_list = []
        for res in res_list:
            dis = self.get_distance(ca_center, res.xtra['ca_coord'])
            res_dis_list.append((res, dis))
        res_dis_list.sort(key=lambda x: x[1])
        center_res_ca_coord = res_dis_list[0][0].xtra['ca_coord']
        center_res_ca_depth = res_dis_list[0][0].xtra['ca_depth']
        return center_res_ca_depth, center_res_ca_coord

    def get_close_res(self):
        for site in self.PDB_info.site_info:
            print('site')
            center_coord = site['center_res_ca_coord']
            res_list = self._get_close_res(center_coord, 15, 5)
            for res in res_list:
                print(res.get_id())
            print('site_res')
            for res in site['site_res']:
                if res in res_list:
                    print('yes')
                else:
                    print('no')

    def test_cover_rate(self, site_res, n, max_depth):
        correct_count = 0
        total_count = 0
        for site in self.PDB_info.site_info:
            if len(site['site_res']) == site_res:
                center_coord = site['center_res_ca_coord']
                for res in site['site_res']:
                    if all(res.xtra['ca_coord'] == center_coord):
                        center_res_depth = res.xtra['ca_depth']
                res_list = self._get_close_res(center_coord, n, max_depth, center_res_depth)
                for res in site['site_res']:
                    total_count += 1
                    if res in res_list:
                        correct_count += 1
        return correct_count, total_count

#从整个蛋白质结构中选取距离活性位点中心最近且深度与中心深度相近的残基
    def _get_close_res(self, center_coord, n, max_depth, center_res_depth, res_list):
        #存储所有残基及其到中心的距离
        res_dis_list = []
        #只保留标准氨基酸，同时跳过没有CA坐标的残基
        for res in self.PDB_info.structure[0].get_residues():
            if not is_aa(res, standard=True):
                continue
            if res.xtra['ca_coord'] is None:
                continue
            dis = self.get_distance(center_coord, res.xtra['ca_coord'])
            res_dis_list.append((res, dis))
        # 按距离升序排列
        res_dis_list.sort(key=lambda x: x[1])
        count = n
        for res_dis in res_dis_list:
            #计算该残基深度与中心深度的绝对差
            relative_depth = abs(res_dis[0].xtra['ca_depth'] - center_res_depth)
            #将符合条件的残基加入列表
            if count > 0 and relative_depth <= max_depth and res_dis[0] not in res_list:
                res_list.append(res_dis[0])
                count -= 1
            else:
                continue
        return res_list

    def positive_set_count(self, max_numres):
        all_site_count = 0
        site_count = 0
        for site in self.PDB_info.site_info:
            all_site_count += 1
            if site['numRes'] > max_numres:
                pass
            else:
                site_count += 1
        print(site_count)
        return all_site_count, site_count


#根据活性位点信息生成正样本数据集，每个正样本对应一个活性位点
    def get_positive_set(self, num, max_depth, max_numres, file_path):
        set_list = []
        out_file_no=0
        #遍历所有活性位点，过滤残基数过多的位点
        for site in self.PDB_info.site_info:
            if site['numRes'] > max_numres:
                continue
            site_dict = {}
            #需要补充的残基数
            count = num
            #最终样本的残基列表
            res_list = []
            #位点中心及深度
            center = site['center_res_ca_coord']
            center_res_depth = site['center_res_ca_depth']
            #将位点本身的信息全部加入列表
            for res in site['site_res']:
                res_list.append(res)
                count -= 1
            #从整个蛋白质结构中选取距离中心最近且深度差在允许范围内的残基，补充到res_list中
            res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
            #存储位点的标识符。如‘AC1’
            site_dict['siteID'] = site['siteID']
            #存储位点在site记录中标注的原始残基总数
            site_dict['numRes'] = site['numRes']
            #存储最终用于该样本的残基对象列表（包括位点自身残基和补充的邻近残基）
            site_dict['site_res'] = res_list
            #存储从 REMARK 800 中提取的EVIDENCE_CODE
            site_dict['EVIDENCE_CODE'] = site['EVIDENCE_CODE']
            # 存储从 REMARK 800 中提取的SITE_DESCRIPTION
            site_dict['SITE_DESCRIPTION'] = site['SITE_DESCRIPTION']
            site_dict['binding_ligand'] = site.get('binding_ligand', '')
            # site_dict['domain_id'] = f"{self.pdb_path.split('/')[-1].split('.')[0]}_{site['siteID']}"
            out_file_no += 1
            pdb_name = file_path.split('.')[-2].split('/')[-1]
            # pdb_save_path = self.w_dir + 'domain_pdb/' + \
            #                 f"{pdb_name}_{out_file_no}_poitive.pdb"
            site_dict['domain_id'] = f"{pdb_name}_{out_file_no}_positive.pdb"
            #存储位点的中心坐标
            site_dict['center_res_ca_coord'] = center
            #存储中心残基的 Cα_depth
            site_dict['center_res_ca_depth'] = center_res_depth
            #为 res_list 中的每个残基提取数值特征
            site_dict['res_info'] = self.get_site_res_info(res_list, center, center_res_depth)

            #设置标签为 1，表示这是一个正样本
            site_dict['isPositive'] = 1
            #将该位点对应的样本字典添加到正样本列表 set_list 中
            set_list.append(site_dict)
        return set_list

    # def get_negative_set(self, num, max_depth, max_numres):
    #     set_list = []
    #     exist_res_list = []
    #
    #     # 1. 收集所有正样本中用到的残基（活性位点及其邻近区域）
    #     for site in self.PDB_info.site_info:
    #         if site['numRes'] > max_numres:
    #             continue
    #         count = num
    #         res_list = []
    #         center = site['center_res_ca_coord']
    #         center_res_depth = site['center_res_ca_depth']
    #         for res in site['site_res']:
    #             res_list.append(res)
    #             count -= 1
    #         res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
    #         exist_res_list += res_list
    #
    #     # 2. 遍历所有残基，构建负样本
    #     for res in self.PDB_info.structure[0].get_residues():
    #         # 跳过已被正样本占用的残基以及非标准氨基酸
    #         if res in exist_res_list or not is_aa(res, standard=True):
    #             continue
    #
    #         center_res_ca_coord = res.xtra['ca_coord']
    #         center_res_ca_depth = res.xtra['ca_depth']
    #         res_list = []
    #         site_res_list = self._get_close_res(center_res_ca_coord, num, max_depth, center_res_ca_depth, res_list)
    #
    #         # 检查补充的残基是否与正样本区域重叠
    #         is_available_flag = True
    #         for site_res in site_res_list:
    #             if site_res in exist_res_list:
    #                 is_available_flag = False
    #                 break
    #         if not is_available_flag:
    #             continue
    #
    #         # 构建负样本字典
    #         site_dict = {}
    #         site_dict['site_res'] = res_list
    #         site_dict['center_res_ca_coord'] = center_res_ca_coord
    #         site_dict['center_res_ca_depth'] = center_res_ca_depth
    #         site_dict['res_info'] = self.get_site_res_info(res_list, center_res_ca_coord, center_res_ca_depth)
    #         site_dict['isPositive'] = 0
    #
    #         # 附加检查：样本大小必须恰好等于 num
    #         if len(res_list) != num:
    #             continue
    #
    #         # 去重：避免生成完全相同的负样本（残基集合相同）
    #         duplicate = False
    #         for other in set_list:
    #             if set(other['site_res']) == set(site_dict['site_res']):
    #                 duplicate = True
    #                 break
    #         if duplicate:
    #             continue
    #
    #         set_list.append(site_dict)
    #
    #     return set_list
    import random

    def get_negative_set(self, num, max_depth, max_numres):
        set_list = []
        exist_res_list = []

        # 1. 收集正样本中用到的残基（保持不变）
        for site in self.PDB_info.site_info:
            if site['numRes'] > max_numres:
                continue
            count = num
            res_list = []
            center = site['center_res_ca_coord']
            center_res_depth = site['center_res_ca_depth']
            for res in site['site_res']:
                res_list.append(res)
                count -= 1
            res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
            exist_res_list += res_list

        # 2. 收集所有可作为负样本中心的残基（未使用且为标准氨基酸）
        candidate_centers = []
        for res in self.PDB_info.structure[0].get_residues():
            if res in exist_res_list or not is_aa(res, standard=True):
                continue
            candidate_centers.append(res)
            
        selected_centers = list(candidate_centers)
        # 4. 对每个选中的中心生成负样本
        for res in selected_centers:
            center_res_ca_coord = res.xtra['ca_coord']
            center_res_ca_depth = res.xtra['ca_depth']
            res_list = []
            site_res_list = self._get_close_res(center_res_ca_coord, num, max_depth, center_res_ca_depth, res_list)

            # 检查补充的残基是否与正样本区域重叠
            is_available_flag = True
            for site_res in site_res_list:
                if site_res in exist_res_list:
                    is_available_flag = False
                    break
            if not is_available_flag:
                continue

            # 检查样本大小
            if len(res_list) != num:
                continue

            # 构建负样本字典
            site_dict = {
                'site_res': res_list,
                'center_res_ca_coord': center_res_ca_coord,
                'center_res_ca_depth': center_res_ca_depth,
                'res_info': self.get_site_res_info(res_list, center_res_ca_coord, center_res_ca_depth),
                'isPositive': 0
            }

            # 去重（可选，由于采样随机，重复概率很低）
            duplicate = False
            for other in set_list:
                if set(other['site_res']) == set(site_dict['site_res']):
                    duplicate = True
                    break
            if duplicate:
                continue

            set_list.append(site_dict)

        return set_list
# #生成负样本，中心残基来自指定的特定残基，并且会避开已用于正样本的残基，以确保负样本与正样本不重叠
#     def get_negative_set(self, num, max_depth, max_numres, specific_res):
#         set_list = []
#         exist_res_list = []
#
#         for site in self.PDB_info.site_info:
#             if site['numRes'] > max_numres:
#                 continue
#             count = num
#             res_list = []
#             center = site['center_res_ca_coord']
#             center_res_depth = site['center_res_ca_depth']
#             for res in site['site_res']:
#                 res_list.append(res)
#                 count -= 1
#             res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
#             exist_res_list += res_list
#         for res in self.PDB_info.structure[0].get_residues():
#             if res.get_id()[1] != specific_res:
#                 continue
#             if res in exist_res_list or not is_aa(res, standard=True):
#                 raise ValueError('residue too close to existing site or not surface residue')
#                 continue
#             center_res_ca_coord = res.xtra['ca_coord']
#             center_res_ca_depth = res.xtra['ca_depth']
#             res_list = []
#             site_res_list = self._get_close_res(center_res_ca_coord, num, max_depth, center_res_ca_depth, res_list)
#             is_available_flag = True
#             for site_res in site_res_list:
#                 if site_res in exist_res_list:
#                     is_available_flag = False
#             if is_available_flag:
#                 site_dict = {}
#                 # exist_res_list += site_res_list
#                 site_dict['site_res'] = res_list
#                 site_dict['center_res_ca_coord'] = center_res_ca_coord
#                 site_dict['center_res_ca_depth'] = center_res_ca_depth
#                 site_dict['res_info'] = self.get_site_res_info(res_list, center_res_ca_coord, center_res_ca_depth)
#                 site_dict['isPositive'] = 0
#                 cmp_flag = True
#                 if len(res_list) != 25:
#                     cmp_flag = False
#                 for other_site_dict in set_list:
#                     if len(set(other_site_dict['site_res']).difference(set(site_dict['site_res']))) == 0:
#                         cmp_flag = False
#                 if cmp_flag:
#                     set_list.append(site_dict)
#             break
#         # set_list = random.sample(set_list, int(len(set_list)/25)+1)
#         # print(len(set_list))
#         return set_list
#调用内部方法生成正负样本
    def get_dataset(self, num, max_depth, max_numres, file_path):
        positive_set = self.get_positive_set(num, max_depth, max_numres, file_path)
        negative_set = self.get_negative_set(num, max_depth, max_numres)
        dataset = positive_set + negative_set
        if len(dataset) == 0:
            return 0
        return dataset
#计算两个三维坐标之间的欧氏距离
    def get_distance(self, coord1, coord2):
        diff = coord1 - coord2
        return numpy.sqrt(numpy.dot(diff, diff))



if __name__ == '__main__':
    filepath = '/home/baoyihang/TMPBDSniffer/integration_pipline/3bs0.pdb.pickle'

    site_info = Site_info(filepath)
    sites = site_info.get_dataset(25, 5.5, 10)
    print(sites)
    # for site in sites:
    #     print(site)
    # for site in sites:
    #     for i in range(4):
    #         del site['res_info'][i]['coord_center']
    #         del site['res_info'][i]['ca_coord']
    #         del site['res_info'][i]['res_depth']
    #         del site['res_info'][i]['relative_ca_coord']
    #         del site['res_info'][i]['ca_dis']
    #         print(site['res_info'][i])
    #     break
    # site_info.picklesave()
    # set_list = site_info.negative_set(25, 5.5)
    # print(set_list)
    # site_info.positive_set(25, 5.5, 10)
    # for site in set_list:
    #     for key, value in site.items():
    #         print(key, ' :', value)
    #         # print(value)
    #         if key == 'res_info':
    #             for res in value:
    #                 print(res)
        # print(site)
    # num = site_info.get_site_num()
    # print(num)
    # n, m = site_info.test_cover_rate(4, 15, 4)
    # print(n, m)
    # list = site_info.get_site_res_num()
    # print(list)
    # site_info.get_site_info()
    # site_info.get_site_center_res_coord()
    # site_info.get_site_info()
    # for site in site_info.PDB_info.site_info:
    #     print(site)
