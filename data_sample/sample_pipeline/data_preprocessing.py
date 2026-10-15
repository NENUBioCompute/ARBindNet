import os
from site_info import Site_info
from PDB_info import PDBInfo
from Bio.PDB.PDBParser import PDBParser
import warnings
import pickle
import random
import copy
from Bio.PDB import PDBIO
import argparse
import sys
sys.path.append('../')
from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein, PyProteinBatch
import csv


class data_preprocessing:
#self指向实例本身的引用，w_dir为自定义参数
    def __init__(self, file_path, w_dir='/datapool/home/info_wang/tongyanting/GCL/ARBindNet/data_sample/sample_output/'):
        #将传入的文件属性保存为实例属性，供其他方法使用
        self.Site_info = None
        self.file_path = file_path
        #调用一个方法，检查指定文件路径是否存在
        self.check_file_exist()
        #调用另一个方法，验证文件格式是否符合预期
        self.check_file_format()
        #将输出目录保存为实例属性
        self.w_dir = w_dir

#定义一个实例方法，通过self访问实例属性
    def check_file_format(self):
        if not self.file_path.endswith('.pdb'):
            raise ValueError('File format should be pdb!')
        return True

    def check_file_exist(self):
        if not os.path.exists(self.file_path):
            raise ValueError('File does not exist!')
        return True

#创建一个PDBInfo的类的实例，并给他传入两个参数
#指定使用 msms 程序来计算分子表面等相关属性
    def PDBInfo_Process(self):
        res_info = PDBInfo(self.file_path, MSMS='msms')
        return res_info
#用于生成活性位点相关的数据集
    def SiteInfo_Process(self):
        #创建一个site_info类的实例，传入两个参数：pdbinfo对象和原始pdb文件路径
        self.Site_info = Site_info(self.PDBInfo_Process(), self.file_path)
        #调用 site_info 对象的 get_dataset 方法，并传入四个数值参数
        #max_numres表示位点本身的最大残基数
        dataset = self.Site_info.get_dataset(25, 5.5, 25, self.file_path)
        return dataset

#将数据集中的样本转换为便于查看或存储的简化格式，并分别提取正负样本的残基信息列表
    def form_residue_list(self, residue_list):
        positive_domain = []
        negative_domain = []
        #遍历每个样本，对样本中的每个残基对象，提取残基名，链的标识符，残基序号，存入临时列表tmp_dic
        for domain in residue_list:
            tmp_dic = []
            for residue in domain['site_res']:
                tmp_dic.append((residue.resname, residue.parent.id, str(residue.id[1])))
            #根据样本的标签，将tmp_dic的深拷贝分别添加到 positive_domain 或 negative_domain 中
            if domain['isPositive'] == 1:
                positive_domain.append(copy.deepcopy(tmp_dic))
            else:
                negative_domain.append(copy.deepcopy(tmp_dic))
        #对负样本列表进行随机打乱
        random.shuffle(negative_domain)
        return positive_domain, negative_domain

        # negative_domain = negative_domain[:len(positive_domain)]
        # output_file = open(
        #     '/home/baoyihang/TMPBDSniffer/test_data_prepare/positive_domain_list/' + file_name.split('.')[
        #         0] + '.pickle', 'wb')
        # pickle.dump(positive_domain, output_file)
        # output_file.close()
        # output_file = open(
        #     '/home/baoyihang/TMPBDSniffer/test_data_prepare/negative_domain_list/' + file_name.split('.')[
        #         0] + '.pickle', 'wb')
        # pickle.dump(negative_domain, output_file)
        # output_file.close()
#使用 Biopython 解析 PDB 文件，并返回第一个模型的结构对象（大多数X射线晶体学结构只有一个模型，而NMR结构可能有多个，这里只取第一个模型）
    def form_structure(self):
        parser = PDBParser()
        structure = parser.get_structure('1', self.file_path)
        return structure[0]
#从原始 PDB 结构中提取出指定的残基集合，生成新的 PDB 文件
    def from_domain_to_pdb(self, domain_list, tag='positive', csv_path=None):
        out_file_no = 0
        self.tag = tag
        for idx, domain in enumerate(domain_list):
            #调用form_structure，获取原始pdb结构的深拷贝
            structure = self.form_structure()
            #遍历结构中的每条链
            for chain in structure:
                del_list = []
                #遍历链中的每个残基
                for residue in chain:
                    #构建一个元组，如果该元组不在当前 domain 的元组列表中，则将该残基的 ID 加入待删除列表del_list
                    if (residue.resname, residue.parent.id, str(residue.id[1])) not in domain:
                        del_list.append(residue.id)
                #删除链中所有未被选中的残基
                for i in del_list:
                    chain.detach_child(i)
            del_list = []
            #删除空链，如果某条链在删除残基后为空，则将其 ID 加入待删除列表，然后从结构中移除该链
            for chain in structure:
                if len(chain) == 0:
                    del_list.append(chain.id)
            for i in del_list:
                structure.detach_child(i)
            #保存pdb文件在w_dir + 'domain_pdb/'
            io = PDBIO()
            io.set_structure(structure)
            out_file_no += 1
            pdb_save_path = self.w_dir + 'domain_pdb/' + self.file_path.split('.')[-2].split('/')[-1] + '_' + str(out_file_no) + '_' + tag + '.pdb'
            if not os.path.exists(self.w_dir + 'domain_pdb/'):
                os.mkdir(self.w_dir + 'domain_pdb/')
            io.save(pdb_save_path)
            # 将文件名记录回 dataset 对应的字典
            if isinstance(domain, dict):
                domain['domain_pdb_name'] = os.path.basename(pdb_save_path)
                # domain_pdb_name = os.path.basename(pdb_save_path)
                # domain['domain_pdb_name'] = domain_pdb_name
                #
                # # --- 新增：如果在生成正样本并传入了 csv_path，则直接追加写入 CSV ---
                # if csv_path:
                #     site_id = domain.get('siteID', '')
                #     ligand = domain.get('binding_ligand', '')
                #
                #     with open(csv_path, 'a', newline='') as f:
                #         writer = csv.DictWriter(f, fieldnames=['pdb_file', 'site_id', 'binding_ligand'])
                #         writer.writerow({
                #             'pdb_file': domain_pdb_name,
                #             'site_id': site_id,
                #             'binding_ligand': ligand
                #         })
#将之前生成的 PDB 文件（正负样本区域）转换为 HDF5 格式，并保存为结构化数据文件，同时记录生成的文件列表
    def generate_hdf5(self):
        file_list = []
        #使用os.walk遍历目录下的所有文件
        for root, dirs, files in os.walk(self.w_dir + 'domain_pdb/'):
            for name in files:
                #只保留文件名前四个字符与原始pdb文件名相同的文件
                if name[:4] == self.file_path.split('.')[-2].split('/')[-1]:
                    #将符合条件的文件名存入file_list
                    file_list.append(name)
        co = 0
        #记录正负样本的 HDF5 文件名，每行一个，最后写入文本文件
        potitive_name_list = ''
        negative_name_list = ''
        domain_name_list =''
        #对每个pdb文件进行处理
        for file_name in file_list:
            #创建一个周期表对象
            periodicTable_ = PyPeriodicTable()
            #创建一个蛋白质对象，传入周期表对象
            curProtein = PyProtein(periodicTable_)
            #加载pdb文件，解析原子坐标信息
            curProtein.load_molecular_file(self.w_dir + 'domain_pdb/' + file_name)
            #计算共价键
            curProtein.compute_covalent_bonds()
            #计算氢键
            curProtein.compute_hydrogen_bonds()
            #如果计算出的氨基酸邻居列表、氢键邻居列表或氨基酸类型数量过少（少于 10 种），则跳过该文件
            if len(curProtein.aminoNeighs_) == 0 or len(curProtein.aminoNeighsHB_) == 0 or len(
                    curProtein.aminoType_) < 10:
                continue
            #确保 HDF5 输出目录存在，若不存在则创建。
            if not os.path.exists(self.w_dir + 'domain_hdf5/'):
                os.mkdir(self.w_dir + 'domain_hdf5/')
            #将当前蛋白质对象保存为HDF5文件，文件名与原始PDB文件名相同（但扩展名改为 .hdf5）
            curProtein. save_hdf5(self.w_dir + 'domain_hdf5/' + file_name.split('.')[0] + '.hdf5')
            if file_name.split('.')[0].split('_')[-1] == 'positive':
                potitive_name_list += file_name.split('.')[0] + '\n'
            else:
                negative_name_list += file_name.split('.')[0] + '\n'
        #将正样本文件名列表、负样本文件名列表以及合并后的列表分别写入三个文本文件中，保存在输出目录下。
        
        txt_dir = os.path.join(self.w_dir, 'txt_list')
        os.makedirs(txt_dir, exist_ok=True)

        base_name = self.file_path.split('.')[-2].split('/')[-1]

        with open(os.path.join(txt_dir, base_name + '_positive_domain_list.txt'), 'w') as f:
            f.write(potitive_name_list)

        with open(os.path.join(txt_dir, base_name + '_negative_domain_list.txt'), 'w') as f:
            f.write(negative_name_list)

        with open(os.path.join(txt_dir, base_name + '_domain_list.txt'), 'w') as f:
            f.write(domain_name_list)

    def start_processing(self):
        preprocessed_data = self.SiteInfo_Process()
        # # ========================================================
        print('Preprocessing finished!')
        positive_domain, negative_domain = self.form_residue_list(preprocessed_data)
        print('Domain form finished!')
        # # --- 新增：提前初始化 CSV 文件并写入表头 ---
        # csv_path = self.w_dir + 'binding_ligands.csv'
        # with open(csv_path, 'w', newline='') as f:
        #     writer = csv.DictWriter(f, fieldnames=['pdb_file', 'site_id', 'binding_ligand'])
        #     writer.writeheader()
        if positive_domain.__len__() > 0:
            self.from_domain_to_pdb(positive_domain, 'positive')
        if negative_domain.__len__() > 0:
            self.from_domain_to_pdb(negative_domain, 'negative')
        print('Domain PDB file form finished!')
        csv_path = self.w_dir + 'binding_ligands.csv'
        self.Site_info.save_binding_ligands_csv(preprocessed_data, csv_path)
        print('Binding ligand CSV saved!')
        self.generate_hdf5()
        print('HDF5 file form finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='transform to PDB')
    parser.add_argument('--pdbpath', required=True, help='Path to a PDB file or a folder containing PDB files')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')

    # 新增：失败日志文件
    fail_log_path = './wrong.txt'

    # 👉 如果是文件夹（批处理）
    if os.path.isdir(args.pdbpath):
        pdb_files = [f for f in os.listdir(args.pdbpath) if f.endswith('.pdb')]

        print(f"Total {len(pdb_files)} pdb files")

        for pdb_file in pdb_files:
            pdb_path = os.path.join(args.pdbpath, pdb_file)
            print(f"\nProcessing {pdb_file}")

            try:
                a = data_preprocessing(pdb_path)
                a.start_processing()

            except Exception as e:
                print(f"[FAILED] {pdb_file}")
                print(f"Reason: {e}")

                # ✅ 关键：写入失败日志（追加模式）
                with open(fail_log_path, 'a') as f:
                    f.write(f"{pdb_file} | {str(e)}\n")

                continue

        print("\nBatch processing finished!")

    # 👉 如果是单文件
    else:
        try:
            a = data_preprocessing(args.pdbpath)
            a.start_processing()
        except Exception as e:
            print(f"[FAILED] {args.pdbpath}")

            with open(fail_log_path, 'a') as f:
                f.write(f"{args.pdbpath} | {str(e)}\n")

# if __name__ == '__main__':
#     #argparse用于解析命令行参数，创建一个命令行参数解析器对象
#     parser = argparse.ArgumentParser(description='transform to PDB')
#     ##允许用户通过命令行指定PDB路径
#     parser.add_argument('--pdbpath')
#     #解析实际传入的命令行
#     args = parser.parse_args()
#     warnings.filterwarnings('ignore')
#     # #实例化一个名为 data_preprocessing 的类，并传入pdbpath 参数
#     # a = data_preprocessing(args.pdbpath)
#     # #调用实例a的 start_processing方法，开始数据处理
#     # a.start_processing()
#     if os.path.isdir(args.pdbpath):
#         pdb_files = [f for f in os.listdir(args.pdbpath) if f.endswith('.pdb')]
#
#         for pdb_file in pdb_files:
#             pdb_path = os.path.join(args.pdbpath, pdb_file)
#             print(f"Processing {pdb_file}")
#
#             try:
#                 a = data_preprocessing(pdb_path)
#                 a.start_processing()
#             except Exception as e:
#                 print(f"Error: {e}")
#     else:
#         a = data_preprocessing(args.pdbpath)
#         a.start_processing()