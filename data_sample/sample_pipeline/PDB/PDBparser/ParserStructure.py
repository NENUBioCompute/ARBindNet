import PDB.PDBparser.DataFormat as df
import PDB.Utilities.Files.TxtFileDealer as tf
import PDB.Utilities.Files.FileSysDealer as fs
import PDB.PDBparser.ParserBase as ppd


def func():
    pass


class ParserStructure(object):
    '''
    This parser is an advanced parser of PDBparser file.
    Classify the result of ParserBase into a structural dict
    '''

    # 结构解析器，产生一个包含蛋白质结构层级信息的字典
    def __init__(self):
        self._data_format = dict()

    def parse(self, file):
        '''
        the interface of the class
        :param file:    PDBparser file name
        :return:        a structural dict
        '''
        #创建基础解析器
        pb = ppd.ParserBase(func)
        #执行解析，解析文件中的所有内容
        pdblist = pb.parser(file, target="ALL")
        #调用parseStructure，将pdblist转化为结构化的字典
        result = self.parseStructure(pdblist)
        #将源文件名存入字典，便于追溯
        result["File"] = file
        return result

    def parseStructure(self, list):
        '''
        the interface of the class
        :param list:    the result of ParserBase into a structural dict
        :return:        a structural dict
        '''
        #初始化返回字典
        dict = {}
        #从_data_format获取预定义的数据格式
        self._data_format = df.DataFormat().txt_based_data
        #遍历所有预定义的记录类型，排除掉原子坐标相关记录和模型边界记录，对于允许保留的，在dict中创建空列表
        for key in self._data_format.keys():
            if key not in ["MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ", "ENDMDL", "TER", "END"]:
                dict.update({key: []})
        #外层循环遍历list中的每个字典，内层循环获取该字典中的唯一键mark和值content
        for line in list:
            for mark, content in line.items():
                if mark not in ["MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ", "ENDMDL", "TER", "END"]:
                    if mark in dict.keys():
                        dict[mark].append(content)
        #对特定记录类型进行进一步的结构化处理
        dict["REMARK"] = self.remark_classification(dict["REMARK"])
        dict["SEQRES"] = self.seqres_classification(dict["SEQRES"])
        dict["SITE"] = self.site_classification(dict["SITE"])
        dict["CHAINS"] = self.chain_classification(list)
        # for mark, content in dict.items():
        #     print(mark)
        #     print(content)
        return dict
#解析pdb中的site记录，并将其转换为结构化的位点信息（若一个位点跨越多行，对每个残基字段逐一处理，确保一个位点所有行的残基都收集到同一列表中）
    def site_classification(self, list):
        '''
        classify the 'SITE' information
        :param list: a list of dicts, each dict correspond a line start with 'site' in PDBparser file
        :return: a list of dict dicts include the site info
                 every dict include: 'siteID':      the name of this site
                                     'numRes':      the number of residues in this site
                                     'resName':     the name of the residues in this site
        '''
        site_list = []
        current_sitename = ""
        for content in list:
            if content["siteID"] != current_sitename:
                site = {"siteID": content["siteID"], "numRes": content["numRes"], "resName": []}
                site_list.append(site)
                current_sitename = content["siteID"]
            #解析一行SITE记录中包含的残基信息，并将其添加到当前位点的残基列表中
            for i in range(1, 5):
                resName = "resName"+str(i)
                chainID = "chainID"+str(i)
                seq = "seq"+str(i)
                iCode = "iCode"+str(i)
                #检查当前残基位置是否有实际内容，如果非空就创建字典site_res存储该残基的完整描述信息
                if content[resName] != "":
                    site_res = {"chainID": content[chainID], "seq": content[seq], "resName": content[resName],
                                "iCode": content[iCode]}
                    for site in site_list:
                        if site["siteID"] == current_sitename:
                            site["resName"].append(site_res)
        return site_list
#将pdb中的SEQRES记录按链分组，并合并每条链的残疾名称序列（SEQRES中记录：链标识符，该链中的残基总数，残基名称列表）
    def seqres_classification(self, list):
        '''
        classify the 'SEQRES' information
        :param list: a list of dicts, each dict correspond a line start with 'seqres' in PDBparser file
        :return: a list of dict dicts include the seqres info
                 every dict include: 'chainID':     the name of this chain
                                     'numRes':      the number of residues in this chain
                                     'resName':     the name of the residues in this chain
        '''
        chain_list = []
        current_chainid = ""
        #遍历每个content(一行SEQRES记录)
        for content in list:
            #如果chainID更新，则创建一个新字典，包含chain ID，numRES和空的resName列表
            if content["chainID"] != current_chainid:
                chain = {"chainID": content["chainID"], "numRes": content["numRes"], "resName": []}
                #将新字典添加到chain_list
                chain_list.append(chain)
                #更新current_chainid
                current_chainid = content["chainID"]
            #将该行中的残基名称列表追加到该链的resName列表末尾
            for chain in chain_list:
                if chain["chainID"] == current_chainid:
                    chain["resName"] += content["resName"]
        return chain_list

#将remark记录按编号分组
    def remark_classification(self, list):
        '''
        classify the 'REMARK' information
        :param list: a list of dicts, each dict correspond a line start with 'remark' in PDBparser file
        :return:     a list of dict dicts include the remark info
                     every dict include: 'remarkNum':   the mark number of this remark
                                         'empty':       a list of str, every str correspond one line start with 'remark'
        '''
        #存储分组后的结果
        remark_list = []
        #记录当前正在处理的组号
        current_remarkNum = 0
        #遍历每一行remark记录
        for content in list:
            #遇到新的组号，创建新组
            if content["remarkNum"] != current_remarkNum:
                remark = {"remarkNum": content["remarkNum"], "empty": []}
                #将创建的新字典添加到remark_list,并更新current_remarkNum
                remark_list.append(remark)
                current_remarkNum = content["remarkNum"]
            for remark in remark_list:
                if remark["remarkNum"] == current_remarkNum:
                    remark["empty"].append(content["empty"])
        return remark_list

#处理pdb文件中与原子坐标相关的记录，并按chain和model进行组织
    def chain_classification(self, list):
        '''
        classify the atomic related information. Atomic related include "MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ",
                                                                        "ENDMDL", "TER".
        :param list: a list of dicts, each dict correspond a atomic related line in PDBparser file
        :return:     a list of dict dicts include the atomic info
                     every dict include: 'chainID':         the name of this chain
                                         'chainStructure':  a list of models
        '''
        #扫描输入的list,收集所有出现的链ID
        chains = self.init_chains(list)
        for line in list:
            for mark, content in line.items():
                #遇到MODEL和ENDMDL，将这一整条记录添加到所有链的chainstructure中
                if mark == "MODEL" or mark == "ENDMDL":
                    for chain in chains:
                        chain["chainStructure"].append(line)
                #对于这四种，提取其chainID字段，将记录添加到对应链的chainstructure中
                if mark == "ATOM" or mark == "HETATM" or mark == "ANISOU" or mark == "SIGUIJ":
                    for chain in chains:
                        if content["chainID"] == chain["chainID"]:
                            chain["chainStructure"].append(line)
        chains = self.chain_to_model(chains)
        return chains
#扫描原子记录列表，提取所有出现的chainID,并为每条链初始化一个包含chainStructure列表的字典
    def init_chains(self, linelist):
        '''
        iterate the list to get all the chainIDs
        :param linelist: list of atomic related info
        :return: a list of dict of chains, 'chainStructure' is an empty list to be complete with models
        '''
        #用于存储去重后的链ID
        idlist = []
        #用于存储最终的结果字典列表
        resultlist = []
        #遍历输入列表中的每一行记录
        for line in linelist:
            for mark, content in line.items():
                #只关注ATOM和HETATM
                if mark == "ATOM" or mark == "HETATM":
                    #从content中提取chainID字段，如果该链ID尚未被记录，则将其加入idlist
                    if content["chainID"] not in idlist:
                        idlist.append(content["chainID"])
        #遍历收集到的chainID列表，为每个链ID创建一个字典
        for chainid in idlist:
            chain ={}
            chain["chainID"] = chainid
            chain["chainStructure"] = []
            resultlist.append(chain)
        #返回包含链信息的列表
        return resultlist

#将每条链的扁平原子记录列表转换为按模型和残基组织的层级结构
    def chain_to_model(self, chains):
        '''
        :param list: a list of dicts, each dict is a chain
        :return:     Complete 'chainStructure' with models
                     every dict include: 'serial':          the mark number of this remark
                                         'modelStructure':  a list of residues
        '''
        for chain in chains:
            #标志当前是否处于模型内部（0=未进入模型，1=已进入模型）
            model_open = 0
            #当前正在处理的模型编号
            current_model_id = 0
            #用于存储该链所有模型的列表
            model_list = []
            for line in chain["chainStructure"]:
                for mark, content in line.items():
                    #当遇到MODEL时，将model_open置为1，记录模型编号，并创建一个新模型字典添加到model_list
                    if mark == "MODEL":
                        model_open = 1
                        current_model_id = content["serial"]
                        model = {"serial": current_model_id, "modelStructure": [] }
                        model_list.append(model)
                    #如果遇到原子记录但model_open == 0，自动创建一个模型，并打开模型标志。
                    if mark == "ATOM" or mark == "HET，ATM" or mark == "ANISOU" or mark == "SIGUIJ":
                        if model_open == 0:
                            current_model_id = 1
                            model_open = 1
                            model = {"serial": current_model_id, "modelStructure": []}
                            model_list.append(model)
                        #遍历 model_list，找到与 current_model_id 匹配的模型，将当前记录追加到该模型的modelStructure列表中
                        for model in model_list:
                            if model["serial"] == current_model_id:
                                model["modelStructure"].append(line)
            model_list = self.model_to_residue(model_list)
            chain["chainStructure"] = {"models": model_list}
        return chains

#将每个记录中的原子纪录列表按残基进行分组，形成模型-残基-原子记录的层级结构
    def model_to_residue(self, models):
        '''
        :param list: a list of dicts, each dict is a model
        :return:     Complete 'modelStructure' with residues
                     every dict include: 'resName':         the name of the residue
                                         'resSeq':          the sequence number of this residude
                                         'resStructure':    a list of 'ATOM' 'HETATM' 'ANISOU' 'SIGUIJ'
        '''
        for model in models:
            #记录当前正在处理的残基序号，用于检测是否进入新残基
            current_res_id = 0
            #存储该模型下的所有残基字典
            res_list = []
            #遍历模型中的所有原子记录
            for atoms in model["modelStructure"]:
                for mark, content in atoms.items():
                    #如果当前原子的残基序号与current_res_id不同，说明开始了新的残基，创建一个新的残基字典，添加到 res_list，更新 current_res_id。
                    if content["resSeq"] != current_res_id:
                        res = {"resName": content["resName"], "resSeq": content["resSeq"], "resStructure": []}
                        res_list.append(res)
                        current_res_id = content["resSeq"]
                    #遍历已创建的残基列表，找到与 current_res_id 匹配的残基，将当前原子记录追加到该残基的 resStructure 列表中
                    for res in res_list:
                        if res["resSeq"] == current_res_id:
                            res["resStructure"].append(atoms)
            #将模型的 modelStructure 从原始的原子记录列表替换为按残基组织的列表。
            model["modelStructure"] = res_list
        return models


if __name__ == '__main__':
    file = r'H:\biodata\pdbtm\pdb_all\1a0s.pdb'
    ps = ParserStructure()
    result = ps.parse(file)
    for key, content in result.items():
        print(key)
        print(content)

