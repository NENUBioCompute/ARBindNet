#将用CD-HIT工具得到的cluster进行连通分量聚类
import re
from collections import defaultdict

def build_supergroups_from_clstr(clstr_file):
    """
    从 CD-HIT .clstr 构建 protein-level SuperGroups
    """

    # ===== 1. 解析 cluster =====
    clusters = []
    current_cluster = []

    with open(clstr_file, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
            else:
                # 提取 chain id，例如 1abc_A
                match = re.search(r'>([^\.]+)', line)
                if match:
                    chain_id = match.group(1)
                    pdb_id = chain_id[:4].lower()  # 只取 protein id
                    current_cluster.append(pdb_id)

        if current_cluster:
            clusters.append(current_cluster)

    # ===== 2. 构建 protein graph =====
    graph = defaultdict(set)

    for cluster in clusters:
        proteins = set(cluster)
        for p1 in proteins:
            _ = graph[p1]
            for p2 in proteins:
                if p1 != p2:
                    graph[p1].add(p2)
                    graph[p2].add(p1)

    # ===== 3. 找连通分量（DFS）=====
    visited = set()
    supergroups = []

    def dfs(start):
        stack = [start]
        component = []

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            component.append(node)

            for nei in graph[node]:
                if nei not in visited:
                    stack.append(nei)

        return component

    for node in graph:
        if node not in visited:
            comp = dfs(node)
            supergroups.append(sorted(comp))

    return supergroups

def save_supergroups(supergroups, output_file):
    with open(output_file, "w") as f:
        f.write(f":Summary: 共发现 {len(supergroups)} 个彼此独立的 SuperGroups\n")
        f.write("="*50 + "\n\n")

        for i, group in enumerate(supergroups):
            f.write(f"SuperGroup_{i}:\n")
            f.write(", ".join(group) + "\n")
            f.write("-"*30 + "\n")

clstr_file = "./clustered_0.5.fasta.clstr"
output_file = "./supergroups50.txt"

supergroups = build_supergroups_from_clstr(clstr_file)
save_supergroups(supergroups, output_file)

print(f"Done! 共 {len(supergroups)} 个 SuperGroups")



# #提取10%的蛋白质id作为独立测试集，并将提取后的SuperGroups保存下来
# #!/usr/bin/env python3
# """
# 按蛋白质比例（目标10%）从 SuperGroups 中抽取测试集，保持簇完整。
# 严格要求测试蛋白数 >= 目标值，且尽量不超过 (目标值 + epsilon)。
# 输出测试集和训练集的簇文件及蛋白质列表。
# """
#
# import random
# import re
# import os
#
# # ======================
# # 配置
# # ======================
# INPUT_FILE = "/datapool/home/info_wang/tongyanting/GCL/model/0510/cdhit/supergroups50.txt"      # 输入的 SuperGroups 文件
# OUTPUT_DIR = "/datapool/home/info_wang/tongyanting/GCL/model/0510"                             # 输出目录
# TEST_FRAC = 0.1                              # 目标测试蛋白占总蛋白的比例
# RANDOM_SEED = 0                             # 随机种子，可复现
# EPSILON_FRAC = 0.005                        # 允许超过目标的最大比例（总蛋白数 * 0.005）
# MAX_ATTEMPTS = 200                           # 随机贪心最大尝试次数
#
# # 输出文件名
# TEST_CLUSTER_FILE = os.path.join(OUTPUT_DIR, "test_supergroups.txt")
# TEST_LIST_FILE = os.path.join(OUTPUT_DIR, "test_ids.txt")
# TRAIN_CLUSTER_FILE = os.path.join(OUTPUT_DIR, "train_supergroups.txt")
# TRAIN_LIST_FILE = os.path.join(OUTPUT_DIR, "train_ids.txt")
#
# # ======================
# # 解析 SuperGroups 文件
# # ======================
# def parse_supergroups(filepath):
#     """返回 [(cluster_name, [protein_id, ...]), ...] 列表"""
#     clusters = []
#     current_name = None
#     current_prots = []
#     header_passed = False
#
#     with open(filepath) as f:
#         for line in f:
#             line = line.strip()
#             # 跳过开头的 Summary 和分隔线
#             if not header_passed:
#                 if line.startswith(":Summary:") or line.startswith("=="):
#                     header_passed = True
#                 continue
#
#             if line.startswith("SuperGroup_"):
#                 # 保存上一个簇
#                 if current_name is not None and current_prots:
#                     clusters.append((current_name, current_prots))
#                 current_name = line.rstrip(":")
#                 current_prots = []
#             elif line.startswith("---"):
#                 if current_name is not None:
#                     clusters.append((current_name, current_prots))
#                 current_name = None
#                 current_prots = []
#             elif current_name is not None:
#                 # 提取类似 6uqk, 1aig 的标识符
#                 ids = re.findall(r'[A-Za-z0-9]+', line)
#                 current_prots.extend(ids)
#
#     # 文件末尾可能没有分隔符
#     if current_name is not None and current_prots:
#         clusters.append((current_name, current_prots))
#
#     return clusters
#
# # ======================
# # 簇选择：确保蛋白数 >= target，尽量不超过 target+epsilon
# # ======================
# def select_clusters_by_count(clusters, target, epsilon, seed):
#     """
#     通过多次随机贪心选择，返回满足 target <= actual_count <= target+epsilon 的簇索引集合。
#     如果多次尝试失败，返回最接近且满足 >= target 的组合。
#     """
#     random.seed(seed)
#     best_selected = None
#     best_count = -1
#
#     for attempt in range(MAX_ATTEMPTS):
#         indices = list(range(len(clusters)))
#         random.shuffle(indices)
#
#         selected = set()
#         current = 0
#
#         # 第一阶段：只要不超过上限就加
#         for idx in indices:
#             size = len(clusters[idx][1])
#             if current + size <= target + epsilon:
#                 selected.add(idx)
#                 current += size
#             if current >= target:
#                 break
#
#         # 如果还没达到 target，从未选中簇中补充（仍不能超上限）
#         if current < target:
#             remaining = [i for i in indices if i not in selected]
#             for idx in remaining:
#                 size = len(clusters[idx][1])
#                 if current + size <= target + epsilon:
#                     selected.add(idx)
#                     current += size
#                 if current >= target:
#                     break
#
#         if target <= current <= target + epsilon:
#             return selected, current
#
#         # 记录最优（优先 >= target 且接近 target）
#         if current >= target:
#             if best_selected is None or current < best_count:
#                 best_selected = selected.copy()
#                 best_count = current
#
#     if best_selected is not None:
#         print(f"警告：经 {MAX_ATTEMPTS} 次尝试未完全满足上限，"
#               f"使用蛋白数 {best_count} 的组合（超出上限 {best_count - target - epsilon} 个）")
#         return best_selected, best_count
#     else:
#         raise RuntimeError(f"无法选出足够蛋白数的测试集，总蛋白：{sum(len(c[1]) for c in clusters)}，目标：{target}")
#
# # ======================
# # 输出 SuperGroup 格式簇文件
# # ======================
# def write_supergroups(filepath, clusters, description=""):
#     with open(filepath, "w") as f:
#         f.write(f":Summary: 共发现 {len(clusters)} 个彼此独立的 SuperGroups\n")
#         f.write("==================================================\n\n")
#         for idx, (name_orig, proteins) in enumerate(clusters):
#             f.write(f"SuperGroup_{idx}:\n")
#             f.write(", ".join(proteins) + "\n")
#             f.write("------------------------------\n")
#         if description:
#             f.write(f"# {description}\n")
#     total_prots = sum(len(p) for _, p in clusters)
#     print(f"已写入 {filepath}（{len(clusters)} 个簇，{total_prots} 个蛋白质）")
#
# # ======================
# # 输出纯蛋白质列表（每行一个）
# # ======================
# def write_protein_list(filepath, clusters, description=""):
#     proteins = set()
#     for _, prots in clusters:
#         proteins.update(prots)
#     with open(filepath, "w") as f:
#         for p in sorted(proteins):
#             f.write(p + "\n")
#         if description:
#             f.write(f"# {description}\n")
#     print(f"已写入蛋白列表 {filepath}（{len(proteins)} 个蛋白）")
#
# # ======================
# # 主程序
# # ======================
# if __name__ == "__main__":
#     print("读取 SuperGroups 文件...")
#     clusters = parse_supergroups(INPUT_FILE)
#     total_clusters = len(clusters)
#     total_proteins = sum(len(prots) for _, prots in clusters)
#     print(f"总簇数: {total_clusters}, 总蛋白数: {total_proteins}")
#
#     target = int(total_proteins * TEST_FRAC)
#     epsilon = int(total_proteins * EPSILON_FRAC)
#     print(f"目标测试蛋白数: {target}，允许区间: [{target}, {target+epsilon}]")
#
#     # 选择测试簇
#     selected_idx, actual_count = select_clusters_by_count(clusters, target, epsilon, RANDOM_SEED)
#     print(f"实际选中测试簇数: {len(selected_idx)}, 包含蛋白数: {actual_count} "
#           f"({actual_count/total_proteins:.2%})")
#
#     # 构造测试集和训练集
#     test_clusters = [clusters[i] for i in range(total_clusters) if i in selected_idx]
#     train_clusters = [clusters[i] for i in range(total_clusters) if i not in selected_idx]
#
#     # 输出测试集相关文件
#     write_supergroups(TEST_CLUSTER_FILE, test_clusters,
#                       description="测试集簇（≥10%蛋白数，保持簇完整）")
#     write_protein_list(TEST_LIST_FILE, test_clusters,
#                        description="测试集蛋白质列表")
#
#     # 输出训练集相关文件
#     write_supergroups(TRAIN_CLUSTER_FILE, train_clusters,
#                       description="训练集簇（去除测试集后）")
#     write_protein_list(TRAIN_LIST_FILE, train_clusters,
#                        description="训练集蛋白质列表")
#
#     print("\n全部输出完毕。")







# #根据supergroups聚成的类按照7：2：1的比例划分训练集验证集和测试集
# import random
# import re
#
#
# def split_super_groups_v3(super_group_file, train_ratio=0.8, val_ratio=0.2, test_ratio=0.1, tolerance=0.001):
#     """
#     按蛋白 ID 总数比例 (7:2:1) 划分 SuperGroup，
#     当组过大导致集合数量超出 目标+容忍值 时，自动分配给训练集。
#     """
#     groups = []
#     all_pdbs_list = []
#
#     # 1. 解析 SuperGroup 文件
#     with open(super_group_file, 'r', encoding='utf-8') as f:
#         content = f.read()
#         blocks = re.findall(r'SuperGroup_\d+:\n(.*?)\n', content)
#         for b in blocks:
#             pdbs = [p.strip() for p in b.split(',') if p.strip()]
#             if pdbs:
#                 groups.append(pdbs)
#                 all_pdbs_list.extend(pdbs)
#
#     total_pdb_count = len(all_pdbs_list)
#
#     # 2. 随机打乱组
#     random.seed(42)
#     random.shuffle(groups)
#
#     # 3. 计算各集合目标数量和容忍上限
#     # target_test = int(total_pdb_count * test_ratio)
#     target_val = int(total_pdb_count * val_ratio)
#     # max_test = target_test + int(total_pdb_count * tolerance)
#     max_val = target_val + int(total_pdb_count * tolerance)
#     # 训练集不设上限
#
#     train_pdbs1, val_pdbs1 = [], []
#     train_groups_count = val_groups_count = test_groups_count = 0
#
#     # 4. 分配
#     for group in groups:
#         # 优先测试集：当前数量+该组大小 ≤ 上限时放入
#         # if len(test_pdbs1) + len(group) <= max_test:
#         #     test_pdbs1.extend(group)
#         #     test_groups_count += 1
#         # 其次验证集：同样检查上限
#         if len(val_pdbs1) + len(group) <= max_val:
#             val_pdbs1.extend(group)
#             val_groups_count += 1
#         # 否则放进训练集
#         else:
#             train_pdbs1.extend(group)
#             train_groups_count += 1
#
#     # 5. 统计报告
#     print(f"--- 蛋白质 ID 统计报告 ---")
#     print(f"蛋白质 ID 总数: {total_pdb_count}")
#     print(f"独立 SuperGroups 总数: {len(groups)}")
#     print(f"容忍上限 (绝对值):  验证集 {max_val}")
#     print("-" * 40)
#
#     def print_stat(name, pdbs, groups_cnt, target_count):
#         actual_ratio = len(pdbs) / total_pdb_count
#         print(f"{name}: {len(pdbs)} 个 ID ({actual_ratio:.2%}), 分属 {groups_cnt} 个组 (目标 {target_count})")
#
#     print_stat("训练集 (Train)", train_pdbs1, train_groups_count, int(total_pdb_count * train_ratio))
#     print_stat("验证集 (Val)  ", val_pdbs1, val_groups_count, target_val)
#     # print_stat("测试集 (Test) ", test_pdbs1, test_groups_count, target_test)
#
#     # 6. 保存
#     for name, lst in [('train', train_pdbs1), ('val', val_pdbs1)]:
#         with open(f'{name}_pdbs.txt', 'w') as f:
#             f.write("\n".join(lst))
#
#     print(f"\n划分完成！实际比例 -> 训练:{len(train_pdbs1)/total_pdb_count:.2f}  "
#           f"验证:{len(val_pdbs1)/total_pdb_count:.2f}  ")
#           # f"测试:{len(test_pdbs1)/total_pdb_count:.2f}")
#
# # 执行
# split_super_groups_v3('/datapool/home/info_wang/tongyanting/GCL/model/0510/cdhit/recognize_split/pretrain_supergroups.txt')
#






# ##基于tmscore进行连通分量聚类
# # !/usr/bin/env python3
# """
# 基于 USalign 生成的 tmscore.txt 进行连通分量聚类
# 输出类似 SuperGroup 格式的簇，并统计唯一蛋白质 ID 数量
# """
#
# import os
# from collections import defaultdict
#
# # ======================
# # 配置参数
# # ======================
# RAW_FILE = "/datapool/home/info_wang/tongyanting/GCL/beta_data/tmscore.txt"  # TM-score 结果文件
# OUTPUT_FILE = "/datapool/home/info_wang/tongyanting/GCL/model/0510/tmscore_supergroups0.8.txt"        # 输出簇文件
# TM_THRESHOLD = 0.8                             # TM-score 阈值（可根据需要调整）
# USE_AVG = False                               # True:使用平均 TM-score; False:使用两者中的最大值
#
# # ======================
# # 工具函数
# # ======================
# def normalize(name):
#     """标准化蛋白 ID：去除路径和 .pdb 后缀"""
#     name = name.split("/")[-1]      # 去掉可能的前缀路径
#     name = name.replace(".pdb", "") # 去掉 .pdb 后缀
#     return name
#
# # ======================
# # 并查集 (Union-Find)
# # ======================
# class UnionFind:
#     def __init__(self):
#         self.parent = {}
#
#     def add(self, x):
#         if x not in self.parent:
#             self.parent[x] = x
#
#     def find(self, x):
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])  # 路径压缩
#         return self.parent[x]
#
#     def union(self, x, y):
#         rx = self.find(x)
#         ry = self.find(y)
#         if rx != ry:
#             self.parent[rx] = ry
#
# # ======================
# # 解析 tmscore.txt
# # ======================
# def parse_tmscore_file(filepath):
#     """
#     读取 USalign -outfmt 2 的输出文件，返回 (protein1, protein2, tm_score) 列表
#     会自动跳过注释行（以 # 开头）和格式不正确的行
#     """
#     edges = []
#     with open(filepath) as f:
#         for line_num, line in enumerate(f, 1):
#             line = line.strip()
#             if not line or line.startswith("#"):
#                 continue
#             parts = line.split()
#             if len(parts) < 4:
#                 continue
#             a_raw, b_raw = parts[0], parts[1]
#             try:
#                 tm1 = float(parts[2])
#                 tm2 = float(parts[3])
#             except ValueError:
#                 print(f"警告：第 {line_num} 行 TM-score 解析失败，已跳过 -> {line}")
#                 continue
#
#             a = normalize(a_raw)
#             b = normalize(b_raw)
#
#             if USE_AVG:
#                 score = (tm1 + tm2) / 2.0
#             else:
#                 score = max(tm1, tm2)
#
#             edges.append((a, b, score))
#     return edges
#
# # ======================
# # 连通分量聚类
# # ======================
# def clustering_by_tmscore(edges, threshold=0.5):
#     """使用并查集找出所有连通分量（单链接聚类）"""
#     uf = UnionFind()
#     # 收集所有出现过的蛋白 ID
#     nodes = set()
#     for a, b, _ in edges:
#         nodes.add(a)
#         nodes.add(b)
#     for node in nodes:
#         uf.add(node)
#
#     # 连接满足阈值的蛋白对
#     for a, b, score in edges:
#         if score >= threshold:
#             uf.union(a, b)
#
#     # 收集每个簇
#     clusters = defaultdict(list)
#     for node in nodes:
#         root = uf.find(node)
#         clusters[root].append(node)
#
#     return list(clusters.values())
#
# # ======================
# # 输出簇文件（SuperGroup 风格）
# # ======================
# def output_clusters(clusters, output_file):
#     """将聚类结果写入文件，格式类似 SuperGroup_X: 蛋白1, 蛋白2, ..."""
#     with open(output_file, "w") as f:
#         f.write(f":Summary: 共发现 {len(clusters)} 个彼此独立的 SuperGroups\n")
#         f.write("==================================================\n\n")
#         for idx, cluster in enumerate(clusters):
#             f.write(f"SuperGroup_{idx}:\n")
#             # 按字母排序，美观展示
#             sorted_cluster = sorted(cluster)
#             f.write(", ".join(sorted_cluster) + "\n")
#             f.write("------------------------------\n")
#     print(f"聚类结果已写入: {output_file}")
#
# # ======================
# # 主程序
# # ======================
# if __name__ == "__main__":
#     print(f"读取 TM-score 文件: {RAW_FILE}")
#     edges = parse_tmscore_file(RAW_FILE)
#     print(f"共解析 {len(edges)} 对有效比较结果")
#
#     print(f"使用 TM-score 阈值: {TM_THRESHOLD} (使用 {'平均值' if USE_AVG else '最大值'})")
#     clusters = clustering_by_tmscore(edges, threshold=TM_THRESHOLD)
#
#     # 输出簇文件
#     output_clusters(clusters, OUTPUT_FILE)
#
#     # 统计信息
#     all_prots = set()
#     for c in clusters:
#         all_prots.update(c)
#     print(f"\n聚类簇数: {len(clusters)}")
#     print(f"唯一蛋白质 ID 总数: {len(all_prots)}")


# #某个蛋白质与所有蛋白质的tmscore排序
# target = "5ec5.pdb"
# input_file = "/datapool/home/info_wang/tongyanting/GCL/beta_data/tmscore.txt"
#
# # 用字典保存每个配对蛋白的最高 TM-score
# best_tm = {}
#
# with open(input_file) as f:
#     header = f.readline()
#     for line in f:
#         if not line.strip():
#             continue
#         cols = line.strip().split('\t')
#         if len(cols) < 4:
#             continue
#
#         # 提取两个 pdb 名（如 1ar1.pdb）
#         id1 = cols[0].split('/')[-1].split('.pdb')[0] + '.pdb'
#         id2 = cols[1].split('/')[-1].split('.pdb')[0] + '.pdb'
#         try:
#             tm1 = float(cols[2])
#             tm2 = float(cols[3])
#         except ValueError:
#             continue
#
#         tm_max = max(tm1, tm2)
#
#         if id1 == target and tm_max > 0.5:
#             best_tm[id2] = max(best_tm.get(id2, 0), tm_max)
#         elif id2 == target and tm_max > 0.5:
#             best_tm[id1] = max(best_tm.get(id1, 0), tm_max)
#
# # 按 TM-score 降序排序输出
# sorted_hits = sorted(best_tm.items(), key=lambda x: x[1], reverse=True)
#
# print(f"与 {target} 的 TM-score > 0.5 的蛋白（按分数降序）：")
# for pid, score in sorted_hits:
#     print(f"{pid}\t{score:.4f}")




