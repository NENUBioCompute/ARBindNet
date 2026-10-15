import requests
import time
import os

out_dir = './TMPDB'
os.makedirs(out_dir, exist_ok=True)

if not os.access(out_dir, os.W_OK):
    print(f'错误：无法写入文件夹 {out_dir}，请检查权限')
    exit(1)

with open('./pdb_list.txt', 'r') as f:
    ids = [line.strip().replace('.pdb', '') for line in f if line.strip()]

for pdb_id in ids:
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    filepath = os.path.join(out_dir, f'{pdb_id}.pdb')
    print(f'下载中: {pdb_id}')
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(filepath, 'w') as f:
            f.write(resp.text)
        time.sleep(1)
    except Exception as e:
        print(f'  {pdb_id} 下载失败: {e}')



