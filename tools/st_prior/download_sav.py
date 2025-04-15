import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import pandas as pd
import subprocess

import requests
from concurrent.futures import ThreadPoolExecutor

def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查 HTTP 状态
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"Downloaded and saved: {filename}")


def download_files_multithreaded(urls, output_filenames, max_threads=4):
    assert len(urls) == len(output_filenames)

    with ThreadPoolExecutor(max_threads) as executor:
        for url, filename in zip(urls, output_filenames):
            executor.submit(download_file, url, filename)


def download_and_extract():
    meta_file = 'data/st_prior/data/SA-V/SA-V.csv'
    data = pd.read_csv(meta_file, delimiter='	')
    urls, output_filenames = [], []
    for _, row in tqdm(data.iterrows()):
        filename, cdn_link = row['file_name'], row['cdn_link']
        if '.tar' not in filename or any(flag in filename for flag in ['test', 'val']):
            continue
        local_path = f'./{filename}'
        # subprocess.run(["wget", cdn_link], check=True)
        # subprocess.run(["tar", "-xf", local_path], check=True)
        # subprocess.run(["rm", "-f", local_path], check=True)
        urls.append(cdn_link)
        output_filenames.append(local_path)

    download_files_multithreaded(urls, output_filenames)

    for filename in output_filenames:
        subprocess.run(["tar", "-xf", filename], check=True)
        subprocess.run(["rm", "-f", filename], check=True)


if __name__ == '__main__':
    download_and_extract()