import os
import os.path as osp
import json
import argparse
from tqdm import tqdm
from glob import glob
from typing import Dict, List
import re

import pandas as pd
import soundfile as sf


def get_tavgbench_training_meta(meta_src, meta_file, save_file):
    df = pd.read_csv(meta_file)

    with open(meta_src, 'r') as f:
        data_all = f.read().splitlines()
    video2cap = {}
    for line in tqdm(data_all, desc="extracting"):
        video_id, *caps = line.split(' ')
        if not (video_id.endswith('.mp4') and video_id.count("_") >= 2):
            continue
        if len(caps) == 0:
            continue
        video2cap[osp.splitext(video_id)[0]] = " ".join(caps)
    
    text = [video2cap.get(video_id, "") for video_id in tqdm(df['id'], desc="processing")]

    print('saving ...')
    df['text'] = text
    total_num = len(df)
    df = df[df["text"].str.len() > 0]
    valid_num = len(df)
    if save_file is None:
        save_file = meta_file.replace('.csv', '_training.csv')
    df.to_csv(save_file, index=False)
    print(f'{valid_num}/{total_num} samples saved to {save_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('dataset', type=str, help='Dataset to process', choices=['tavgbench'], required=True)
    parser.add_argument('--meta_src', type=str, default='/path/to/TAVGBench_clean/release_captions_clean.txt', help='', required=True)
    parser.add_argument('--meta_file', type=str, default='/path/to/meta.csv', help='', required=True)
    parser.add_argument('--save_file', type=str, default=None, help='')
    args = parser.parse_args()

    eval(f'get_{args.dataset}_training_meta')(args.meta_src, args.meta_file, args.save_file)
