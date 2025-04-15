import sys
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import yaml
import argparse


def preprocess_audiocaps(data_root):
    # https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link
    data_dir = f'{data_root}/AudioCaps'
    src_file = f'{data_dir}/dataset/metadata/audiocaps/datafiles/audiocaps_train_label.json'
    with open(src_file, 'r') as f:
        data = json.load(f)['data']
    audios, captions = [], []
    for item in tqdm(data, desc='AudioCaps'):
        audio_path = f'{data_dir}/dataset/audioset/' + item['wav']
        if not osp.exists(audio_path):
            break
        audios.append(audio_path)
        captions.append(item['caption'])
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_audiocaps_test(data_root):
    # https://drive.google.com/file/d/16J1CVu7EZPD_22FxitZ0TpOd__FwzOmx/view?usp=drive_link
    data_dir = f'{data_root}/AudioCaps'
    src_file = f'{data_dir}/dataset/metadata/audiocaps/datafiles/audiocaps_test_nonrepeat_subset_2.json'
    with open(src_file, 'r') as f:
        data = json.load(f)['data']
    audios, captions = [], []
    for item in tqdm(data, desc='AudioCaps_test'):
        name = osp.sep.join(item['wav'].split('/')[-2:])
        audio_path = f'{data_dir}/dataset/audioset/zip_audios/unbalanced_train_segments/' + name
        assert osp.exists(audio_path), audio_path
        audios.append(audio_path)
        captions.append(item['caption'])
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_vggsound(data_root):
    # https://huggingface.co/datasets/Loie/VGGSound
    data_dir = f'{data_root}/VGGSound'
    src_file = f'{data_dir}/vggsound.csv'
    data = pd.read_csv(src_file)
    audios, captions = [], []
    for i in tqdm(range(len(data)), desc='VGGSound'):
        _id, start, label, split = data.iloc[i]
        if split != 'train':
            continue
        audio_path = f'{data_dir}/audio/{_id}_{int(start):06d}.wav'
        if not osp.exists(audio_path):
            continue
        audios.append(audio_path)
        captions.append(label)
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_audioset(data_root):
    # https://huggingface.co/datasets/agkphysics/AudioSet
    # http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
    data_dir = f'{data_root}/AudioSet'
    if not osp.exists(f'{data_dir}/class_labels_indices.csv'):
        raise ValueError('Download the `class_labels_indices.csv` file from'
                         'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv '
                         'and put it into `\{data_root\}/AudioSet/` first.')
    map_data = pd.read_csv(f'{data_dir}/class_labels_indices.csv')
    label_map = {}
    for i in range(len(map_data)):
        _, label, caption = map_data.iloc[i]
        label_map[label] = caption.lower()
    src_file = f'{data_dir}/balanced_train_segments.csv'
    data = pd.read_csv(src_file, sep=', ')
    audios, captions = [], []
    for i in tqdm(range(len(data)), desc='AudioSet'):
        _id, start, end, label = data.iloc[i]
        audio_path = f'{data_dir}/audio/bal_train/{_id}.flac'
        if not osp.exists(audio_path):
            continue
        audios.append(audio_path)
        labels = label.strip()[1:-1].split(',')
        caption = ', '.join(label_map[label] for label in labels)
        captions.append(caption)
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_wavcaps(data_root):
    # https://huggingface.co/datasets/cvssp/WavCaps
    data_dir = f'{data_root}/WavCaps'
    subset_dict = {
        'as': 'AudioSet_SL', 'bbc': 'BBC_Sound_Effects', 'sb': 'SoundBible',
        'fsd': 'FreeSound'
    }
    audios, captions = [], []
    for jname, aname in subset_dict.items():
        src_file = f'{data_dir}/json_files/{jname}_final.json'
        with open(src_file, 'r') as f:
            data = json.load(f)['data']
        audio_dir = f'{data_dir}/audio_files/{aname}_flac'
        for item in data:
            audio_id = osp.splitext(item["id"])[0]
            audio_path = f'{audio_dir}/{audio_id}.flac'
            if not osp.exists(audio_path):
                continue
            audios.append(audio_path)
            captions.append(item['caption'])
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_clotho(data_root):
    # https://zenodo.org/records/3490684
    data_dir = f'{data_root}/Clotho'
    audios, captions = [], []
    subset_list = ['development', ]  # 'evaluation' subset is eliminated
    for subset in subset_list:
        src_file = f'{data_dir}/clotho_captions_{subset}.csv'
        data = pd.read_csv(src_file)
        for i, row in data.iterrows():
            filename, caption = row['file_name'], row['caption_1']
            audio_path = f'{data_dir}/{subset}/{filename}'
            if not osp.exists(audio_path):
                continue
            audios.append(audio_path)
            captions.append(caption)
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_esc50(data_root):
    # https://github.com/karolpiczak/ESC-50?tab=readme-ov-file#download
    data_dir = f'{data_root}/ESC50'
    audios, captions = [], []
    src_file = f'{data_dir}/meta/esc50.csv'
    data = pd.read_csv(src_file)
    for i, row in data.iterrows():
        filename, caption = row['filename'], row['category'].replace('_', ' ')
        audio_path = f'{data_dir}/audio/{filename}'
        if not osp.exists(audio_path):
            continue
        audios.append(audio_path)
        captions.append(caption)
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_macs(data_root):
    # https://zenodo.org/records/5114771
    # https://zenodo.org/records/2589280
    data_dir = f'{data_root}/MACS'
    audios, captions = [], []
    src_file = f'{data_dir}/MACS.yaml'
    if not osp.exists(src_file):
        raise ValueError('Download the `MACS.yaml` file from'
                         'https://zenodo.org/records/5114771 '
                         'and put it into `\{data_root\}/MACS/` first.')
    with open(src_file, 'r') as f:
        data = yaml.safe_load(f)['files']
    ann = pd.read_csv(f'{data_dir}/MACS_competence.csv', delimiter='	')
    ann2score = {row['annotator_id']: row['competence'] for _, row in ann.iterrows()}
    for item in data:
        filename = item['filename']
        audio_path = f'{data_dir}/audio/{filename}'
        if not osp.exists(audio_path):
            continue
        ann_ids = [ann['annotator_id'] for ann in item['annotations']]
        ann_caps = [ann['sentence'] for ann in item['annotations']]
        ann_scores = [ann2score[annid] for annid in ann_ids]
        caption = ann_caps[np.argmax(ann_scores)]
        audios.append(audio_path)
        captions.append(caption)

    # total_files = glob(f'{data_dir}/audio/*.*')
    # invalid_files = set(total_files) - set(audios)
    # for path in tqdm(invalid_files, desc='removing'):
    #     os.remove(path)
    # print(f'{len(invalid_files)} files have been removed.')
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_urbansound(data_root):
    # https://urbansounddataset.weebly.com/urbansound8k.html
    # https://github.com/soundata/soundata#quick-example
    data_dir = f'{data_root}/UrbanSound8K'
    audios, captions = [], []
    src_file = f'{data_dir}/metadata/UrbanSound8K.csv'
    data = pd.read_csv(src_file)
    name2cap = {row['slice_file_name']: row['class'].replace('_', ' ') \
                    for _, row in data.iterrows()}
    for audio_path in sorted(glob(f'{data_dir}/audio/fold*/*.*')):
        filename = osp.basename(audio_path)
        if filename not in name2cap:
            continue
        audios.append(audio_path)
        captions.append(name2cap[filename])
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_musicinstrument(data_root):
    # https://www.kaggle.com/datasets/soumendraprasad/musical-instruments-sound-dataset
    data_dir = f'{data_root}/MusicInstrument/data'
    audios, captions = [], []
    src_file = f'{data_dir}/Metadata_Train.csv'
    data = pd.read_csv(src_file)
    name2cap = {}
    for _, row in data.iterrows():
        if row['FileName'] in name2cap:
            continue
        name2cap[row['FileName']] = row['Class'].replace('_', ' ')
    for audio_path in sorted(glob(f'{data_dir}/Train_submission/Train_submission/*.*')):
        filename = osp.basename(audio_path)
        if filename not in name2cap:
            # print(audio_path)
            continue
        audios.append(audio_path)
        captions.append(name2cap[filename])
    
    return {path: cap for path, cap in zip(audios, captions)}


def preprocess_gtzan(data_root):
    # https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    data_dir = f'{data_root}/GTZAN/data/genres_original'
    audios, captions = [], []
    for audio_path in sorted(glob(f'{data_dir}/*/*.*')):
        caption = f'music {osp.basename(osp.dirname(audio_path))}'
        audios.append(audio_path)
        captions.append(caption)
    
    return {path: cap for path, cap in zip(audios, captions)}


def get_all_training_meta(data_root, meta_file, save_file):
    print('reading ...')
    path2cap = {}
    for ds in [
        'AudioCaps', 'VGGSound', 'AudioSet', 'WavCaps', 
        'Clotho', 'ESC50', 'MACS', 'UrbanSound8K', 'MusicInstrument', 'GTZAN',
    ]:
        path2cap.update(eval(f'preprocess_{ds}')(data_root))

    print('processing ...')
    df = pd.read_csv(meta_file)
    audio_caps = []
    for path in df['audio_path'].tolist():
        if path not in path2cap:
            print(f'Warning: cannot find the corresponding caption for {path}')
            audio_caps.append('')
        else:
            audio_caps.append(path2cap[path])
    
    print('saving ...')
    df['audio_text'] = audio_caps
    total_num = len(df)
    df = df[df['audio_text'].str.len() > 0]
    valid_num = len(df)
    if save_file is None:
        save_file = meta_file.replace('.csv', '_training.csv')
    df.to_csv(save_file, index=False)
    print(f'{valid_num}/{total_num} samples saved to {save_file}')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dataset processor')
    parser.add_argument('dataset', type=str, default='all', help='Dataset to process')
    parser.add_argument('--data_root', type=str, default='/path/to/audios', help='', required=True)
    parser.add_argument('--meta_file', type=str, default='/path/to/meta.csv', help='', required=True)
    parser.add_argument('--save_file', type=str, default=None, help='')
    args = parser.parse_args()

    eval(f'get_{args.dataset}_training_meta')(args.data_root, args.meta_file, args.save_file)