import argparse
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torchaudio
import soundfile as sf
import warnings
import re
from glob import glob

warnings.filterwarnings("ignore")

import sys
sys.path.append('./third_party/AudioSep')
from pipeline import build_audiosep
import logging
from tqdm import tqdm
import math

logging.getLogger().setLevel(logging.INFO)


local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", -1))
torch.cuda.set_device(local_rank)
logging.info(f'rank {rank} local_rank {local_rank}')


class AudioDataset(Dataset):
    def __init__(self, data_path, descriptions_path, target_rate=32000, duration=10):
        super().__init__()
        self.data = sorted(glob(os.path.join(data_path, '*.wav'))) # audio files
        self.target_rate = target_rate
        self.duration = duration
        with open(descriptions_path, 'r') as f:
            self.descriptions = json.load(f) # from qwen output
        
        # match audio with descriptions
        descs = []
        pop_idx = []
        for i in range(len(self.data)):
            audio_name = os.path.basename(self.data[i]).split('.wav')[0]
            if audio_name not in self.descriptions:
                pop_idx.append(i)
                continue
            descs.append(self.descriptions[audio_name]) 
        for i in sorted(pop_idx, reverse=True):
            self.data.pop(i)
        
        self.descriptions = descs
        assert len(self.descriptions) == len(self.data)
        logging.info(f'total length {len(self.data)}')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]
        descriptions = self.descriptions[index]
        audio, sr = sf.read(audio_path)
        if audio.size == 0:
            logging.warning(f'{audio_path}: audio size is 0')
            return torch.zeros(self.duration * self.target_rate, dtype=torch.float64), audio_path

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        audio = torch.tensor(audio).reshape(1, -1)
        original_audio = audio.clone()
        origin_length = audio.shape[1]
        origin_sr = sr

        if sr != self.target_rate:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=self.target_rate)

        if audio.shape[1] < self.duration * self.target_rate:
            audio = torch.cat([audio, torch.zeros(1, self.duration * self.target_rate - audio.shape[1])], dim=1)
        else:
            audio = audio[:, :self.duration * self.target_rate]
        return audio[0], original_audio[0], descriptions, audio_path, origin_sr, origin_length


def collate_fn(batch):
    audio, original_audio, desc, audio_path, origin_sr, origin_length = zip(*batch)
    audios = torch.stack(audio)
    original_audios = [oa for oa in original_audio]
    descs = [d for d in desc]
    audio_path = [p for p in audio_path]
    origin_sr = [s for s in origin_sr]
    origin_length = [l for l in origin_length]
    return audios, original_audios, descs, audio_path, origin_sr, origin_length


@torch.no_grad()
def separate_audio(model, audio, text, device, use_chunk=False):
    conditions = model.query_encoder.get_query_embed(
        modality='text',
        text=text,
        device=device
    )

    input_dict = {
        'mixture': audio[None, None, :].to(device).float(),
        'condition': conditions,
    }

    if use_chunk:
        sep_segment = model.ss_model.chunk_inference(input_dict)
        sep_segment = np.squeeze(sep_segment)
    else:
        sep_segment = model.ss_model(input_dict)["waveform"]
        sep_segment = sep_segment.squeeze(1)
    return sep_segment


def rms(arr, chunks):
    arr = np.array_split(arr, chunks)
    return [np.sqrt(np.mean(np.square(a), axis=-1)) for a in arr]


def merge_intervals_no_overlap(intervals):
    if not intervals:
        return []
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] == last[1]:
            last[1] = current[1]  # 更新合并后的区间的结束值
        else:
            merged.append(current)
    return merged


def main(args):
    dataset = AudioDataset(args.audio_path, args.descriptions_path)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, sampler=sampler, collate_fn=collate_fn)

    audiosep_model = build_audiosep(
        config_yaml = './third_party/AudioSep/config/audiosep_base.yaml',
        checkpoint_path = './third_party/AudioSep/checkpoint/audiosep_base_4M_steps.ckpt',
        device = local_rank
    )

    if local_rank == 0 or local_rank == -1:
        pbar = tqdm(range(len(dataloader)), ncols=50)

    for data in dataloader:
        audios, original_audios, descriptions, audio_paths, origin_srs, origin_lengths = data

        for audio, origin_audio, desc, audio_path, origin_sr, origin_length in zip(audios, original_audios, descriptions, audio_paths, origin_srs, origin_lengths):
            origin_audio = origin_audio.data.cpu().numpy()
            desc = [d for d in list(set(desc)) if ('(' not in d) and (')' not in d) and ('background' not in d) and (d != 'None')]
            results = {}
            audio_name = os.path.basename(audio_path)
            save_path = os.path.join(args.output_path, f'{audio_name}.json')
            if os.path.exists(save_path):
                continue
            
            for i in range(0, (len(desc) + 3) // 4):
                chunk_desc = desc[i:i+4]
                sep_audios = separate_audio(audiosep_model, audio, chunk_desc, local_rank)
                sep_audios = torchaudio.functional.resample(sep_audios, orig_freq=32000, new_freq=origin_sr).data.cpu().numpy()
                # 先存下来再说
                for j, sep_audio in enumerate(sep_audios):
                    path = os.path.join(args.output_path, f'{audio_name}__sep__{i:03d}_{j:03d}_{chunk_desc[j]}.wav')
                    sep_audio = sep_audio[:origin_length]
                    # 过滤纯静音片段（说明没有什么检测出来的，或者至少在这个视频里它不是主角）
                    chunks = math.ceil(len(sep_audio) / origin_sr)
                    chunk_rms = rms(sep_audio, chunks)
                    # if np.max(chunk_rms) < 0.1:
                    #     # 没有静音片段，不值得留下来
                    #     continue
                    # 检测出来rms比较大作为事件的起始结束时间？
                    try:  # filter invalid path name, e.g, `/` in `opening/closing drawers.wav`
                        sf.write(path, sep_audio, origin_sr)
                        results[f'{chunk_desc[j]}_{i:03d}_{j:03d}'] = merge_intervals_no_overlap([[i * origin_sr, min((i + 1) * origin_sr, origin_length)] for i, r in enumerate(chunk_rms) if r >= 0.1])
                    except:
                        continue

            if len(results) > 0:
                with open(save_path, 'w') as f:
                    json.dump(results, f, indent=4)

        if local_rank == 0 or local_rank == -1:
            pbar.update(1)
    
    if local_rank == 0 or local_rank == -1:
        pbar.close()


if __name__ == '__main__':
    dist.init_process_group('nccl', init_method='env://')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--descriptions_path', type=str, default='./data/st_prior/audio/TAVGBench/qwen_output.json')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True) 

    main(args)

    dist.destroy_process_group()
