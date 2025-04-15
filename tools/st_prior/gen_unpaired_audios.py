import argparse
import scipy
import pandas as pd
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from datetime import timedelta

import torch
import torch.distributed as dist

from diffusers import AudioLDM2Pipeline

from colossalai.utils import get_current_device, set_seed


def infer_audioldm2(
    rank, world_size, pipe, data: pd.DataFrame, bs: int, 
    audio_length_in_s: float = 10.24, match_duration: bool = False
):
    device = get_current_device()
    pipe = pipe.to(device)

    num_per_node = int(np.ceil(len(data) / world_size))
    prompts, durations, out_fps, out_paths = [], [], [], []
    for i in range(rank*num_per_node, min((rank+1)*num_per_node, len(data))):
        row = data.iloc[i]
        if osp.exists(row['unpaired_audio_path']):
            continue
        prompts.append(row['text'])
        durations.append(row['num_frames'] / row['fps'])
        out_fps.append(row['audio_fps'])
        out_paths.append(row['unpaired_audio_path'])

    for i in range(0, len(prompts), bs):
        if rank == 0:
            print(f'\nProcessing {i}/{len(prompts)}')
        audios = pipe(prompts[i:i+bs], num_inference_steps=200, audio_length_in_s=audio_length_in_s).audios
        for j, audio in enumerate(audios):
            try:
                # audio padding
                video_length_in_s = durations[i+j]
                if match_duration:
                    if video_length_in_s < audio_length_in_s:
                        audio = audio[:int(video_length_in_s*out_fps[i+j])]
                    else:
                        audio = np.pad(audio, [0, int((video_length_in_s-audio_length_in_s)*out_fps[i+j])])
                scipy.io.wavfile.write(out_paths[i+j], rate=out_fps[i+j], data=audio)
            except Exception as e:
                print(e)
                # data.iloc[i+j, 'unpaired_audio_path'] = ""

def main(args):
    ## prepare data and model
    dist.init_process_group("nccl", timeout=timedelta(hours=24))

    rank = dist.get_rank()
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank % world_size)

    set_seed(args.seed)

    pipe = AudioLDM2Pipeline.from_pretrained(args.model_name_or_path)

    data = pd.read_csv(args.input_meta)
    os.makedirs(args.output_dir, exist_ok=True)
    audio_paths = [f'{args.output_dir}/{osp.splitext(osp.basename(path))[0]}.wav' for path in data['path']]
    data['unpaired_audio_path'] = audio_paths
    if rank == 0:
        output_meta = args.output_meta or args.input_meta.replace('.csv', '_unpaired_audios.csv')
        data.to_csv(output_meta, index=False)
    data = data[~np.array([osp.exists(path) for path in audio_paths])]

    infer_audioldm2(rank, world_size, pipe, data, args.batch_size, args.audio_length_in_s, args.match_duration)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name_or_path', type=str, default='cvssp/audioldm2')
    parser.add_argument('--audio_length_in_s', type=float, default=10.24)
    parser.add_argument('--input_meta', type=str, default='/path/to/train_jav.csv')
    parser.add_argument('--output_meta', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='./data/st_prior/audio/unpaired')
    parser.add_argument('--match_duration', action='store_true')
    args = parser.parse_args()

    main(args)
    # torchrun --nproc_per_node=4 xx.py