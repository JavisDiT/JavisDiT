import argparse
import os
import os.path as osp
import pandas as pd
from tqdm import tqdm

import logging
logging.warning = lambda *args, **kwargs: None

import torch

import sys
sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)), "src"))
from ..src.dataset import create_dataloader_for_fvd_vanilla, create_dataloader_for_fvd_mmdiff
from ..src.fvd.fvd import get_fvd_logits
from ..src.fvd.download import load_i3d_pretrained
from ..src.AudioCLIP.get_embedding import load_audioclip_pretrained, get_audioclip_embeddings_scores

def extract_gt_fvd_fad_feat(args):
    df = pd.read_csv(args.input_file)

    gt_video_list = df['path'].tolist()
    gt_audio_list = df.get('audio_path', df['path']).tolist()

    mode = args.fvd_mode

    if mode == 'mmdiffusion':
        ## assume audios are integrated into videos
        data_loader = create_dataloader_for_fvd_mmdiff(
            gt_video_list, 
            f"{args.cache_dir}/mmdiff/real", 
            args.video_fps, 
            args.audio_sr
        )
    else:
        data_loader = create_dataloader_for_fvd_vanilla(
            gt_video_list, gt_audio_list, 
            max_frames=args.max_frames, 
            image_size=args.image_size,
            video_fps=args.video_fps,
            audio_sr=args.audio_sr,
            max_audio_len_s=args.max_audio_len_s,
            num_workers=args.num_workers,
        )
    
    device = args.device
    i3d = load_i3d_pretrained(device)
    audioclip = load_audioclip_pretrained(device)

    video_embeds, audio_embeds, indices = [], [], []
    for _, sample in enumerate(tqdm(data_loader, desc='extracting')):
        # b t h w c
        video_sample = sample['video'].to(device)
        audio_sample = sample['audio'].to(device)
        index_sample = sample['index'].to(device)

        video_embed = get_fvd_logits(video_sample, i3d, device=device)
        video_embeds.append(video_embed)
        indices.append(index_sample)

        _, audioclip_audio_embed, _ = get_audioclip_embeddings_scores(audioclip, video_sample, audio_sample)
        audio_embeds.append(audioclip_audio_embed)

    indices = torch.cat(indices).argsort()
    video_embeds = torch.cat(video_embeds)[indices]
    audio_embeds = torch.cat(audio_embeds)[indices]

    embed_dict = {'video': video_embeds, 'audio': audio_embeds}

    os.makedirs(osp.dirname(args.output_file), exist_ok=True)
    torch.save(embed_dict, args.output_file)

    print(f'Features saved to {args.output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="path to input csv file", required=True)
    parser.add_argument("--output_file", type=str, default=None, help="path to output json file", required=True)
    # parameters for evaluation
    parser.add_argument("--max_frames", type=int, default=24, help="size of the input video")
    parser.add_argument("--max_audio_len_s", type=float, default=None, help="maximum length of the audio", required=True)
    parser.add_argument("--video_fps", type=int, default=24, help="frame rate of the input video")
    parser.add_argument("--audio_sr", type=int, default=16000, help="sampling rate of the audio")
    parser.add_argument("--image_size", type=int, default=224, help="size of the input image")
    parser.add_argument("--fvd_mode", type=str, default='vanilla', choices=['vanilla', 'mmdiffusion'], help="mode of fvd calculation, `video` or `audio`")
    # parameters for acceleration
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for data loading")
    parser.add_argument("--device", type=str, default='cuda:0', help="device to load models")
    args = parser.parse_args()

    cache_dir = f'{osp.dirname(args.output_file)}/cache/{osp.basename(osp.splitext(args.output_file)[0])}'
    setattr(args, "cache_dir", cache_dir)

    extract_gt_fvd_fad_feat(args)
