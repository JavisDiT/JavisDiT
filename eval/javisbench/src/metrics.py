from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import os.path as osp
import shutil
import random
import math
from typing import List, Dict, Optional

import librosa
from moviepy.editor import VideoFileClip
from PIL import Image
import cv2

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

from .fvd.fvd import get_fvd_logits, frechet_distance
from .fvd.download import load_i3d_pretrained
from .AudioCLIP.get_embedding import load_audioclip_pretrained, get_audioclip_embeddings_scores
from .utils import polynomial_mmd, Extract_CAVP_Features

sys.path.append(os.path.join(os.path.dirname(__file__), "ImageBind"))
from .ImageBind.imagebind import data
from .ImageBind.imagebind.models import imagebind_model
from .ImageBind.imagebind.models.imagebind_model import ModalityType

from .dataset import (
    create_dataloader, 
    create_dataloader_for_fvd_vanilla, create_dataloader_for_fvd_mmdiff
)


def calc_fvd_kvd_fad(
    gt_video_list, pred_video_list, gt_audio_list, pred_audio_list, device="cuda:0", 
    cat2indices: List[List[List[int]]] = None, eval_num=None, mode="vanilla", **kwargs
):
    # Original code from "https: //github.com/researchmm/MM-Diffusion"
    
    fvd_avcache_path = kwargs.get('fvd_avcache_path', None)
    if fvd_avcache_path is not None:
        gt_embed_dict = torch.load(fvd_avcache_path)
        gt_embed_dict = {k: v.to(device) for k, v in gt_embed_dict.items()}
        use_gt_av_cache = True
        print(f'Ground-truth AV cache loaded from {fvd_avcache_path}')
    else:
        gt_embed_dict = None
        use_gt_av_cache = False

    if mode == 'mmdiffusion':
        cache_dir = kwargs.get('cache_dir')
        fps = kwargs.get('video_fps', 24)
        sr = kwargs.get('audio_sr', 16000)
        ## assume audios are integrated into videos
        if not use_gt_av_cache:
            real_loader = create_dataloader_for_fvd_mmdiff(gt_video_list, f"{cache_dir}/mmdiff/real", fps, sr)
        fake_loader = create_dataloader_for_fvd_mmdiff(pred_video_list, f"{cache_dir}/mmdiff/fake", fps, sr)
    else:
        if not use_gt_av_cache:
            real_loader = create_dataloader_for_fvd_vanilla(gt_video_list, gt_audio_list, **kwargs)
        fake_loader = create_dataloader_for_fvd_vanilla(pred_video_list, pred_audio_list, **kwargs)

    # load models
    i3d = load_i3d_pretrained(device)
    audioclip = load_audioclip_pretrained(device)

    if not use_gt_av_cache:
        loader_dict = {'real': real_loader, 'fake': fake_loader}
        embed_dict = {}
    else:
        loader_dict = {'fake': fake_loader}
        embed_dict = {'real': gt_embed_dict}

    for t, loader in loader_dict.items():
        video_embeds, audio_embeds, indices = [], [], []
        cnt = 0
        for _, sample in enumerate(tqdm(loader, desc=f'fvd_kvd_fad - {t}')):
            # b t h w c
            video_sample = sample['video'].to(device)
            audio_sample = sample['audio'].to(device)
            index_sample = sample['index'].to(device)

            video_embed = get_fvd_logits(video_sample, i3d, device=device)
            video_embeds.append(video_embed)
            indices.append(index_sample)

            _, audioclip_audio_embed, _ = get_audioclip_embeddings_scores(audioclip, video_sample, audio_sample)
            audio_embeds.append(audioclip_audio_embed)

            cnt += video_sample.shape[0]
            if eval_num and cnt >= eval_num: 
                break
        indices = torch.cat(indices).argsort()
        video_embeds = torch.cat(video_embeds)[indices][:eval_num]
        audio_embeds = torch.cat(audio_embeds)[indices][:eval_num]

        embed_dict[t] = {'video': video_embeds, 'audio': audio_embeds}
    
    fvd = frechet_distance(embed_dict['fake']['video'], embed_dict['real']['video']).item()
    kvd = polynomial_mmd(embed_dict['fake']['video'].cpu().numpy(), embed_dict['real']['video'].cpu().numpy()).item()
    fad = frechet_distance(embed_dict['fake']['audio'], embed_dict['real']['audio']).item() * 10000

    if cat2indices is not None:
        fvd, kvd, fad = {'overall': fvd}, {'overall': kvd}, {'overall': fad}
        for ai, index_list in enumerate(cat2indices):
            fvd[ai], kvd[ai], fad[ai] = [], [], []
            for ci, indices in enumerate(index_list):
                fake_video_embed = embed_dict['fake']['video'][indices]
                real_video_embed = embed_dict['real']['video'][indices]
                fake_audio_embed = embed_dict['fake']['audio'][indices]
                real_audio_embed = embed_dict['real']['audio'][indices]

                fvd[ai].append(frechet_distance(fake_video_embed, real_video_embed).item())
                kvd[ai].append(polynomial_mmd(fake_video_embed.cpu().numpy(), real_video_embed.cpu().numpy()).item())
                fad[ai].append(frechet_distance(fake_audio_embed, real_audio_embed).item() * 10000)

    return fvd, kvd, fad


def calc_imagebind_score(video_list, audio_list, prompt_list, audio_prompt_list=None,
                         device='cuda:0', cat2indices: List[List[List[int]]] = None, bs=1):
    # Original code from "https://github.com/sonyresearch/svg_baseline"

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    if audio_prompt_list is None:
        audio_prompt_list = prompt_list
    text_embeds, audio_text_embeds, video_embeds, audio_embeds = [], [], [], []
    for i in tqdm(range(0, len(video_list), bs), desc='ib_score'):
        prompts = prompt_list[i:i+bs] + audio_prompt_list[i:i+bs]
        videos, audios = video_list[i:i+bs], audio_list[i:i+bs]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(prompts, device),
            ModalityType.VISION: data.load_and_transform_video_data(videos, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        text_embed, audio_text_embed = embeddings[ModalityType.TEXT].chunk(2, dim=0)
        text_embeds.append(text_embed)
        audio_text_embeds.append(audio_text_embed)
        video_embeds.append(embeddings[ModalityType.VISION])
        audio_embeds.append(embeddings[ModalityType.AUDIO])

    text_embeds, audio_text_embeds = torch.cat(text_embeds), torch.cat(audio_text_embeds)
    video_embeds, audio_embeds = torch.cat(video_embeds), torch.cat(audio_embeds)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_tv_list = cos(text_embeds, video_embeds)
    sim_tv = sim_tv_list.mean().item()
    sim_ta_list = cos(audio_text_embeds, audio_embeds)
    sim_ta = sim_ta_list.mean().item()
    sim_av_list = cos(video_embeds, audio_embeds)
    sim_av = sim_av_list.mean().item()
    
    if cat2indices is not None:
        sim_tv = {'overall': sim_tv, 'all': sim_tv_list.tolist()}
        sim_ta = {'overall': sim_ta, 'all': sim_ta_list.tolist()}
        sim_av = {'overall': sim_av, 'all': sim_av_list.tolist()}
        for ai, index_list in enumerate(cat2indices):
            sim_tv[ai], sim_ta[ai], sim_av[ai] = [], [], []
            for ci, indices in enumerate(index_list):
                text_embeds_sub, audio_text_embeds_sub, video_embeds_sub, audio_embeds_sub = \
                    text_embeds[indices], audio_text_embeds[indices], video_embeds[indices], audio_embeds[indices]
                sim_tv[ai].append(cos(text_embeds_sub, video_embeds_sub).mean().item())
                sim_ta[ai].append(cos(audio_text_embeds_sub, audio_embeds_sub).mean().item())
                sim_av[ai].append(cos(video_embeds_sub, audio_embeds_sub).mean().item())

    return sim_tv, sim_ta, sim_av


def calc_clip_score(video_list, prompt_list, device='cuda:0', cat2indices=None, num_frames=48):
    # load clip model
    device = device
    model, preprocess = clip.load("ViT-B/32", device=device)

    def _frame_transform(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        return frame

    dataloader = create_dataloader(
        metric='clip-score', 
        video_path_list=video_list,
        prompt_list=prompt_list,
        num_frames=num_frames,
        frame_transform=_frame_transform,
        batch_size=1
    )

    clip_score_list, indices_list = [], []
    for frames, prompts, indices in tqdm(dataloader, desc='clipscore'):
        assert frames.shape[0] == len(prompts) == len(indices) == 1  
        frames = frames.to(device)

        with torch.no_grad():
            text = clip.tokenize(prompts, truncate=True).to(device)
            text_features = model.encode_text(text)
            image_features = model.encode_image(frames[0])
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = (image_features @ text_features.T).mean()
            clip_score_list.append(score.item())
        indices_list.append(indices[0])
    
    clip_scores = np.array(clip_score_list)
    indices = np.array(indices_list)
    clip_scores = clip_scores[indices.argsort()]

    clip_score = np.mean(clip_scores)

    if cat2indices is not None:
        clip_score = {'overall': clip_score, 'all': clip_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            clip_score[ai] = []
            for ci, indices in enumerate(index_list):
                clip_score[ai].append(np.mean(clip_scores[indices]))

    return clip_score


def calc_clap_score(audio_list, prompt_list, device='cuda:0', cat2indices=None):
    # Original code from "https://github.com/sonyresearch/svg_baseline"
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.to(device=device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    dataloader = create_dataloader(
        metric='clap-score', 
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sr=48000,   # CLAP requires sample_rate=48000
        max_audio_len_s=None,
        batch_size=1
    )

    score_list, index_list = [], []
    for audios, prompts, indices in tqdm(dataloader, desc='clapscore'):
        assert len(audios) == len(prompts) == len(indices) == 1
        inputs = processor(text=prompts[0], audios=audios[0].squeeze(), 
                           return_tensors="pt", padding=True, 
                           sampling_rate=48000)   # CLAP requires sample_rate=48000
        inputs.to(device=device)
        outputs = model(**inputs)
        scores = cos(outputs.text_embeds, outputs.audio_embeds).mean()
        score_list.append(scores)
        index_list.append(indices)
    
    indices = torch.cat(index_list).flatten()
    clap_scores = torch.tensor(score_list)[indices.argsort()]

    clap_score = clap_scores.mean().item()

    if cat2indices is not None:
        clap_score = {'overall': clap_score, 'all': clap_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            clap_score[ai] = []
            for ci, indices in enumerate(index_list):
                clap_score[ai].append(torch.mean(clap_scores[indices]).item())

    return clap_score


def calc_cavp_score(video_list, audio_list, device='cuda:0', cat2indices=None, sample_rate=16000,
                    cavp_ckpt_path='./weights/cavp_epoch66.ckpt',
                    cavp_config_path='./configs/Stage1_CAVP.yaml'):
    # Original code from "https://github.com/sonyresearch/svg_baseline"

    fps = 4  #  CAVP default FPS=4, Don't change it.
    batch_size = 40  # Don't change it.

    # Initalize CAVP Model:
    extract_cavp = Extract_CAVP_Features(fps=fps,
                                        batch_size=batch_size,
                                        device=device,
                                        config_path=cavp_config_path,
                                        ckpt_path=cavp_ckpt_path)
    dataloader = create_dataloader(
        metric='cavp-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        sr=sample_rate,
        batch_size=1
    )

    tmp_path = "./tmp"
    cavp_scores, indices_list = [], []
    for video_paths, audios, truncate_seconds, indices in tqdm(dataloader, desc='cavpscore'):
        assert len(video_paths) == len(audios) == len(truncate_seconds) == len(indices) == 1

        # Extract Video CAVP Features & New Video Path:
        try:  # TODO: debug
            cavp_feats, new_video_path = \
                extract_cavp(video_paths[0], 0, truncate_seconds[0].item(), tmp_path=tmp_path)
        except:
            cavp_scores.append(0.0)
            indices_list.append(indices[0])
            continue

        spec = audios.unsqueeze(1).cuda().float()  # B x 1 x Mel x T
        spec = spec.permute(0, 1, 3, 2)  # B x 1 x T x Mel
        spec_feat = extract_cavp.stage1_model.spec_encoder(spec)  # B x T x C
        spec_feat = extract_cavp.stage1_model.spec_project_head(
            spec_feat).squeeze()
        spec_feat = F.normalize(spec_feat, dim=-1)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        score = cos(torch.from_numpy(cavp_feats).cuda(), spec_feat).mean().item()
        
        cavp_scores.append(score)
        indices_list.append(indices[0])
    
    indices = torch.tensor(indices_list).flatten()
    cavp_scores = torch.tensor(cavp_scores)[indices.argsort()]

    cavp_score = cavp_scores.mean().item()

    if cat2indices is not None:
        cavp_score = {'overall': cavp_score, 'all': cavp_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            cavp_score[ai] = []
            for ci, indices in enumerate(index_list):
                cavp_score[ai].append(torch.mean(cavp_scores[indices]).item())

    return cavp_score


def calc_av_align(video_list, audio_list, cat2indices=None, size=None, return_score_list=False):
    # Original code from "https://yzxing87.github.io/Seeing-and-Hearing/"

    dataloader = create_dataloader(
        metric='av-align', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        size=size,
        batch_size=1
    )

    align_score_list, index_list = [], []
    for align_score, index in tqdm(dataloader, desc='av-align'):
        align_score_list.append(align_score)
        index_list.append(index)

    indices = torch.cat(index_list).argsort()
    align_scores = torch.cat(align_score_list)[indices]

    align_score = align_scores.mean().item()

    if cat2indices is not None:
        align_score = {'overall': align_score, 'all': align_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            align_score[ai] = []
            for ci, indices in enumerate(index_list):
                align_score[ai].append(torch.mean(align_scores[indices]).item())

    if return_score_list:
        return align_score, align_scores
    else:
        return align_score


def calc_av_score(video_list, audio_list, prompt_list, device='cuda:0', cat2indices=None,
                  sample_rate=16000, window_size_s=0.5, window_overlap_s=0, topk_min=0.4,
                  return_score_list=False):
    
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    dataloader = create_dataloader(
        metric='av-score', 
        video_path_list=video_list,
        audio_path_list=audio_list,
        prompt_list=prompt_list,
        sample_rate=sample_rate,
        window_size_s=window_size_s,
        window_overlap_s=window_overlap_s,
        batch_size=1
    )

    avh_score_list, javis_score_list, index_list = [], [], []
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    for avh_inputs, javis_inputs, video_windows_indices, index in tqdm(dataloader, desc='av-score'):
        assert len(index) == video_windows_indices.shape[0] == 1

        # image shape: (B,C,H,W), video shape: (B,15,C,2,H,W), audio shape(B,3,C,T,S), 
        avh_inputs = {k: v[0].to(device) for k, v in avh_inputs.items()}
        javis_inputs = {k: v[0].to(device) for k, v in javis_inputs.items()}
        video_windows_indices = video_windows_indices[0]

        # for AVHScore
        with torch.no_grad():
            embeddings = model(avh_inputs)
        embed_frames = embeddings[ModalityType.VISION]  # shape(T,1024)
        embed_audio = embeddings[ModalityType.AUDIO]    # shape(1,1024)
        avh_score = cos(embed_frames, embed_audio).mean().item() #* 1000
        avh_score_list.append(avh_score)

        # for JavisScore
        M, N = video_windows_indices.shape[:2]
        with torch.no_grad():
            embeddings = model(javis_inputs)
        embed_video = embed_frames[video_windows_indices.flatten()].view(M, N, -1)  # shape(M,N,1024)
        embed_audio = embeddings[ModalityType.AUDIO].unsqueeze(1)    # shape(M,1,1024)

        javis_score_clip = cos(embed_video, embed_audio)  # shape(M,N)
        k = topk_min if isinstance(topk_min, int) else int(N * topk_min)
        topk_values, _ = torch.topk(javis_score_clip, k, dim=1, largest=False, sorted=False)
        javis_score_window = topk_values.mean(dim=1)
        javis_score = javis_score_window.mean(dim=0).item()

        # javis_score_clip = cos(embed_video, embed_audio).mean(dim=1)  # shape(M)
        # k = topk_min if isinstance(topk_min, int) else math.ceil(M * topk_min)
        # topk_values, _ = torch.topk(javis_score_clip, k, dim=0, largest=False, sorted=False)
        # javis_score = topk_values.mean().item()
        
        javis_score_list.append(javis_score)

        index_list.append(index[0])
    
    indices = torch.tensor(index_list).argsort()
    avh_scores = torch.tensor(avh_score_list)[indices]
    javis_scores = torch.tensor(javis_score_list)[indices]

    avh_score, javis_score = avh_scores.mean().item(), javis_scores.mean().item()

    if cat2indices is not None:
        avh_score = {'overall': avh_score, 'all': avh_scores.tolist()}
        javis_score = {'overall': javis_score, 'all': javis_scores.tolist()}
        for ai, index_list in enumerate(cat2indices):
            avh_score[ai], javis_score[ai] = [], []
            for ci, indices in enumerate(index_list):
                avh_score[ai].append(torch.mean(avh_scores[indices]).item())
                javis_score[ai].append(torch.mean(javis_scores[indices]).item())

    if return_score_list:
        return avh_score, javis_score, avh_scores, javis_scores
    else:
        return avh_score, javis_score


def calc_audio_score(gt_audio_list, pred_audio_list, prompt_list, device='cuda:0', 
                     eval_num=None, bs=8, **kwargs):
    ############### Part I - FAD ###############
    from .AudioCLIP.get_embedding import preprocess_audio

    audioclip = load_audioclip_pretrained(device)

    real_loader, fake_loader = create_dataloader_for_fvd_vanilla(
        gt_audio_list, gt_audio_list, pred_audio_list, pred_audio_list, 
        audio_sr=kwargs.pop('audio_sr'),
        num_workers=kwargs.pop('num_workers'),
        audio_only=True,
        **kwargs
    )

    loader_dict = {'real': real_loader, 'fake': fake_loader}
    embed_dict = {}
    for t, loader in loader_dict.items():
        audio_embeds = []
        cnt = 0
        for _, sample in enumerate(tqdm(loader, desc=f'fad: {t}')):
            audio_sample = sample['audio'].to(device)

            audios = preprocess_audio(audio_sample).to(device)

            with torch.no_grad():
                audioclip_audio_embed = audioclip(audio=audios, video=None)[0][0][0]
            assert audio_sample.shape[0] == audioclip_audio_embed.shape[0]
            
            audio_embeds.append(audioclip_audio_embed)

            cnt += audio_sample.shape[0]
            if eval_num and cnt >= eval_num: 
                break

        embed_dict[t] = torch.cat(audio_embeds)
    
    fad = frechet_distance(embed_dict['fake'], embed_dict['real']).item() * 10000
    ############### Part I - FAD ###############


    ############### Part II - IB_TA ###############
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)

    text_embeds, audio_embeds = [], []
    # fast enough in a for-loop
    for i in tqdm(range(0, len(pred_audio_list), bs), desc='ib_score'):
        prompts, audios = prompt_list[i:i+bs], pred_audio_list[i:i+bs]
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(prompts, device),
            ModalityType.AUDIO: data.load_and_transform_audio_data(audios, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        text_embeds.append(embeddings[ModalityType.TEXT])
        audio_embeds.append(embeddings[ModalityType.AUDIO])

    text_embeds, audio_embeds = torch.cat(text_embeds), torch.cat(audio_embeds)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sim_ta = cos(text_embeds, audio_embeds).mean().item()
    ############### Part II - IB_TA ###############
    

    ############### Part III - CLAP ###############
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    model.to(device=device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    dataloader = create_dataloader(
        metric='clap-score', 
        audio_path_list=pred_audio_list,
        prompt_list=prompt_list,
        sr=48000,   # CLAP requires sample_rate=48000
        max_audio_len_s=None,
        batch_size=1
    )

    score_list, index_list = [], []
    for audios, prompts, indices in tqdm(dataloader, desc='clapscore'):
        assert len(audios) == len(prompts) == len(indices) == 1
        inputs = processor(text=prompts[0], audios=audios[0].squeeze(), 
                           return_tensors="pt", padding=True, 
                           sampling_rate=48000)  # CLAP requires sample_rate=48000
        inputs.to(device=device)
        outputs = model(**inputs)
        scores = cos(outputs.text_embeds, outputs.audio_embeds).mean()
        score_list.append(scores)
        index_list.append(indices)
    
    indices = torch.cat(index_list).flatten()
    clap_scores = torch.tensor(score_list)[indices.argsort()]

    clap_score = clap_scores.mean().item()
    ############### Part III - CLAP ###############

    return fad, sim_ta, clap_score