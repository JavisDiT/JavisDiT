from typing import List, Dict, Literal, Optional, Tuple, Union
from collections import OrderedDict

import os
import os.path as osp
from glob import glob
from time import time
import json
import math
import cv2
import numpy as np
import random

import torch
import torchaudio
from torchvision.transforms.functional import resize

import audioflux as af
import soundfile as sf
import bisect
import librosa

from javisdit.datasets.read_audio import read_audio


class VASpatioTemporalAugmentor(object):
    def __init__(
        self, 
        aug_num: Optional[int] = None,
        video_augmentation_pool: str = './data/st_prior/video/SA-V',  # TODO: multi-source
        audio_augmentation_pool: str = './data/st_prior/audio/TAVGBench',
        aug_once: bool = False
    ):
        self.aug_num = aug_num
        self.aug_once = aug_once
        if not aug_num:
            return 
        self.audio_meta_path = audio_augmentation_pool

        # TODO: set probs
        self.aug_video_spatial_type = OrderedDict({
            # 'mirror_reflection': video_mirror_reflection, # 左右镜像
            'random_mask': video_random_mask, # 随机做 grid mask
            # 'copy_paste': video_copy_paste, # 随机从图像中抠图然后复制到其他区域
            'add_masklet': video_add_masklet, # 随机在视频中增加一段轨迹，注意轨迹可能比较短，不知道会不会影响比较大
        })

        self.aug_video_temporal_type = OrderedDict({
            'temporal_shift': temporal_shift_video, # 视频轴平移
            'video_stall': video_stall, # 视频在随机帧暂停一段时间
        })

        self.aug_audio_spatial_type = OrderedDict({
            "add_sound_source": add_sound_source, # 随机在音轨中加入一段其他音轨
            "del_sound_source": del_sound_source, # 随机/有目的的在音轨中删除一段声源
            "scale_volume": scale_volume, # 将声音音量做 sin/cos/linear up/linear down 变换
            # "change_frequency": shift_audio_spectrum # 平移频谱，改变声音频率
        })

        self.aug_audio_temporal_type = OrderedDict({
            "temporal_shift": temporal_shift,
            "audio_stall": insert_mute_segment,
            "audio_repeat": insert_repeat_segment,
            "scale_speed": scale_speed,
        })

        self.video_augmentation_pool = self.get_aug_pool(video_augmentation_pool, '*.json')
        self.video_augmentation_lengths = self.masklet_video_length()
        sorted_index = np.argsort(self.video_augmentation_lengths)
        self.video_augmentation_pool = [self.video_augmentation_pool[i] for i in sorted_index]
        self.video_augmentation_lengths = [self.video_augmentation_lengths[i] for i in sorted_index]
        self.audio_augmentation_pool = self.get_aug_pool(audio_augmentation_pool, '*.wav')

    def masklet_video_length(self):
        length = []
        for video_augmentation in self.video_augmentation_pool:
            with open(video_augmentation, 'r') as f:
                data = json.load(f)
            visible = data['visible']
            valid_index = []
            continue_false = 0
            for i in range(len(visible)):
                if visible[i]:
                    continue_false = 0
                    valid_index.append(i)
                else:
                    if continue_false >= 10: # 最多 10 帧不在画面中，否则跳过这些帧
                        continue
                    else:
                        valid_index.append(i)
                        continue_false += 1
            length.append(len(valid_index))
        return length
    
    def audio_augmentation_length(self):
        pass

    @staticmethod
    def get_aug_pool(root, pattern="*.*"):
        pool_txt = osp.join(osp.dirname(root), 'pool_list.txt')
        if osp.exists(pool_txt):
            with open(pool_txt, 'r') as f:
                pool = f.read().splitlines()
        else:
            pool = sorted(glob(osp.join(root, pattern)))
            with open(pool_txt, 'w+') as f:
                f.writelines([line + '\n' for line in pool])
        return pool

    def __call__(self, video: torch.Tensor, audio: torch.Tensor, va_info: dict, **aug_kwargs):
        if not self.aug_num:
            return None, None
        
        time_tic = [(None, time())]

        aug_data: Dict[str, Dict[str, List]] = {
            'videos': {'spatial': [], 'temporal': []},
            'audios': {'spatial': [], 'temporal': []}
        }
        aug_cfg: Dict[str, Dict[str, Tuple]] = {
            'videos': {
                'spatial': [self.aug_video_spatial, list(self.aug_video_spatial_type.keys())], 
                'temporal': [self.aug_video_temporal, list(self.aug_video_temporal_type.keys())]
            },
            'audios': {
                'spatial': [self.aug_audio_spatial, list(self.aug_audio_spatial_type.keys())], 
                'temporal': [self.aug_audio_temporal, list(self.aug_audio_temporal_type.keys())]
            }
        }

        if va_info.get("unpaired_audio_path"):
            unpaired_audio, ua_info = read_audio(va_info['unpaired_audio_path'])
            assert ua_info['audio_fps'] == va_info['audio_fps']
            unpaired_audio = unpaired_audio[:len(audio)]
            pad = torch.zeros_like(unpaired_audio[:len(audio)-len(unpaired_audio)])
            unpaired_audio = torch.cat((unpaired_audio, pad), dim=0)
        else:
            unpaired_audio = None
        use_unpaired_audio = unpaired_audio is not None

        audio_meta = get_audio_meta(va_info['audio_path'], self.audio_meta_path)
        
        if self.aug_once:
            modality = aug_kwargs.get("aug_modality", random.choice(list(aug_cfg.keys())))
            prior_type = aug_kwargs.get("aug_prior_type", random.choice(list(aug_cfg[modality].keys())))
            aug_func, aug_types = aug_cfg[modality][prior_type]
            aug_type = random.choice(aug_types)
            if modality == 'videos':
                aug_res = aug_func(video, va_info, aug_type)
            elif modality == 'audios':
                aug_res = aug_func(audio, va_info, audio_meta, aug_type)
            else:
                raise NotImplementedError(modality)
            return modality, prior_type, aug_res

        for _ in range(self.aug_num):
            for modality in ['videos', 'audios']:
                for prior_type in ['spatial', 'temporal']:
                    aug_func, aug_types = aug_cfg[modality][prior_type]
                    aug_type = random.choice(aug_types)
                    if modality == 'videos':
                        aug_res = aug_func(video, va_info, aug_type)
                    elif modality == 'audios':
                        if use_unpaired_audio and random.random() > 0.5:
                            aug_res = unpaired_audio
                            use_unpaired_audio = False  # use only once
                        else:
                            aug_res = aug_func(audio, va_info, audio_meta, aug_type)
                    else:
                        raise NotImplementedError(modality)
                    
                    aug_data[modality][prior_type].append(aug_res)
                    time_tic.append((f'{modality}_{prior_type}_{aug_type}', time()))

        # for i in range(1, len(time_tic)):
        #     op, t = time_tic[i][0], time_tic[i][1] - time_tic[i-1][1]
        #     if t > 2.:
        #         print(f'{op:35s} costs {t:.2f} seconds to augment one sample.')

        return aug_data['videos'], aug_data['audios']

    def aug_video_spatial(self, video: torch.Tensor, va_info: dict, aug_func=None):
        # video shape: [T, C, H, W]. uint8, C=3, RGB, compatible with cv2
        # Video Spatial Augmentation, add other objects, mirror_reflection, etc.
        T, _, _, _ = video.shape
        if not aug_func:
            aug_func = random.choice(list(self.aug_video_spatial_type.keys()))
        
        if aug_func == 'mirror_reflection': 
            return video_mirror_reflection(video)
        elif aug_func == 'add_masklet':
            masklet_length = random.randint(max(12, T // 10), T) # 至少 12 帧 (0.5s), 低于11会出现异常
            try:
                aug_video = video_add_masklet(
                    video, 
                    self.video_augmentation_pool, 
                    self.video_augmentation_lengths, 
                    masklet_length
                )
            except Exception as e: # in case the add_masklet function has critical error
                print('video_add_masklet_error', e)
                aug_video = video_random_mask(video, mask_ratio=(0.2, 0.8))
            return aug_video
        elif aug_func == 'random_mask':
            return video_random_mask(video, mask_ratio=(0.2, 0.8))
        elif aug_func == 'copy_paste':
            return video_copy_paste(video, copy_ratio=(0.2, 0.8))
        else:
            raise NotImplementedError(f'Video Spatial Augmentation {aug_func} not implemented')
        
    def aug_video_temporal(self, video: torch.Tensor, va_info: dict, aug_func=None):
        # video shape: [T, C, H, W]
        T, C, H, W = video.shape
        if not aug_func:
            aug_func = random.choice(list(self.aug_video_temporal_type.keys()))

        if aug_func == 'temporal_shift':    
            return temporal_shift_video(video)
        elif aug_func == 'video_stall':
            duration = random.randint(12, T) # 至少暂停 12 帧
            start_frame = random.randint(0, T - duration)
            return video_stall(video, start_frame, duration)
        else:
            raise NotImplementedError(f'Video Temporal Augmentation {aug_func} not implemented')

    def aug_audio_spatial(self, audio: torch.Tensor, va_info: dict, audio_meta, aug_func=None):
        # audio shape: [S]. float(-1,1)
        s, = audio.shape
        sr = va_info['audio_fps']

        target_transform_duration_in_seconds = random.randint(s // 3, s - 10) / sr # 负样本至少需要变换 L/3, 不然有点太短了，-10 是为了后面的区间计算不会超出长度
        try:
            start_idx, all_silence = generate_random_start_and_detect_silence(audio, sr, target_transform_duration_in_seconds)
        except Exception as e:
            print(f"{va_info['audio_path']} generate_random_start_and_detect_silence_error", e)
            start_idx, all_silence = 0, False

        if not aug_func:
            aug_func = random.choice(list(self.aug_audio_spatial_type.keys()))
        
        if all_silence:
            aug_func = 'add_sound_source' # 纯静音片段只有做加法才是有效的，其他的增强都没用

        audio_sep_path = self.get_audio_sep_path(va_info, target_transform_duration_in_seconds, start_idx)

        if aug_func == "add_sound_source":
            return add_sound_source(audio, self.audio_augmentation_pool, va_info, target_transform_duration_in_seconds)
        elif aug_func == "del_sound_source":
            # return del_sound_source(audio, va_info, start_idx, target_transform_duration_in_seconds, audio_sep_path)
            try:
                aug_audio = del_sound_source(audio, va_info, start_idx, target_transform_duration_in_seconds, audio_sep_path)
            except Exception as e:
                print(f"{va_info['audio_path']} del_sound_source_error", e)
                aug_audio = del_sound_source(audio, va_info, start_idx, target_transform_duration_in_seconds, None)
            return aug_audio
        elif aug_func == "scale_volume":
            return scale_volume(audio, start_idx, target_transform_duration_in_seconds)
        elif aug_func == 'change_frequency':
            try:
                aug_audio = shift_audio_spectrum(audio, va_info) # 25 ms
            except Exception as e:
                print(f"{va_info['audio_path']} shift_audio_spectrum_error", e)
                aug_audio = del_sound_source(audio, va_info, start_idx, target_transform_duration_in_seconds, None) # 替换一下变换，保证有个输出
            return aug_audio 
        else:
            raise NotImplementedError(f'Audio Spatial Augmentation {aug_func} not implemented')

    def aug_audio_temporal(self, audio: torch.Tensor, va_info: dict, audio_meta, aug_func=None):
        # audio shape: [S].
        s, = audio.shape
        sr = va_info['audio_fps']

        if not aug_func:
            aug_func = random.choice(list(self.aug_audio_temporal_type.keys()))
        
        if aug_func not in ['audio_repeat', 'audio_stall'] :
            target_transform_duration_in_seconds = random.randint(s // 3, s - 10) / sr # 负样本至少需要改变 L / 3, -10 是为了后面的区间计算不会超出长度
        else:
            target_transform_duration_in_seconds = random.randint(s // 5, s // 2) / sr # repeat 的时间不能太长了，不然就体现不出来在repeat了

        try:
            start_idx, all_silence = generate_random_start_and_detect_silence(audio, sr, target_transform_duration_in_seconds)
        except Exception as e:
            print(f"{va_info['audio_path']} generate_random_start_and_detect_silence_error", e)
            start_idx, all_silence = 0, False

        if all_silence:
            aug_func = 'temporal_shift' # 纯静音片段只有做时间平移才是可能有效的，其他的增强都没用

        if aug_func == 'temporal_shift':
            return temporal_shift(audio, va_info['raw_audio'], 
                                va_info['start_audio_frame_ind'], va_info['end_audio_frame_ind'])
        elif aug_func == 'audio_stall':
            return insert_mute_segment(audio, start_idx, target_transform_duration_in_seconds, va_info['audio_fps'])
        elif  aug_func == 'audio_repeat':
            return insert_repeat_segment(audio, start_idx, target_transform_duration_in_seconds, va_info['audio_fps']) 
        elif aug_func == 'scale_speed':
            ratio = 0.5 if random.uniform(0, 1) < 0.5 else 2
            return scale_speed(audio, start_idx, target_transform_duration_in_seconds, va_info, ratio)
        else:
            raise NotImplementedError(f'Audio Temporal Augmentation {aug_func} not implemented')

    def get_audio_sep_path(self, va_info, target_del_duration_in_seconds, start_idx):
        target_del_duration_in_frames = int(target_del_duration_in_seconds * va_info['audio_fps'])
        audio_name = osp.basename(va_info['audio_path'])
        audio_meta_path = osp.join(self.audio_meta_path, audio_name + '.json')
        start_audio_frame_ind = va_info['start_audio_frame_ind']
        end_audio_frame_ind = va_info['end_audio_frame_ind']
        if not osp.exists(audio_meta_path): # 没有分割结果
            return None
        
        with open(audio_meta_path, 'r') as f:
            meta = json.load(f)
        
        if len(meta) == 0: # 其实也是没有分割结果
            return None 
        
        valid_seqs = []
        for name, seqs in meta.items():
            for seq in seqs:
                del_start = start_audio_frame_ind + start_idx
                del_end = start_audio_frame_ind + start_idx + target_del_duration_in_frames
                if seq[0] <= del_start and seq[1] >= del_end:
                    valid_seqs.append(name)
                    break
        # 从所有有效的文件里挑选一个出来做增即可
        if len(valid_seqs) > 0:
            selected_name = random.choice(valid_seqs)
            audio_sep_name = audio_name + '__sep__' + selected_name[-7:] + '_' + selected_name[:-8] + '.wav'
            audio_sep_path = osp.join(self.audio_meta_path, audio_sep_name)
            if osp.exists(audio_sep_path):
                return audio_sep_path
            else:
                return None
        else:
            return None

def get_audio_meta(audio_path, meta_path):
    # 找到相关的分割文件
    meta = None
    audio_name = osp.basename(audio_path)
    meta_path = osp.join(meta_path, audio_name + '.json')
    if osp.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if len(meta) == 0:
            meta = None
        else:
            meta_ret = {}
            for key, value in meta.items():
                if len(value) > 0:
                    meta_ret[key] = value
            if len(meta_ret) == 0:
                meta = None
            else:
                meta = meta_ret
    else:
        meta = None
    return meta


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


def generate_random_start_and_detect_silence(audio, sr, target_transform_durations):
    s, = audio.shape
    # 静音检测
    rms_results = rms(audio.numpy(), 10) # 分成10段做增强
    silence_idx = np.where(np.array(rms_results) < 0.1)[0]
    non_silence_idx = np.setdiff1d(np.arange(len(rms_results)), silence_idx)
    non_silence_segments = []
    for idx in non_silence_idx:
        non_silence_segments.append([idx * (s // 10), (idx + 1) * (s // 10)])
    non_silence_segments = merge_intervals_no_overlap(non_silence_segments)

    target_transform_length = int(target_transform_durations * sr)
    valid_segments = [d[1] - d[0] >= target_transform_length for d in non_silence_segments]

    all_silence = False
    if len(non_silence_idx) == 0: # 所有的片段都是静音的，那么再随机生成一段
        start_idx = random.randint(0, max(0, s - target_transform_length - 1)) 
        all_silence = True
    elif len(non_silence_idx) != 0 and sum(valid_segments) == 0: # 有的片段都是非静音的，但是长度不够，那么随机选择一段并补充进来：
        seq = random.choice(non_silence_segments)
        min_start = max(0, seq[1] - target_transform_length)
        max_start = min(s - target_transform_length, seq[0])
        if min_start <= max_start:
            start_idx = random.randint(min_start, max_start)
        else:
            start_idx = 0
    else: # 非静音片段比较长，选择那个非静音片段即可
        idx = random.choice(np.nonzero(np.array(valid_segments))[0])
        if non_silence_segments[idx][0] <= non_silence_segments[idx][1] - target_transform_length:
            start_idx = random.randint(non_silence_segments[idx][0], non_silence_segments[idx][1] - target_transform_length)
        else:
            start_idx = 0
    return start_idx, all_silence


def temporal_shift(data, raw_data, start_frame_ind, end_frame_ind):
    total_frames = len(raw_data)
    num_frames = end_frame_ind - start_frame_ind
    assert num_frames == len(data)
    while True:
        rand_start_frame_ind = np.random.randint(0, total_frames-end_frame_ind)
        if rand_start_frame_ind != start_frame_ind:
            rand_end_frame_ind = rand_start_frame_ind + num_frames
            break
    return raw_data[rand_start_frame_ind:rand_end_frame_ind]


def temporal_shift_video(video):
    T, C, H, W = video.shape
    rand_start_frame_ind = np.random.randint(T // 10, T // 2)
    return torch.cat([video[rand_start_frame_ind:], video[:rand_start_frame_ind]])


def video_mirror_reflection(video: torch.Tensor) -> torch.Tensor:
    """
    Perform a mirror reflection (flip) along the width (horizontal axis) of the video.

    Args:
        video (torch.Tensor): Input video tensor of shape [T, C, H, W].

    Returns:
        torch.Tensor: Video tensor with mirrored reflection along the width, shape [T, C, H, W].
    """
    return video.flip(3).contiguous()


def video_random_mask(
    video: torch.Tensor, 
    mask_ratio: Union[float, Tuple[float, float]] = 0.5,
    padding_value: int = 0,
    grid_size: Union[int, str] = 'auto',
):
    """
    Args:
        video (torch.Tensor): Input video tensor of shape [T, C, H, W].
        mask_ratio (float or Tuple[float, float]): Mask ratio or a tuple for the range of the mask ratio.
        padding_value: (int): Value to use for padding the masked regions.
        grid_size (int or auto): Grid size for the mask, 'auto' for min(H, W) // 6
        
    Returns:
        torch.Tensor: Video tensor with random masks applied of shape [T, C, H, W].
    """
    # faster
    T, C, H, W = video.shape

    if isinstance(mask_ratio, tuple):
        mask_ratio = random.uniform(mask_ratio[0], mask_ratio[1])
    
    if grid_size == 'auto':
        grid_size = min(H, W) // 6

    grids_h, grids_w = H // grid_size, W // grid_size

    total_grids = grids_h * grids_w
    copy_count = int(total_grids * mask_ratio)

    target_coords_h = torch.randint(0, grids_h, (copy_count,)) * grid_size
    target_coords_w = torch.randint(0, grids_w, (copy_count,)) * grid_size

    aug_video = video.clone()

    for i in range(copy_count):
        h_start = target_coords_h[i]
        w_start = target_coords_w[i]
        aug_video[:, :, h_start:h_start + grid_size, w_start:w_start + grid_size] = padding_value

    return aug_video


def video_copy_paste(
    video: torch.Tensor, 
    copy_ratio: Union[float, Tuple[float, float]] = 0.5,
    grid_size: Union[int, str] = 'auto',   
):
    """
    copy some grids from the video and paste them to random positions.
    Args:
        video (torch.Tensor): Input video tensor of shape [T, C, H, W].
        copy_ratio (float or Tuple[float, float]): Copy ratio or a tuple for the range of the mask ratio.
        grid_size (int or 'auto'): Grid size for the mask. 'auto' for min(H, W) // 6
        
    Returns:
        torch.Tensor: Video tensor with random masks applied of shape [T, C, H, W].
    """ 
    T, C, H, W = video.shape

    if isinstance(copy_ratio, tuple):
        copy_ratio = random.uniform(copy_ratio[0], copy_ratio[1])
    
    if grid_size == 'auto':
        grid_size = min(H, W) // 6

    grids_h, grids_w = H // grid_size, W // grid_size

    total_grids = grids_h * grids_w
    copy_count = int(total_grids * copy_ratio)

    grid_coords = torch.tensor([
        (i * grid_size, j * grid_size) 
        for i in range(grids_h) for j in range(grids_w)
    ])

    selected_idx = torch.randperm(total_grids)[:copy_count]
    selected_grids = grid_coords[selected_idx]

    target_coords_h = torch.randint(0, grids_h, (copy_count,)) * grid_size
    target_coords_w = torch.randint(0, grids_w, (copy_count,)) * grid_size

    selected_grids_tensor = torch.stack([
        video[:, :, selected_grids[i, 0]:selected_grids[i, 0] + grid_size, selected_grids[i, 1]:selected_grids[i, 1] + grid_size]
        for i in range(copy_count)
    ])

    aug_video = video.clone()

    for i in range(copy_count):
        h_start = target_coords_h[i]
        w_start = target_coords_w[i]
        # Paste the selected grid at the target position
        aug_video[:, :, h_start:h_start + grid_size, w_start:w_start + grid_size] = selected_grids_tensor[i]

    return aug_video


def xywh2xyxy(bbox):
    return torch.cat([bbox[:, :2], bbox[:, :2] + bbox[:, 2:]], dim=-1)


def video_add_masklet(
    video: torch.Tensor,
    video_augmentation_pool: List[str],
    video_augmentation_lengths: List[int],
    aug_masklet_length: int, 
    better: bool = True
):
    # video: TCHW
    # video_augmentation_pool: list of paths to the augmentation videos
    # masklet_length: in frames
    # better: whether to use the better masklet
    T, C, H, W = video.shape
    assert aug_masklet_length <= T, "masklet length should be less than or equal to the video length"

    # 随机选择一个满足大于 masklet_length 的 masklet
    start_index = bisect.bisect_right(video_augmentation_lengths, aug_masklet_length)
    if start_index >= len(video_augmentation_pool):
        start_index = len(video_augmentation_pool) - 1
    masklet_index = random.randint(start_index, len(video_augmentation_pool) - 1)
    
    # 读取 masklet
    masklet_info_path = video_augmentation_pool[masklet_index]
    with open(masklet_info_path, 'r') as f:
        masklet_info = json.load(f)
    masklet_video_h, masklet_video_w = masklet_info['video_frame_height'], masklet_info['video_frame_width']
    visible = np.array(masklet_info['visible'])
    crop_bbox_xywh = torch.tensor(masklet_info['crop_bbox'])

    # 读取出来 valid_index
    valid_index = []
    continue_false = 0
    for i in range(len(visible)):
        if visible[i]:
            continue_false = 0
            valid_index.append(i)
        else:
            if continue_false >= 10: # 最多 10 帧不在画面中，否则跳过这些帧
                continue
            else:
                valid_index.append(i)
                continue_false += 1

    # 选择 masklet 的开始区间
    masklet_start_index = random.randint(0, max(0, len(valid_index) - aug_masklet_length))
    masklet_end_index = masklet_start_index + aug_masklet_length
    valid_index = np.array(valid_index)[masklet_start_index:masklet_end_index]
    valid_crop_bbox_xywh = crop_bbox_xywh[valid_index]
    valid_crop_bbox_xyxy = xywh2xyxy(valid_crop_bbox_xywh)

    # 计算轨迹的活动范围
    active_region_x1 = valid_crop_bbox_xyxy[:, 0].min()
    active_region_y1 = valid_crop_bbox_xyxy[:, 1].min()
    active_region_x2 = valid_crop_bbox_xyxy[:, 2].max()
    active_region_y2 = valid_crop_bbox_xyxy[:, 3].max()
    active_region_w = active_region_x2 - active_region_x1
    active_region_h = active_region_y2 - active_region_y1
    # 将轨迹范围的顶点归零
    valid_crop_bbox_xyxy[:, 0] -= active_region_x1
    valid_crop_bbox_xyxy[:, 1] -= active_region_y1
    valid_crop_bbox_xyxy[:, 2] -= active_region_x1
    valid_crop_bbox_xyxy[:, 3] -= active_region_y1

    # 计算缩放比例
    max_ratio = min((W - 1) / active_region_w, (H - 1) / active_region_h) # 至少要放的进原视频
    min_ratio = min((W - 1) / masklet_video_w, (H - 1) / masklet_video_h) # 肯定能放的进原视频
    if active_region_w == 0 or active_region_h == 0:
        min_ratio = max_ratio = 1
    ratio = random.uniform(min_ratio, max_ratio)

    valid_crop_bbox_xyxy = valid_crop_bbox_xyxy * ratio
    scaled_active_region_w = math.ceil(active_region_w * ratio)
    scaled_active_region_h = math.ceil(active_region_h * ratio)

    # 随机选择一个位置
    x_start = random.randint(0, max(0, W - scaled_active_region_w - 1))
    y_start = random.randint(0, max(0, H - scaled_active_region_h - 1))
    valid_crop_bbox_xyxy[:, 0] += x_start
    valid_crop_bbox_xyxy[:, 1] += y_start
    valid_crop_bbox_xyxy[:, 2] += x_start
    valid_crop_bbox_xyxy[:, 3] += y_start

    # 随机选择一个开始插入的帧
    start_frame = random.randint(0, T - aug_masklet_length) # T >= aug_masklet_length
    end_frame = min(T, start_frame + aug_masklet_length)
    # 生成新的视频
    aug_video = video.clone()
    
    mask_video_path = masklet_info_path.replace('json', 'mp4').replace('meta', 'mask')
    content_video_path = masklet_info_path.replace('json', 'mp4').replace('meta', 'masklet')
    mask_cap = cv2.VideoCapture(mask_video_path)
    content_cap = cv2.VideoCapture(content_video_path)
    last_frame_idx = -1
    mask, content = None, None
    for masklet_idx, origin_video_idx in enumerate(range(start_frame, end_frame)):
        if not masklet_idx < len(valid_index):
            break
        masklet_frame_index = int(valid_index[masklet_idx])
        if masklet_frame_index != last_frame_idx + 1:
            mask_cap.set(cv2.CAP_PROP_POS_FRAMES, masklet_frame_index)
            content_cap.set(cv2.CAP_PROP_POS_FRAMES, masklet_frame_index)
        last_frame_idx = masklet_frame_index
        ret1, mask = mask_cap.read()
        ret2, content = content_cap.read()
        if not ret1 or not ret2:
            break
        
        if better:
            mask[mask == 255] = 1
            contoures, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = []
            for k in range(len(contoures)):
                area.append(cv2.contourArea(contoures[k]))
            if len(area) > 0:
                max_index = np.argmax(np.array(area))

                mask = torch.from_numpy(cv2.drawContours(np.ascontiguousarray(mask[:, :, 0]), contoures, max_index, (255), cv2.FILLED)).to(torch.bool)[None, ...].repeat(3, 1, 1).contiguous()
            else:
                mask = torch.from_numpy(mask == 1).to(torch.bool).permute(2, 0, 1).contiguous()
        else:
            mask = torch.from_numpy(mask == 255).to(torch.bool).permute(2, 0, 1).contiguous()
        content = torch.from_numpy(content).permute(2, 0, 1)[[2, 1, 0]].contiguous() 
        ## TODO: normarlize
        content = content.to(video.dtype) / 255.0
        content = (content - 0.5) / 0.5
        ##
        bbox = valid_crop_bbox_xyxy[masklet_idx]
        left_x, left_y = int(bbox[0]), int(bbox[1])
        right_x, right_y = int(bbox[2]), int(bbox[3])
        h, w = aug_video[origin_video_idx, :, left_y:right_y, left_x:right_x].shape[-2:]
        if h > 0 and w > 0:
            mask = resize(mask, (h, w))
            content = resize(content, (h, w))
            mask = mask[:, :h, :w]
            content = content[:, :h, :w]
            aug_video[origin_video_idx, :, left_y:right_y, left_x:right_x] = torch.where(mask, content, aug_video[origin_video_idx, :, left_y:right_y, left_x:right_x])    
    mask_cap.release()
    content_cap.release()
    del visible, crop_bbox_xywh, valid_index, valid_crop_bbox_xywh, valid_crop_bbox_xyxy, mask, content
    return aug_video


def video_stall(
    video: torch.Tensor,
    stall_start: int,
    stall_duration: int
):
    """
    Simulate a stall at a specific frame for a duration.

    Args:
        video (torch.Tensor): Input video tensor of shape [T, C, H, W].
        stall_start (int): The frame number to start the stall (0-indexed).
        stall_duration (int): The duration (in frames) for the stall.

    Returns:
        torch.Tensor: Video tensor with the stall applied.
    """    
    T, C, H, W = video.shape

    assert stall_start < T, "Stall start frame must be less than the total number of frames in the video."

    frame_to_repeat = video[stall_start]
    # frame_left = T - stall_duration
    # assert frame_left >= 0, f'{[T, stall_start, stall_duration]}'

    aug_video = torch.cat([
        video[:stall_start],
        frame_to_repeat.unsqueeze(0).repeat(stall_duration, 1, 1, 1),
        video[stall_start+1:]
    ])[:T]

    return aug_video


def scale_audio_ampltitude(audio: torch.Tensor, scale_factor: Union[float, torch.Tensor]):
    ## audio: [S]
    ## scale_factor: float or [S]
    ## output: audio: [S]
    return (scale_factor * audio).clip(-1.0, 1.0)


def scale_audio_sine(audio: torch.Tensor):
    audio_len = len(audio)
    random_scale =random.uniform(0.5, 2)
    scale_factor = random_scale * torch.sin(torch.linspace(0, np.pi, audio_len))
    return scale_audio_ampltitude(audio, scale_factor)


def scale_audio_cosine(audio: torch.Tensor):
    audio_len = len(audio)
    random_scale = random.uniform(0.5, 2)
    scale_factor = random_scale * torch.cos(torch.linspace(0, np.pi, audio_len))
    return scale_audio_ampltitude(audio, scale_factor)


def scale_audio_linear_increasing(audio: torch.Tensor):
    audio_len = len(audio)
    random_scale = random.uniform(0.5, 2)
    scale_factor = random_scale * torch.linspace(0, 1, audio_len)
    return scale_audio_ampltitude(audio, scale_factor)


def scale_audio_linear_decreasing(audio: torch.Tensor):
    audio_len = len(audio)
    random_scale = random.uniform(0.5, 2)
    scale_factor = random_scale * torch.linspace(1, 0, audio_len)
    return scale_audio_ampltitude(audio, scale_factor)


def scale_audio_random(audio: torch.Tensor):
    scale_factor = torch.fron_numpy(np.random.uniform(0.5, 2, len(audio)))
    return scale_audio_ampltitude(audio, scale_factor)


def reverse_audio_trend_without_time_change(audio: torch.Tensor, sample_rate: int):
    """TODO: not satified"""
    import librosa
    hop_length = 512
    envelope = librosa.onset.onset_strength(audio.numpy(), sr=sample_rate, hop_length=hop_length)
    reversed_envelope = np.flip(envelope)
    scaled_audio = np.copy(audio)
    for i in range(len(envelope)):
        scaled_audio[i * hop_length:(i + 1) * hop_length] *= reversed_envelope[i] / np.max(envelope)
    return torch.from_numpy(scaled_audio)


def change_speed(raw_audio: torch.Tensor, audio: torch.Tensor, start_frame_ind):
    from librosa.effects import time_stretch
    speed_factor = random.uniform(0.5, 2)
    origin_len = len(audio)
    if origin_len * speed_factor + start_frame_ind > len(raw_audio):
        speed_factor = random.uniform(0.5, 1.0) # TODO: need to be more careful
    new_end_frame_ind = int(origin_len * speed_factor + start_frame_ind)
    return torch.from_numpy(time_stretch(raw_audio[start_frame_ind:new_end_frame_ind].numpy(), speed_factor))


def scale_speed(audio:torch.Tensor, speed_start:int, duration:float, va_info: dict, factor: float):
    s, = audio.shape
    sample_rate = va_info['audio_fps']
    end_sample = speed_start + int(duration * sample_rate)
    end_sample = min(end_sample, s)
    segment_to_adjust = audio[speed_start:end_sample].numpy()
    time_stretch_obj = af.TimeStretch(radix2_exp=12, window_type=af.type.WindowType.HANN, slide_length=102)
    new_audio_arr = torch.from_numpy(time_stretch_obj.time_stretch(segment_to_adjust, factor))
    if new_audio_arr.shape[0] < end_sample - speed_start:
        new_audio_arr = torch.cat([new_audio_arr, torch.zeros(end_sample - speed_start - new_audio_arr.shape[0])])
    aug_audio = torch.cat([
        audio[:speed_start],
        new_audio_arr,
        audio[end_sample:]
    ])[:s]
    return aug_audio


def add_sound_source(
    audio: torch.Tensor,
    audio_pool_list: List[str],
    va_info: dict,
    target_transform_duration: float,
):
    s, = audio.shape
    original_audio_name = osp.basename(va_info['audio_path'])
    extra_audio_path = random.choice(audio_pool_list) # wav file

    roll_cnt = 0
    while original_audio_name in extra_audio_path:
        extra_audio_path = random.choice(audio_pool_list)  # 这个会不会造成死循环？
        roll_cnt += 1
        if roll_cnt == 100: # 如果100 次都摇不出来，那么就摆烂了
            break
    extra_sound_source, extra_sr = read_audio(extra_audio_path)
    extra_sr = extra_sr['audio_fps']
    target_sr = va_info['audio_fps']
    json_path = extra_audio_path.split('__sep__')[0] + '.json' # 原音频的标注信息，记录了声源发声的区间

    target_transform_length_for_extra = int(target_transform_duration * extra_sr) # in frames

    random_seq = False
    if osp.exists(json_path):
        with open(json_path, 'r') as f:
            annot_data = json.load(f)
        name = extra_audio_path.split('__sep__')[-1]
        sound_key = name[8:-4] + '_' + name[:7]
        if sound_key in annot_data.keys():
            sound_seq = annot_data[sound_key]
            if len(sound_seq) > 0:
                choice_seq = random.choice(sound_seq)
                start_frame_ind = choice_seq[0]
                end_frame_ind = choice_seq[1]
                if end_frame_ind - start_frame_ind >= target_transform_length_for_extra:
                    start_frame_ind = random.randint(start_frame_ind, end_frame_ind - target_transform_length_for_extra)
                    end_frame_ind = start_frame_ind + target_transform_length_for_extra
                else:
                    s2, = extra_sound_source.shape
                    min_start = max(0, end_frame_ind - target_transform_length_for_extra)
                    max_start = min(s2 - target_transform_length_for_extra - 1, start_frame_ind)
                    if min_start < max_start:
                        start_frame_ind = random.randint(min_start, max_start)
                    else:
                        start_frame_ind = min_start
                    end_frame_ind = start_frame_ind + target_transform_length_for_extra
            else:
                random_seq = True # 没有声源，随便选
        else:
            random_seq = True # 没有声源，随便选
    else:
        random_seq = True # 没有分割文件，随便选
    
    if random_seq:
        start_frame_ind, _ = generate_random_start_and_detect_silence(extra_sound_source, extra_sr, target_transform_length_for_extra / extra_sr)
        # 如果是一个全静音的片段，还是不做处理了，比较麻烦
        end_frame_ind = start_frame_ind + target_transform_length_for_extra
    
    extra_sound_source = extra_sound_source[start_frame_ind:end_frame_ind]

    if extra_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=extra_sr, new_freq=target_sr)
        extra_sound_source = resampler(extra_sound_source)

    aug_audio = audio.clone()
    # 随机挑选一个起始点
    start_frame_ind = random.randint(0, max(0, len(audio) - len(extra_sound_source)))

    end_frame_ind = min(start_frame_ind + len(extra_sound_source), s)

    audio_len = end_frame_ind - start_frame_ind
    if audio_len > len(extra_sound_source):
        extra_sound_source = torch.cat([extra_sound_source, torch.zeros(audio_len - len(extra_sound_source))])
    else:
        extra_sound_source = extra_sound_source[:audio_len]
    aug_audio[start_frame_ind:end_frame_ind] += extra_sound_source
    return aug_audio


def del_sound_source(
    audio: torch.Tensor,
    va_info: dict,
    start_ind: int,
    duration: float,
    audio_sep: Optional[str] = None,
):
    s, = audio.shape
    if audio_sep == None:
        end_ind = start_ind + int(duration * va_info['audio_fps'])
        aug_audio = audio.clone()
        aug_audio[start_ind:end_ind] = 0
    else:
        sep_audio, sr2 = read_audio(audio_sep)
        sr2 = sr2['audio_fps']
        ori_audio = va_info['raw_audio']
        length = min(len(sep_audio), len(ori_audio))
        diff_audio = sep_audio[:length] - ori_audio[:length]
        # aug_audio = diff_audio[va_info['start_audio_frame_ind']:va_info['end_audio_frame_ind']]
        del_start = va_info['start_audio_frame_ind'] + start_ind
        del_end = del_start + int(duration * va_info['audio_fps'])
        diff_audio = diff_audio[del_start:del_end]
        if diff_audio.shape[0] < del_end - del_start:
            diff_audio = torch.cat([aug_audio, torch.zeros(del_end - del_start - diff_audio.shape[0])])
        else:
            diff_audio = diff_audio[:del_end - del_start]
        aug_audio = torch.cat([
            audio[:start_ind],
            diff_audio,
            audio[start_ind + int(duration * va_info['audio_fps']):],
        ])
        if aug_audio.shape[0] < s:
            aug_audio = torch.cat([aug_audio, torch.zeros(s - aug_audio.shape[0])])
        else:
            aug_audio = aug_audio[:s]

    return aug_audio


def scale_volume(
    audio: torch.Tensor,
    start_idx: int,
    duration: float,
):
    s, = audio.shape
    end_idx = start_idx + int(duration * s)
    aug_audio = audio.clone()
    scale_functions = [scale_audio_sine, scale_audio_cosine, scale_audio_linear_increasing, scale_audio_linear_decreasing]
    scale_func = random.choice(scale_functions)
    aug_audio[start_idx:end_idx] = scale_func(aug_audio[start_idx:end_idx])
    return aug_audio


def change_audio_frequency_ampltitude(audio: torch.Tensor):
    ## 没什么用
    import librosa
    D = librosa.stft(audio.numpy(), n_fft=1024, hop_length=160, win_length=1024)
    magnitude, phase = librosa.magphase(D)
    magnitude = magnitude * np.random.uniform(0.1, 10)
    enhanced_D = magnitude * phase
    return torch.from_numpy(librosa.istft(enhanced_D))


def shift_audio_spectrum(audio: torch.Tensor, va_info):
    ### 有没有用还不可知
    D = librosa.stft(audio.numpy(), n_fft=1024, hop_length=160, win_length=1024)
    magnitude, phase = librosa.magphase(D)
    freq_axis = librosa.fft_frequencies(sr=va_info['audio_fps'], n_fft=1024)
    shift_amount = random.uniform(-1000, 1000)
    shift_bins = int(shift_amount / (freq_axis[1] - freq_axis[0]))
    shifted_magnitude = np.roll(magnitude, shift=shift_bins, axis=0)
    shift_D = shifted_magnitude * phase
    shifted_audio = librosa.istft(shift_D)    
    stretched_audio = librosa.effects.time_stretch(shifted_audio, rate=1.0 / len(audio) * len(shifted_audio))[:len(audio)]
    return torch.from_numpy(stretched_audio)


def mask_mute_frames(
    audio: torch.Tensor,
    va_info, 
    num_silence_segments, 
    silence_duration_range=(0.1, 0.5),
):
    sr = va_info['audio_fps']
    masked_audio = audio.clone()
    audio_length = len(audio) / sr
    audio_samples = len(audio)

    for _ in range(num_silence_segments):
        start_time = np.random.uniform(0, audio_length - silence_duration_range[1])
        silence_duration = np.random.uniform(*silence_duration_range)
        start_sample = int(start_time * sr)
        end_sample = min(start_sample + int(silence_duration * sr), audio_samples)
        masked_audio[start_sample:end_sample] = 0
    return masked_audio


def insert_mute_segment(audio: torch.Tensor, stall_start: int, mute_duration: float, sample_rate: int):
    # audio: torch.Tensor[S]
    s, = audio.shape
    assert stall_start < s, "stall_start should be less than audio length"
    aug_audio = torch.cat([
        audio[:stall_start],
        torch.zeros(int(mute_duration * sample_rate)),
        audio[stall_start:]
    ])[:s]
    return aug_audio


def insert_repeat_segment(audio:torch.Tensor, repeat_start: int, repeat_duration: float, sample_rate: int):
    # audio: torch.Tensor[S]
    s, = audio.shape
    assert repeat_start < s, "repeat_start should be less than audio length"
    repeat_frame = audio[repeat_start:repeat_start + int(repeat_duration * sample_rate)]
    aug_audio = torch.cat([
        audio[:repeat_start],
        repeat_frame,
        repeat_frame,
        audio[repeat_start:]
    ])[:s]
    return aug_audio
