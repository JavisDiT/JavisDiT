import os
import re

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.io import write_video
from torchvision.utils import save_image
import soundfile as sf

from . import video_transforms
from .audio_utils import AudioLDM2MelTransform

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
AUD_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".m4a")

regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_img(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in IMG_EXTENSIONS


def is_vid(path):
    ext = os.path.splitext(path)[-1].lower()
    return ext in VID_EXTENSIONS


def is_url(url):
    return re.match(regex, url) is not None


def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path, delimiter=',')
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def download_url(input_path):
    output_dir = "cache"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, base_name)
    img_data = requests.get(input_path).content
    with open(output_path, "wb") as handler:
        handler.write(img_data)
    print(f"URL {input_path} downloaded to {output_path}")
    return output_path


def generate_temporal_window(total_frames, num_frames, frame_interval):
    temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert (
        end_frame_ind - start_frame_ind >= num_frames
    ), f"Not enough frames to sample, {end_frame_ind} - {start_frame_ind} < {num_frames}"
    return start_frame_ind, end_frame_ind


def temporal_random_crop(vframes, num_frames, frame_interval, aframes=None, require_info=False):
    total_frames = len(vframes)
    start_frame_ind, end_frame_ind = generate_temporal_window(total_frames, num_frames, frame_interval)
    
    ## here to skip every `frame_interval` frames to keep the final `num_frames`
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, num_frames, dtype=int)
    video = vframes[frame_indice]
    if aframes is None:
        if not require_info:
            return video
        else:
            v_info = {'raw_video': vframes, 'start_frame_ind': start_frame_ind, 'end_frame_ind': end_frame_ind}
            return video, v_info
    
    total_audio_frames = len(aframes)
    # assert frame_interval == 1, 'Currently unsupport `frame_interval > 1` for audio crop'
    av_scale = total_audio_frames / total_frames
    end_audio_frame_ind = int(round(end_frame_ind * av_scale))
    num_audio_frames = int(round(num_frames * av_scale))
    start_audio_frame_ind = end_audio_frame_ind - num_audio_frames * frame_interval
    ## equal to `np.linspace` when `frame_interval == 1`
    audio_frame_indices = np.linspace(start_audio_frame_ind, end_audio_frame_ind - 1, 
                                      num_audio_frames, dtype=int)
    audio = aframes[audio_frame_indices]
    if not require_info:
        return video, audio
    else:
        va_info = {'raw_video': vframes, 'start_frame_ind': start_frame_ind, 'end_frame_ind': end_frame_ind,
                   'raw_audio': aframes, 'start_audio_frame_ind': start_audio_frame_ind, 'end_audio_frame_ind': end_audio_frame_ind}
        return video, audio, va_info


def temporal_random_crop_v2(vframes, total_frames, frame_interval, start_frame_ind, end_frame_ind, aframes=None, require_info=False):
    video = vframes
    num_frames = end_frame_ind - start_frame_ind
    if aframes is None:
        if not require_info:
            return video
        else:
            v_info = {'start_frame_ind': start_frame_ind, 'end_frame_ind': end_frame_ind}
            return video, v_info
    total_audio_frames = len(aframes)
    # assert frame_interval == 1, 'Currently unsupport `frame_interval > 1` for audio crop'
    av_scale = total_audio_frames / total_frames
    end_audio_frame_ind = int(round(end_frame_ind * av_scale))
    num_audio_frames = int(round(num_frames * av_scale))
    start_audio_frame_ind = end_audio_frame_ind - num_audio_frames * frame_interval
    ## equal to `np.linspace` when `frame_interval == 1`
    audio_frame_indices = np.linspace(start_audio_frame_ind, end_audio_frame_ind - 1, 
                                      num_audio_frames, dtype=int)
    audio = aframes[audio_frame_indices]
    if not require_info:
        return video, audio
    else:
        va_info = {'start_frame_ind': start_frame_ind, 'end_frame_ind': end_frame_ind,
                   'raw_audio': aframes, 'start_audio_frame_ind': start_audio_frame_ind, 'end_audio_frame_ind': end_audio_frame_ind}
        return video, audio, va_info


def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                # video_transforms.RandomHorizontalFlipVideo(),
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def get_transforms_image(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "Image size must be square for center crop"
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size[0])),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: resize_crop_to_fill(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform


def get_transforms_audio(name="mel_spec_audioldm2", cfg={}):
    if name is None:
        return None
    elif name == "mel_spec_audioldm2":
        transform_audio = AudioLDM2MelTransform(cfg)
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_audio


def read_image_from_path(path, transform=None, transform_name="center", num_frames=1, image_size=(256, 256)):
    image = pil_loader(path)
    if transform is None:
        transform = get_transforms_image(image_size=image_size, name=transform_name)
    image = transform(image)
    video = image.unsqueeze(0).repeat(num_frames, 1, 1, 1)
    video = video.permute(1, 0, 2, 3)
    return video


def read_video_from_path(path, transform=None, transform_name="center", image_size=(256, 256)):
    vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
    if transform is None:
        transform = get_transforms_video(image_size=image_size, name=transform_name)
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)
    return video


def read_from_path(path, image_size, transform_name="center"):
    if is_url(path):
        path = download_url(path)
    ext = os.path.splitext(path)[-1].lower()
    if ext.lower() in VID_EXTENSIONS:
        return read_video_from_path(path, image_size=image_size, transform_name=transform_name)
    else:
        assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
        return read_image_from_path(path, image_size=image_size, transform_name=transform_name)


def save_sample(x, save_path=None, fps=8, normalize=True, value_range=(-1, 1), force_video=False, verbose=True,
                audio=None, audio_fps=None, audio_only=False):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
        audio (Tensor): shape [Ta]
    """
    assert x.ndim == 4

    if not force_video and x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    elif audio_only:
        assert audio is not None and audio_fps is not None
        save_path += ".wav"
        sf.write(save_path, audio.cpu().numpy(), audio_fps)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        if audio is not None and len(audio.shape) == 1:
            audio = audio[None].repeat(2, 1)
        write_video(save_path, x, fps=fps, video_codec="h264",
                    audio_array=audio, audio_fps=audio_fps, audio_codec='aac')
    if verbose:
        print(f"Saved to {save_path}")
    return save_path


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resize_crop_to_fill(pil_image, image_size):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), Image.BICUBIC)
        i = int(round((sh - th) / 2.0))
        j = 0
    arr = np.array(image)
    assert i + th <= arr.shape[0] and j + tw <= arr.shape[1]
    return Image.fromarray(arr[i : i + th, j : j + tw])
