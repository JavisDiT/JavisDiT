import gc
import math
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
from torchvision.io.video import read_video as read_video_tv
import torchaudio
import soundfile as sf

from .utils import VID_EXTENSIONS, AUD_EXTENSIONS

MAX_NUM_FRAMES = 2500


def read_audio(audio_path, backend: Literal["auto", "torch", "sf"] = "auto") -> Tuple[torch.Tensor, Dict[str, int]]:
    ext = os.path.splitext(audio_path)[-1].lower()
    if backend == 'auto': 
        if ext not in VID_EXTENSIONS and ext in ['.wav', '.aiff', '.flac', '.ogg']:
            backend = 'sf'
        else:
            backend = 'torch'
    # normalized, (-1.0 ~ 1.0)
    if backend == "torch":
        if ext in VID_EXTENSIONS:
            _, aframes, ainfo = read_video_tv(filename=audio_path, pts_unit="sec", output_format="TCHW")
            del ainfo['video_fps']
            ainfo = {'audio_fps': float(ainfo['audio_fps'])}
        elif ext in AUD_EXTENSIONS:
            aframes, fs = torchaudio.load(audio_path)
            ainfo = {'audio_fps': float(fs)}
        else:
            raise ValueError(f"Unsupported audio format: {audio_path}")
        aframes = aframes[0]  # dual track
    elif backend == 'sf':
        if ext not in ['.wav', '.aiff', '.flac', '.ogg']:
            warnings.warn(f'Unsupported audio extension: {ext}')
        aframes, fs = sf.read(audio_path)
        ainfo = {'audio_fps': float(fs)}
        aframes = torch.from_numpy(aframes).to(torch.float32)
        if len(aframes.shape) > 1:  # TODO: check format
            if aframes.shape[1] < 0.1 * fs:
                aframes = aframes[:, 0]
            else:
                aframes = aframes[0, :]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return aframes, ainfo

