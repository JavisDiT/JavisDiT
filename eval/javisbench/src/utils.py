import os
import os.path as osp
from pathlib import Path
from typing import Literal
import subprocess
import importlib
import warnings
from omegaconf import OmegaConf

import librosa
from PIL import Image
import cv2
from decord import VideoReader, cpu
import av
import numpy as np

import torch
import torchvision.transforms as transforms

from sklearn.metrics.pairwise import polynomial_kernel

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
AUD_EXTENSIONS = (".wav", ".mp3", ".flac", ".aac", ".m4a")


##########################################  common  #######################################

def read_video_cv2(video_path, num_frames, mode:Literal['raster', 'linspace']='linspace',
                   frame_transform=None):
    assert osp.exists(video_path)
    cap = cv2.VideoCapture(video_path)
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  unsafe
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    if num_frames is None:
        indices = list(range(frame_count))
    elif mode == 'raster':
        indices = list(range(num_frames))
    elif mode == 'linspace':
        indices = torch.linspace(0, frame_count - 1, num_frames).round().long().tolist()

    frames = []
    for start_frame_ind in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_ind)
        success, frame = cap.read()
        if not success:
            break
        frame = np.ascontiguousarray(frame[..., ::-1])
        if frame_transform:
            frame = frame_transform(frame)
        frames.append(frame)

    return torch.stack(frames) if isinstance(frames[0], torch.Tensor) else np.stack(frames)


def read_video_decord(video_path, num_frames, mode:Literal['raster', 'linspace']='linspace',
                      frame_transform=None):
    vr = VideoReader(video_path)  # , ctx=cpu(0), num_threads=1
    total_frame_num = len(vr)

    if num_frames is None:
        indices = list(range(total_frame_num))
    elif mode == 'raster':
        indices = list(range(num_frames))
    elif mode == 'linspace':
        indices = torch.linspace(0, total_frame_num - 1, num_frames).round().long().tolist()
    
    frames = vr.get_batch(indices).asnumpy()

    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)

    if frame_transform:
        frames = [frame_transform(frame) for frame in frames]

    return torch.stack(frames) if isinstance(frames[0], torch.Tensor) else np.stack(frames)


def read_audio_librosa(audio_path, sr=16000, max_audio_len_s=None, padding=False, audio_transform=None):
    if osp.splitext(audio_path)[-1] in VID_EXTENSIONS:
        warnings.warn(f'Inefficiency in using librosa to read audio from a video file {audio_path}')
    audio, _ = librosa.load(audio_path, sr=sr)
    if len(audio.shape) == 1:
        audio = audio[None]
    audio_len = audio.shape[1]

    if max_audio_len_s:
        max_audio_len = int(max_audio_len_s * sr)   
        if max_audio_len < audio_len:
            audio = audio[:, :max_audio_len]
        elif max_audio_len > audio_len and padding:
            audio = np.pad(audio, ((0, 0), (0, max_audio_len-audio_len)), 'constant', constant_values=0.)

    if audio_transform:
        audio = audio_transform(audio)

    return audio

def read_audio_from_video_pyav(video_path, sr=16000):
    container = av.open(video_path)
    audio_stream = container.streams.audio[0]

    if sr is None or audio_stream.rate == sr:
        audio_frames = []
        for frame in container.decode(audio_stream):
            audio_frames.append(frame.to_ndarray())
        assert len(audio_frames)

        audio_data = np.concatenate(audio_frames, axis=1)
        sr = audio_stream.rate
    else:
        resampler = av.AudioResampler(
            format='s16', 
            layout=audio_stream.layout.name, 
            rate=sr
        )
        resampled_frames = []
        for frame in container.decode(audio_stream):
            resampled_frames.extend(resampler.resample(frame))
            
        resampled_frames.extend(resampler.resample(None))
        
        audio_data = np.concatenate([frame.to_ndarray() for frame in resampled_frames], axis=1)
    
    audio_data = audio_data.reshape((-1, audio_stream.channels))
    if np.issubdtype(audio_data.dtype, np.integer):  # Need normalization
        max_val = np.iinfo(audio_data.dtype).max + 1
        audio_data = audio_data.astype(np.float32) / max_val

    container.close()

    ainfo = {'audio_fps': float(sr)}

    return audio_data[:, 0], ainfo


VIDEO_READER_BACKENDS = {
    "decord": read_video_decord,
    "cv2": read_video_cv2,
}
FORCE_VIDEO_READER = os.getenv("FORCE_VIDEO_READER", None)


def read_video(*args, **kwargs):
    backend = FORCE_VIDEO_READER or kwargs.pop('backend', 'cv2')
    return VIDEO_READER_BACKENDS[backend](*args, **kwargs)


AUDIO_READER_BACKENDS = {
    "librosa": read_audio_librosa,
    'av': read_audio_from_video_pyav
}
FORCE_AUDIO_READER = os.getenv("FORCE_AUDIO_READER", None)


def read_audio(*args, **kwargs):
    backend = FORCE_AUDIO_READER or kwargs.pop('backend', 'librosa')
    if osp.splitext(args[0])[-1] not in AUD_EXTENSIONS:
        backend = 'av'
    return AUDIO_READER_BACKENDS[backend](*args, **kwargs)


##########################################  common  #######################################

######################################## fvd, kvd, fad ####################################

class ResizeAndPad:
    def __init__(self, target_size=256):
        self.target_size = target_size

    def __call__(self, img):
        _, w, h = img.shape
        if w > h:
            new_w = self.target_size
            new_h = int(self.target_size * h / w)
        else:
            new_h = self.target_size
            new_w = int(self.target_size * w / h)
        
        img = transforms.Resize((new_w, new_h))(img)

        # Step 2: Calculate padding to make image square
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        pad_right = self.target_size - new_w - pad_left
        pad_bottom = self.target_size - new_h - pad_top

        # Step 3: Apply padding
        padding = (pad_top, pad_left, pad_bottom, pad_right)
        img = transforms.Pad(padding, fill=0)(img)  # Fill with black (0)

        return img


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]

    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)

    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)

    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    return mmd

######################################## fvd, kvd, fad ####################################

########################################  CAVPScore  ######################################

"""
source code from https://github.com/SonyResearch/SVG_baseline/blob/main/py_scripts/evaluation/demo_util.py
"""


def which_ffmpeg() -> str:
    '''Determines the path to ffmpeg library

    Returns:
        str -- path to the library
    '''
    result = subprocess.run(['which', 'ffmpeg'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ffmpeg_path = result.stdout.decode('utf-8').replace('\n', '')
    return ffmpeg_path


def reencode_video_with_diff_fps(video_path: str, tmp_path: str, extraction_fps: int, start_second, truncate_second) -> str:
    '''Reencodes the video given the path and saves it to the tmp_path folder.

    Args:
        video_path (str): original video
        tmp_path (str): the folder where tmp files are stored (will be appended with a proper filename).
        extraction_fps (int): target fps value

    Returns:
        str: The path where the tmp file is stored. To be used to load the video from
    '''
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    # assert video_path.endswith('.mp4'), 'The file does not end with .mp4. Comment this if expected'
    # create tmp dir if doesn't exist
    os.makedirs(tmp_path, exist_ok=True)

    # form the path to tmp directory
    if truncate_second is None:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}.mp4')
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    else:
        new_path = os.path.join(tmp_path, f'{Path(video_path).stem}_new_fps_{str(extraction_fps)}_truncate_{start_second}_{truncate_second}.mp4')
        cmd = f'{which_ffmpeg()} -hide_banner -loglevel panic '
        cmd += f'-y -ss {start_second} -t {truncate_second} -i {video_path} -an -filter:v fps=fps={extraction_fps} {new_path}'
        subprocess.call(cmd.split())
    return new_path


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class Extract_CAVP_Features(torch.nn.Module):

    def __init__(self, fps=4, batch_size=2, device=None, tmp_path="./", video_shape=(224,224), config_path=None, ckpt_path=None):
        super(Extract_CAVP_Features, self).__init__()
        self.fps = fps
        self.batch_size = batch_size
        self.device = device
        self.tmp_path = tmp_path

        # Initalize Stage1 CAVP model:
        print("Initalize Stage1 CAVP Model")
        config = OmegaConf.load(config_path)
        self.stage1_model = instantiate_from_config(config.model).to(device)

        # Loading Model from:
        assert ckpt_path is not None
        print("Loading Stage1 CAVP Model from: {}".format(ckpt_path))
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
        
        # Transform:
        self.img_transform = transforms.Compose([
            transforms.Resize(video_shape),
            transforms.ToTensor(),
        ])
    
    
    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.stage1_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    
    @torch.no_grad()
    def forward(self, video_path, start_second=None, truncate_second=None, tmp_path="./tmp_folder"):
        self.tmp_path = tmp_path
        
        # print("video_path", video_path)
        # print("truncate second: ", truncate_second)
        # Load the video, change fps:
        video_path_low_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, self.fps, start_second, truncate_second)
        video_path_high_fps = reencode_video_with_diff_fps(video_path, self.tmp_path, 21.5, start_second, truncate_second)
        
        # read the video:
        cap = cv2.VideoCapture(video_path_low_fps)

        feat_batch_list = []
        video_feats = []
        first_frame = True
        # pbar = tqdm(cap.get(7))
        i = 0
        while cap.isOpened():
            i += 1
            # pbar.set_description("Processing Frames: {} Total: {}".format(i, cap.get(7)))
            frames_exists, rgb = cap.read()
            
            if first_frame:
                if not frames_exists:
                    continue
            first_frame = False

            if frames_exists:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb_tensor = self.img_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
                feat_batch_list.append(rgb_tensor)      # 32 x 3 x 224 x 224
                
                # Forward:
                if len(feat_batch_list) == self.batch_size:
                    # Stage1 Model:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                    feat_batch_list = []
            else:
                if len(feat_batch_list) != 0:
                    input_feats = torch.cat(feat_batch_list,0).unsqueeze(0).to(self.device)
                    contrastive_video_feats = self.stage1_model.encode_video(input_feats, normalize=True, pool=False)
                    video_feats.extend(contrastive_video_feats.detach().cpu().numpy())
                cap.release()
                break
        
        video_contrastive_feats = np.concatenate(video_feats)
        return video_contrastive_feats, video_path_high_fps


########################################  CAVPScore  ######################################
