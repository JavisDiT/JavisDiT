import os
from glob import glob
from tqdm import tqdm
from time import time

import numpy as np
import torch
from PIL import ImageFile
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from javisdit.registry import DATASETS

from .read_video import read_video
from .read_audio import read_audio
from .utils import (
    VID_EXTENSIONS, AUD_EXTENSIONS,
    get_transforms_image, get_transforms_video, get_transforms_audio,
    read_file, temporal_random_crop, temporal_random_crop_v2
)
from .audio_utils import detect_audio_peaks
from .augment import VASpatioTemporalAugmentor

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


def load_video_audio_transform(
    video_path, direct_load_video_clip, num_frames, frame_interval,
    audio_path, audio_fps, aframes, unpaired_audio_path,
    video_transform, audio_transform, augmenter, require_onset
):
    # loading
    if not direct_load_video_clip:
        vframes, vinfo = read_video(video_path, backend="av")
        total_frames = len(vframes)
    else:
        assert num_frames is not None  # double-check
        vframes, vinfo = read_video(video_path, backend="cv2", 
                                    num_frames=num_frames, frame_interval=frame_interval)
        total_frames = vinfo.pop('total_frames')
        start_frame_ind, end_frame_ind = vinfo.pop('start_frame_ind'), vinfo.pop('end_frame_ind')
    video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

    # align video and audio
    video_duration = total_frames / video_fps
    target_length = int(np.ceil(video_duration*audio_fps))
    audio_length = len(aframes)
    if audio_length > target_length:
        aframes = aframes[:target_length]
    else:
        pad = torch.zeros((target_length-audio_length), dtype=aframes.dtype)
        aframes = torch.cat((aframes, pad))

    # Sampling video frames
    if not direct_load_video_clip:
        video, audio, va_info = \
            temporal_random_crop(vframes, num_frames, frame_interval, aframes, require_info=True)
    else:
        video, audio, va_info = \
            temporal_random_crop_v2(vframes, total_frames, frame_interval, 
                                    start_frame_ind, end_frame_ind, aframes, require_info=True)

    # audio & duration
    video_fps = video_fps // frame_interval
    # audio_fps = audio_fps // frame_interval
    duration = num_frames / video_fps

    # transform
    video = video_transform(video)  # T C H W

    # augment after video transform but before audio transform
    va_info.update({
        'video_path': video_path, 'video_fps': video_fps, 
        'audio_path': audio_path, 'audio_fps': audio_fps, 'unpaired_audio_path': unpaired_audio_path
    })
    if augmenter:
        neg_videos, neg_audios = augmenter(video, audio, va_info)
    else:
        neg_videos, neg_audios = None, None
    raw_audio = audio

    # audio transform
    audio = audio_transform(audio, duration=duration)

    if neg_videos:
        ## {'spatial': Tensor(N, T, C, H, W), 'temporal': Tensor(N, T, C, H, W)}
        neg_videos = {
            # aug_type: torch.stack([transform(aug_video) for aug_video in aug_videos]) \
            aug_type: torch.stack(aug_videos) \
                for aug_type, aug_videos in neg_videos.items()
        }
    if neg_audios: 
        ## {'spatial': Tensor(N, S, M), 'temporal': Tensor(N, S, M)}
        neg_audios = {
            aug_type: torch.stack([audio_transform(aug_audio, duration=duration) for aug_audio in aug_audios]) \
                for aug_type, aug_audios in neg_audios.items()
        }

    if require_onset:
        peaks = detect_audio_peaks(y=raw_audio.numpy(), sr=audio_fps)
        peaks = np.clip((peaks / duration * audio.shape[1]).astype(np.int32), 0, audio.shape[1]-1)
        onset = np.zeros((audio.shape[1], ))
        onset[peaks] = 1.
    else:
        onset = None

    return video_fps, video, audio, onset, neg_videos, neg_audios


@DATASETS.register_module()
class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        elif ext.lower() in AUD_EXTENSIONS:
            return "audio"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]
        return ret

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VideoAudioTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
        direct_load_video_clip=False,
        audio_transform_name=None,
        audio_cfg=None,
        neg_aug=None,
        neg_aug_kwargs={},
        load_data=None,
        require_onset=False,
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name)
        
        self.audio_transform = get_transforms_audio(audio_transform_name, audio_cfg)
        self.augmenter = VASpatioTemporalAugmentor(neg_aug, **neg_aug_kwargs)
        self.neg_aug = neg_aug
        self.require_onset = require_onset

        self.direct_load_video_clip = direct_load_video_clip
        self.buffer_dir = load_data

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        audio_path = sample.get('audio_path', None)
        aframes, ainfo = read_audio(audio_path, backend='auto')
        audio_fps = ainfo.get("audio_fps", 16000)
        assert audio_fps == sample["audio_fps"], audio_fps

        if file_type == "video":
            video_fps, video, audio, onset, neg_videos, neg_audios = load_video_audio_transform(
                path, self.direct_load_video_clip, self.num_frames, self.frame_interval, 
                audio_path, audio_fps, aframes, sample.get("unpaired_audio_path", None),
                self.transforms["video"], self.audio_transform, self.augmenter,
                self.require_onset
            )
        else:
            raise NotImplementedError(file_type)
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {"video": video, "fps": video_fps, "audio": audio, "audio_fps": audio_fps, "num_frames": self.num_frames, }
        if neg_videos:
            # NTCHW -> NCTHW
            for aug_type, aug_video in neg_videos.items():
                neg_videos[aug_type] = aug_video.permute(0, 2, 1, 3, 4)
            ret.update({"neg_videos": neg_videos, "neg_audios": neg_audios})
        if self.require_onset:
            ret.update({"onset": onset, })    # audio peak,

        _, H, W = self.get_data_info(index)
        ret.update({'height': H, 'width': W})
        if self.get_text:
            ret["text"] = sample["text"]
        
        ## load pre-processed data
        if self.buffer_dir is not None:
            buffer_path = os.path.join(self.buffer_dir, f'{index}.bin')
            assert os.path.exists(buffer_path), buffer_path
            ret.update(torch.load(buffer_path, map_location='cpu'))

        return ret

    def __getitem__(self, index):
        # return self.getitem(index)
        try:
            return self.getitem(index)
        except Exception as e:
            print(e)
            return None


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except:
            return None


@DATASETS.register_module()
class VariableVideoAudioTextDataset(VariableVideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        direct_load_video_clip=False,
        audio_transform_name=None,
        audio_cfg=None,
        load_data=None,
        dummy_text_feature=False,
        audio_only=False,
        require_onset=False,
        neg_aug=None,
        neg_aug_kwargs={},
        **kwargs
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, 
                         transform_name=transform_name, dummy_text_feature=dummy_text_feature)

        self.audio_transform = get_transforms_audio(audio_transform_name, audio_cfg)

        self.buffer_dir = load_data
        self.audio_only = audio_only
        self.require_onset = require_onset
        if self.audio_only and self.data.get('audio_text') is not None:
            self.data['text'] = self.data['audio_text']
            del self.data['audio_text']

        self.augmenter = VASpatioTemporalAugmentor(neg_aug, **neg_aug_kwargs)
        self.neg_aug = neg_aug
        self.direct_load_video_clip = direct_load_video_clip

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        # return torch.load(os.path.join(self.buffer_dir, f'{index}.bin')) # debug

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        audio_path = sample.get('audio_path', None)
        aframes, ainfo = read_audio(audio_path, backend='auto')
        audio_fps = ainfo.get("audio_fps", 16000)
        assert audio_fps == sample["audio_fps"], audio_fps

        video_fps = 24  # default fps
        if self.audio_only:
            duration = num_frames / video_fps
            num_audio_frames = int(duration * audio_fps)  # lower bound
            audio = temporal_random_crop(aframes, num_audio_frames, 1)
            audio = self.audio_transform(audio, duration=duration)
            video = None
            neg_videos, neg_audios = None, None
        elif file_type == "video":
            video_fps, video, audio, onset, neg_videos, neg_audios = load_video_audio_transform(
                path, self.direct_load_video_clip, num_frames, self.frame_interval, 
                audio_path, audio_fps, aframes, sample.get("unpaired_audio_path", None), 
                get_transforms_video(self.transform_name, (height, width)), self.audio_transform, self.augmenter,
                self.require_onset
            )
        else:
            raise NotImplementedError(file_type)
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        if video is not None:
            video = video.permute(1, 0, 2, 3)
        else:
            video = torch.zeros((1, num_frames, 1, 1))  # dummy video
        
        ret = {
            "index": index,
            "video": video,
            "audio": audio,  # mel_spec, shape(1, Ta, M)
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
            "audio_fps": audio_fps
        }
        if self.require_onset:
            ret.update({"onset": onset, })    # audio peak,
        if neg_videos:
            # NTCHW -> NCTHW
            for aug_type, aug_video in neg_videos.items():
                neg_videos[aug_type] = aug_video.permute(0, 2, 1, 3, 4)
            ret.update({"neg_videos": neg_videos, "neg_audios": neg_audios})

        if self.get_text:
            ret["text"] = sample["text"]
            ret["video_text"] = sample.get("video_text", "")
            ret["audio_text"] = sample.get("audio_text", "")
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
            ret["audio_text"] = torch.zeros((text_len, 1024))  # for AudioLDM2-T5
            ret["audio_text_2"] = torch.zeros((8, 768))        # for AudioLDM2-GPT2
            ret["audio_mask"] = text_len
        
        ## load pre-processed data
        if self.buffer_dir is not None:
            buffer_path = os.path.join(self.buffer_dir, f'{index}.bin')
            assert os.path.exists(buffer_path), buffer_path
            ret.update(torch.load(buffer_path, map_location='cpu'))

        return ret

    def __getitem__(self, index):
        # return self.getitem(index)
        try:
            return self.getitem(index)
        except Exception as e:
            print(e)
            return None


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None, **kwargs):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "audio": batch.get("ax", torch.zeros((0,))),
            "raw_text": batch.get('raw_text', batch["text"]),
            "text": batch["text"],
            "fps": batch["fps"],
            "audio_fps": batch.get("audio_fps", 16000),
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        if 'y' in batch:
            ret.update({
                "text": batch.get("y"),
                "mask": batch.get("mask", None),
            })
        if 'ay' in batch:
            ret.update({
                "audio_text": batch.get("ay", None),
                "audio_text_2": batch.get("ay2", None),
                "audio_mask": batch.get("audio_mask", None),
            })
        if 'neg_vx' in batch:
            assert 'neg_ax' in batch
            ret.update({
                "neg_videos": batch["neg_vx"],
                "neg_audios": batch["neg_ax"],
            })
        if 'onset' in batch:
            ret.update({'onset': batch['onset']})
        
        return ret

    def set_step(self, step):
        # TODO
        pass