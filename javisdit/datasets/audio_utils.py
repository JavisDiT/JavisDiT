import os
import pandas as pd
from scipy.signal import get_window
import librosa
import librosa.util as librosa_util
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn

from typing import Union, Optional

import random
import torch.nn.functional as F
import torch
import numpy as np
import torchaudio


def detect_audio_peaks(audio_file=None, y=None, sr=None):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        onset_times (np.ndarray): List of times (in seconds) where audio peaks occur.
    """
    if y is None:
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        assert y is not None and sr is not None
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


class AudioLDM2MelTransform(object):
    def __init__(
        self,
        config=None,
        split="train",
        waveform_only=False,
    ):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        self.config = config
        self.split = split
        self.pad_wav_start_sample = 0  # If none, random choose
        self.trim_wav = False
        self.waveform_only = waveform_only

        self.build_setting_parameters()
        self.build_dsp()
    
    def get_mel_from_wav(self, audio):
        audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec, magnitudes, phases, energy = self.STFT.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
        magnitudes = torch.squeeze(magnitudes, 0).numpy().astype(np.float32)
        energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
        return melspec, magnitudes, energy

    def __call__(self, waveform: Union[np.ndarray, torch.Tensor], duration: Optional[float] = None):
        # TODO: resample augmentation

        # check data
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
        if len(waveform.shape) == 2:
            waveform = waveform[0]

        # normalize
        waveform = self.normalize_wav(waveform)

        # get mel_spec
        log_mel_spec, stft, energy = self.get_mel_from_wav(waveform)
        log_mel_spec = torch.from_numpy(log_mel_spec.T)
        log_mel_spec = self.pad_spec(log_mel_spec, duration)

        # reshape(1, t, 64)
        log_mel_spec = log_mel_spec[None, :, :]

        return log_mel_spec

    def build_setting_parameters(self):
        # Read from the json config
        self.melbins = self.config["preprocessing"]["mel"]["n_mel_channels"]
        # self.freqm = self.config["preprocessing"]["mel"]["freqm"]
        # self.timem = self.config["preprocessing"]["mel"]["timem"]
        self.sampling_rate = self.config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = self.config["preprocessing"]["stft"]["hop_length"]
        self.duration = self.config["preprocessing"]["audio"]["duration"]
        self.target_length = int(self.duration * self.sampling_rate / self.hopsize)

        self.mixup = self.config["augmentation"]["mixup"]

        # Calculate parameter derivations
        # self.waveform_sample_length = int(self.target_length * self.hopsize)

        # if (self.config["balance_sampling_weight"]):
        #     self.samples_weight = np.loadtxt(
        #         self.config["balance_sampling_weight"], delimiter=","
        #     )

        if "train" not in self.split:
            self.mixup = 0.0
            # self.freqm = 0
            # self.timem = 0

    def build_dsp(self):
        self.STFT = TacotronSTFT(
            self.config["preprocessing"]["stft"]["filter_length"],
            self.config["preprocessing"]["stft"]["hop_length"],
            self.config["preprocessing"]["stft"]["win_length"],
            self.config["preprocessing"]["mel"]["n_mel_channels"],
            self.config["preprocessing"]["audio"]["sampling_rate"],
            self.config["preprocessing"]["mel"]["mel_fmin"],
            self.config["preprocessing"]["mel"]["mel_fmax"],
        )
        # self.stft_transform = torchaudio.transforms.Spectrogram(
        #     n_fft=1024, hop_length=160
        # )
        # self.melscale_transform = torchaudio.transforms.MelScale(
        #     sample_rate=16000, n_stft=1024 // 2 + 1, n_mels=64
        # )

    def normalize_wav(self, waveform):
        waveform = waveform - np.mean(waveform)
        waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
        return waveform * 0.5  # Manually limit the maximum amplitude into 0.5

    def pad_spec(self, log_mel_spec, duration=None, scale_factor=32):
        if duration is not None:
            target_length = int(duration * self.sampling_rate / self.hopsize)
        else:
            target_length = self.target_length
        if target_length % scale_factor != 0:
            target_length = int(np.ceil(target_length / scale_factor)) * scale_factor

        n_frames = log_mel_spec.shape[0]
        p = target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            log_mel_spec = m(log_mel_spec)
        elif p < 0:
            log_mel_spec = log_mel_spec[:target_length, :]

        if log_mel_spec.size(-1) % 2 != 0:
            log_mel_spec = log_mel_spec[..., :-1]

        return log_mel_spec

    # TODO: below are temporally unused functions
    def resample(self, waveform, sr):
        waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)
        # waveform = librosa.resample(waveform, sr, self.sampling_rate)
        return waveform

        # if sr == 16000:
        #     return waveform
        # if sr == 32000 and self.sampling_rate == 16000:
        #     waveform = waveform[::2]
        #     return waveform
        # if sr == 48000 and self.sampling_rate == 16000:
        #     waveform = waveform[::3]
        #     return waveform
        # else:
        #     raise ValueError(
        #         "We currently only support 16k audio generation. You need to resample you audio file to 16k, 32k, or 48k: %s, %s"
        #         % (sr, self.sampling_rate)
        #     )

    def random_segment_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        # Too short
        if (waveform_length - target_length) <= 0:
            return waveform, 0

        random_start = int(self.random_uniform(0, waveform_length - target_length))
        return waveform[:, random_start : random_start + target_length], random_start

    def pad_wav(self, waveform, target_length):
        waveform_length = waveform.shape[-1]
        assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

        if waveform_length == target_length:
            return waveform

        # Pad
        temp_wav = np.zeros((1, target_length), dtype=np.float32)
        if self.pad_wav_start_sample is None:
            rand_start = int(self.random_uniform(0, target_length - waveform_length))
        else:
            rand_start = 0

        temp_wav[:, rand_start : rand_start + waveform_length] = waveform
        return temp_wav

    def trim_wav(self, waveform):
        if np.max(np.abs(waveform)) < 0.0001:
            return waveform

        def detect_leading_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = 0
            while start + chunk_size < waveform_length:
                if np.max(np.abs(waveform[start : start + chunk_size])) < threshold:
                    start += chunk_size
                else:
                    break
            return start

        def detect_ending_silence(waveform, threshold=0.0001):
            chunk_size = 1000
            waveform_length = waveform.shape[0]
            start = waveform_length
            while start - chunk_size > 0:
                if np.max(np.abs(waveform[start - chunk_size : start])) < threshold:
                    start -= chunk_size
                else:
                    break
            if start == waveform_length:
                return start
            else:
                return start + chunk_size

        start = detect_leading_silence(waveform)
        end = detect_ending_silence(waveform)

        return waveform[start:end]

    # def read_wav_file(self, filename):
    #     # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    #     waveform, sr = torchaudio.load(filename)

    #     waveform, random_start = self.random_segment_wav(
    #         waveform, target_length=int(sr * self.duration)
    #     )

    #     waveform = self.resample(waveform, sr)
    #     # random_start = int(random_start * (self.sampling_rate / sr))

    #     waveform = waveform.numpy()[0, ...]

    #     waveform = self.normalize_wav(waveform)

    #     if self.trim_wav:
    #         waveform = self.trim_wav(waveform)

    #     waveform = waveform[None, ...]
    #     waveform = self.pad_wav(
    #         waveform, target_length=int(self.sampling_rate * self.duration)
    #     )
    #     return waveform, random_start

    def random_uniform(self, start, end):
        val = torch.rand(1).item()
        return start + (end - start) * val

    def frequency_masking(self, log_mel_spec, freqm):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(freqm // 8, freqm))
        mask_start = int(self.random_uniform(start=0, end=freq - mask_len))
        log_mel_spec[:, mask_start : mask_start + mask_len, :] *= 0.0
        return log_mel_spec

    def time_masking(self, log_mel_spec, timem):
        bs, freq, tsteps = log_mel_spec.size()
        mask_len = int(self.random_uniform(timem // 8, timem))
        mask_start = int(self.random_uniform(start=0, end=tsteps - mask_len))
        log_mel_spec[:, :, mask_start : mask_start + mask_len] *= 0.0
        return log_mel_spec

    # def augmentation(self, log_mel_spec: torch.Tensor):
    #     assert torch.min(log_mel_spec) < 0
    #     log_mel_spec = log_mel_spec.exp()

    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     # this is just to satisfy new torchaudio version.
    #     log_mel_spec = log_mel_spec.unsqueeze(0)
    #     if self.freqm != 0:
    #         log_mel_spec = self.frequency_masking(log_mel_spec, self.freqm)
    #     if self.timem != 0:
    #         log_mel_spec = self.time_masking(
    #             log_mel_spec, self.timem)  # self.timem=0

    #     log_mel_spec = (log_mel_spec + 1e-7).log()
    #     # squeeze back
    #     log_mel_spec = log_mel_spec.squeeze(0)
    #     log_mel_spec = torch.transpose(log_mel_spec, 0, 1)
    #     return log_mel_spec


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length / 2) :]
        inverse_transform = inverse_transform[:, :, : -int(self.filter_length / 2) :]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, 
            n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def spectral_normalize(self, magnitudes, normalize_fun):
        output = dynamic_range_compression(magnitudes, normalize_fun)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, normalize_fun=torch.log):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1, torch.min(y.data)
        assert torch.max(y.data) <= 1, torch.max(y.data)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output, normalize_fun)
        energy = torch.norm(magnitudes, dim=1)

        return mel_output, magnitudes, phases, energy


def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
