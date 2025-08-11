"""
AV-Align Metric: Audio-Video Alignment Evaluation

AV-Align is a metric for evaluating the alignment between audio and video modalities in multimedia data.
It assesses synchronization by detecting audio and video peaks and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.

Usage:
- Provide a folder of video files as input.
- The script calculates the AV-Align score for the set of videos.
"""


import argparse
import glob
import cv2
import os.path as osp
import librosa
import librosa.display
import json
# from sqlalchemy import desc
# from sympy import N
from tqdm import tqdm
import pdb
import numpy as np
import os

# cache_path = "./video_cache.json"
cache_json = None

# resie frames
def resize_frames(frames, new_size_scheme):
    """
    Args:
        frames (list): the elements in frames are numpy.ndarray.
        new_size_scheme (str):  resize scheme.
    Return:
        frames: the elements in list are resized frames.
    """
    h, w, _ = frames[0].shape
    # new_w, new_h = w, h
    if new_size_scheme.startswith("min"):
        min_edge = int(new_size_scheme.split("=")[1])
        scale_ratio = min_edge / min(w, h)
        new_h = int(scale_ratio * h)
        new_w = int(scale_ratio * w)
    elif new_size_scheme.find(":") != -1:
        new_w = int(new_size_scheme.split(":")[0])
        new_h = int(new_size_scheme.split(":")[1])

    if (w, h) == (new_w, new_h):
        return frames

    new_frames = []
    for img in frames:
        new_frames.append(
            cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR))

    return new_frames

# Function to extract frames from a video file
def extract_frames(video_path, resize_scheme=None, max_length_s=None):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        frames (list): List of frames extracted from the video.
        frame_rate (float): Frame rate of the video.
    """

    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open the video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_length_s is not None and len(frames) >= frame_rate * max_length_s:
            break
    cap.release()
    if resize_scheme is not None:
        frames = resize_frames(frames, resize_scheme)
    return frames, frame_rate


# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file=None, y=None, sr=None, max_length_s=None):
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
    if max_length_s is not None and len(y) > max_length_s * sr:
        y = y[:int(max_length_s * sr)]
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """

    local_extrema_indexes = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:  # Local maximum
            local_extrema_indexes.append(i / fps)

    return local_extrema_indexes


# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps, use_tqdm=True):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    if len(frames) == 0:
        return None, []
    
    if isinstance(frames[0], float):
        return None, frames
    
    # flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    flow_trajectory = [compute_of(frames[0], frames[1])]
    pbar = range(1, len(frames))
    if use_tqdm:
        pbar = tqdm(pbar, desc="Process Frames")
    for i in pbar:
        flow_trajectory.append(compute_of(frames[i - 1], frames[i]))

    video_peaks = find_local_max_indexes(flow_trajectory, fps)

    return flow_trajectory, video_peaks


# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    intersection_length = 0
    for audio_peak in audio_peaks:
        for video_peak in video_peaks:
            if video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                break
                
    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length + 1e-6)