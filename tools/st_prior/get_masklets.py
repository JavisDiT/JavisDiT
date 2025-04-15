import argparse
import warnings
import os
import torch
import tqdm
import torch.distributed as dist
import cv2
import logging
import numpy as np
import sys
from typing import List, Tuple, Union, Optional, Dict

sys.path.append('./third_party/Grounded-Segment-Anything/EfficientSAM')
from RepViTSAM.setup_repvit_sam import build_sam_repvit
import math
import torch.nn.functional as F
import json
import pycocotools.mask as mask_util
from torchvision.transforms.functional import resize
from copy import deepcopy
from glob import glob
import time

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

local_rank = int(os.environ.get("LOCAL_RANK", -1))
rank = int(os.environ.get("RANK", -1))
torch.cuda.set_device(local_rank)
logging.info(f'rank {rank} local_rank {local_rank}')

IMG_SIZE = 1024
MAX_CROP_SIZE = 256
dtype = torch.float32


class ResizeLongestSideForVideo:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: torch.Tensor) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        t, c, h, w = image.shape
        target_size = self.get_preprocess_shape(h, w, self.target_length)
        return resize(image, target_size)

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


global_resize = ResizeLongestSideForVideo(IMG_SIZE)


def scale_bboxes_with_limit(image_size, bboxes, scale_factor):
    image_height, image_width = image_size
    
    width = bboxes[:, 2] - bboxes[:, 0]
    height = bboxes[:, 3] - bboxes[:, 1]
    
    new_width = width * scale_factor
    new_height = height * scale_factor
    
    x_min_new = bboxes[:, 0] - (new_width - width) / 2
    y_min_new = bboxes[:, 1] - (new_height - height) / 2
    x_max_new = bboxes[:, 2] + (new_width - width) / 2
    y_max_new = bboxes[:, 3] + (new_height - height) / 2
    
    x_min_new = np.clip(x_min_new, 0, image_width)
    y_min_new = np.clip(y_min_new, 0, image_height)
    x_max_new = np.clip(x_max_new, 0, image_width)
    y_max_new = np.clip(y_max_new, 0, image_height)
    
    scaled_bboxes = np.column_stack((x_min_new, y_min_new, x_max_new, y_max_new))
    
    return scaled_bboxes


def read_video_cv(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    return frames


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, output_dir, annot_every_frames=4):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.data = []
        with open(video_path, 'r') as f:
            data = [x.strip() for x in f.readlines()]
        for video_path in tqdm.tqdm(data, desc="scaning processed data"):
            video_name = os.path.basename(video_path).split('.')[0]
            done = True
            for i in range(1000):
                video_out = os.path.exists(f'{output_dir}/{video_name}_masklet_{i:03d}.mp4')
                mask_out = os.path.exists(f'{output_dir}/{video_name}_mask_{i:03d}.mp4')
                meta_out = os.path.exists(f'{output_dir}/{video_name}_meta_{i:03d}.json')
                if not any([video_out, mask_out, meta_out]):  # at least one missing
                    if i == 0 or any([video_out, mask_out, meta_out]):  # start or imcomplete
                        done = False
                    # else:
                    #     done = True
                    break
            if not done:
                self.data.append(video_path)
        
        self.total_videos = len(self.data)
        self.img_size = IMG_SIZE
        self.annot_every_frames = annot_every_frames
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)

        per_rank = len(self.data) // self.world_size
        start = self.rank * per_rank
        end = min(start + per_rank, len(self.data))
        rank_total_frame = 0
        for i in range(start, end):
            video_path = self.data[i]
            manual_annot_path = video_path.replace('.mp4', '_manual.json')
            with open(manual_annot_path, 'r') as f:
                annots = json.load(f)
            frame_count = annots['video_frame_count']
            rank_total_frame += frame_count
        self.rank_total_frame = rank_total_frame
        self.data = self.data[start:end]
        logging.info(f'rank {self.rank} total video {end - start}')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video_path = self.data[index]
        vframes = read_video_cv(video_path) # opensora 提供的 read_video_av 有内存泄漏的问题
        manual_annot_path = video_path.replace('.mp4', '_manual.json')
        videoname = os.path.basename(video_path).split('.')[0]
        with open(manual_annot_path, 'r') as f:
            annots = json.load(f)
        return vframes, annots['masklet'], annots['masklet_size_bucket'], annots['masklet_size_rel'], videoname, annots['video_resolution']


def collate_fn(batch):
    vframes, masklets, size_bucket, size_rel, video_name, video_resolution = zip(*batch)
    vframes = torch.stack(vframes, dim=0)
    masklets = [masklet for masklet in masklets]
    size_buckets = [sb for sb in size_bucket]
    size_rels = [sr for sr in size_rel]
    video_names = [vn for vn in video_name]
    video_resolutions = [vr for vr in video_resolution]
    return vframes, masklets, size_buckets, size_rels, video_names, video_resolutions


def norm_and_pad(x: torch.Tensor, mean, std):
    x = (x - mean) / std
    h, w = x.shape[-2:]
    padh = IMG_SIZE - h
    padw = IMG_SIZE - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def precese(frames_ori, mean, std):
    original_image_shape = frames_ori.shape[-2:]
    frames = global_resize.apply_image(frames_ori)
    input_size = frames.shape[-2:]
    frames = norm_and_pad(frames, mean, std)
    return frames, original_image_shape, input_size


@torch.no_grad()
def segment(model, batch_images, batch_bboxes, original_img_sizes, model_input_sizes):
    features, _ = model.image_encoder(batch_images)

    batch_masks = []
    nums = []

    for feature, bbox in zip(features, batch_bboxes):
        boxes = global_resize.apply_boxes(bbox, original_img_sizes) 
        box_torch = torch.as_tensor(boxes, dtype=dtype, device=feature.device)
            
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box_torch, masks=None)

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=feature[None],
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=None
        )

        masks = model.postprocess_masks(low_res_masks, model_input_sizes, original_img_sizes)
        masks = masks > model.mask_threshold # logit > 0, sigmoid > 0.5
        batch_masks.append(masks[:, 0]) # only one mask for every box
        nums.append(masks.shape[0])    
    return batch_masks, nums


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] masks, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    return bbox_coords


def xyxy2xywh(xyxy):
    return np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], axis=-1)


@torch.no_grad()
def main(args):
    dataset = VideoDataset(args.data_path, args.output_dir)
    # sampler = DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        # sampler=sampler,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    if local_rank == 0:
        time.sleep(0.5)
        total_batch = len(data_loader)
        pbar = tqdm.tqdm(total=total_batch, ncols=50)

    repvit_sam = build_sam_repvit(checkpoint=args.repvit_sam_checkpoint_path).to(local_rank).to(dtype)
    repvit_sam.eval()
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1).to(local_rank)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1).to(local_rank)

    cnt = 0
    skip_videos = []
    for frames, masklets, size_buckets, size_rels, video_names, video_resolutions in data_loader:
        try:
            frames = frames[0] # shape(T, C, H, W)
            frames = frames[:int(frames.shape[0]) // 4 * 4]
            masklets = masklets[0] # List[List[Dict]]
            size_buckets = size_buckets[0]
            size_rels = size_rels[0]
            video_name = video_names[0]
            video_resolution = video_resolutions[0]

            relative_size = np.array(size_rels) / np.array(size_rels).max()
            valid_bbox = np.array([True if relative_size[i] >= 0.33 and size_buckets[i] != 'small' else False for i in range(len(masklets[0]))])

            if valid_bbox.sum() >= 10:
                skip_videos.append(video_name)
                del frames
                continue
            if video_resolution > 1920 * 1080:
                skip_videos.append(video_name)
                del frames
                continue

            video_crops = [[] for _ in range(valid_bbox.sum())]
            video_masklet_bbox = [[] for _ in range(valid_bbox.sum())]
            video_visible = []

            if not valid_bbox.any(): # 没有有效的 masklets
                del frames
                del masklets
                continue
            # print(f'{video_name} {valid_bbox.sum()}')
            
            for i in range(math.ceil(frames.shape[0] / args.sam_batch_size)):
                vframes_ori = frames[i * args.sam_batch_size: (i + 1) * args.sam_batch_size].to(local_rank).to(dtype)
                vframes, original_image_shape, input_size = precese(vframes_ori, pixel_mean, pixel_std)

                # decode bboxes
                start_frame_ind = i * args.sam_batch_size
                end_frame_ind = min((i + 1) * args.sam_batch_size, frames.shape[0])

                annoted_start_frame_ind = int(start_frame_ind // 4)
                annoted_end_frame_ind = min(int(end_frame_ind // 4), len(masklets) - 1)

                all_bboxes = [] # List[np.ndarray[n,4]]
                all_visible = [] # List[List[bool, n]]
                for frame_idx in range(start_frame_ind, end_frame_ind):
                    left_annoted_frame = frame_idx // 4
                    right_annoted_frame = min(left_annoted_frame + 1, annoted_end_frame_ind)
                    left_bboxes = mask_util.toBbox([m for i, m in enumerate(masklets[left_annoted_frame]) if valid_bbox[i]])
                    right_bboxes = mask_util.toBbox([m for i, m in enumerate(masklets[right_annoted_frame]) if valid_bbox[i]])
                    bboxes = []
                    visible = []
                    for l_bbox, r_bbox in zip(left_bboxes, right_bboxes):
                        if np.sum(l_bbox) == 0 or np.sum(r_bbox) == 0:
                            bboxes.append(np.zeros((4,)))
                            visible.append(False)
                        else:
                            factor = (frame_idx % 4) / 4
                            bboxes.append((1 - factor) * l_bbox + factor * r_bbox)
                            visible.append(True)
                    bboxes = np.array(bboxes)
                    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
                    bboxes = scale_bboxes_with_limit(original_image_shape, bboxes, 1.1)                
                    all_bboxes.append(bboxes)
                    all_visible.append(visible)
                
                batch_masks, _ = segment(repvit_sam, vframes, all_bboxes, original_image_shape, input_size)
                N, H, W = batch_masks[0].shape
                
                batch_masks = torch.stack(batch_masks) # 32,N,H,W
                
                batch_bboxes = mask_to_box(batch_masks.view(-1, 1, H, W)).view(-1, N, 4)

                all_crops = [[] for _ in range(valid_bbox.sum())]
                all_bboxes = [[] for _ in range(valid_bbox.sum())]
                for vframe, masks, visible, bboxes in zip(vframes_ori, batch_masks, all_visible, batch_bboxes): # 32 次
                    for obj_idx, (mask, vis, bbox) in enumerate(zip(masks, visible, bboxes)):
                        if not vis:
                            all_crops[obj_idx].append(torch.zeros((4, 32, 32), dtype=torch.uint8).to(local_rank))
                            all_bboxes[obj_idx].append(np.array([0, 0, 0, 0]))
                        else:
                            x1, y1, x2, y2 = bbox.cpu().numpy().astype(int).tolist()
                            if x2 - x1 < 1 or y2 - y1 < 1:
                                all_crops[obj_idx].append(torch.zeros((4, 32, 32), dtype=torch.uint8).to(local_rank))
                                all_bboxes[obj_idx].append(np.array([0, 0, 0, 0]))
                            else:
                                crop_with_mask = torch.cat([
                                    vframe[:, y1:y2, x1:x2].clone(),
                                    mask[y1:y2, x1:x2][None].clone(),
                                ], axis=0)
                                if y2 - y1 > MAX_CROP_SIZE or x2 - x1 > MAX_CROP_SIZE:
                                    crop_with_mask = resize(crop_with_mask, (MAX_CROP_SIZE, MAX_CROP_SIZE))
                                all_crops[obj_idx].append(crop_with_mask)
                                all_bboxes[obj_idx].append(bbox.cpu().numpy())
                                
                video_visible.extend(all_visible)     

                for i in range(valid_bbox.sum()):
                    video_crops[i].extend(all_crops[i])
                    video_masklet_bbox[i].extend(all_bboxes[i])
                del vframes, vframes_ori
                torch.cuda.empty_cache()
            
            # 首先resize到同一个大小，然后存储下来
            for i in range(valid_bbox.sum()):
                visible = np.array(video_visible)[:, i] # L
                masklet_bbox = np.array(video_masklet_bbox[i]) # L,4
                crop_with_masks = video_crops[i]
                video_output_path = os.path.join(args.output_dir, f'{video_name}_masklet_{i:03d}.mp4')
                mask_output_path = os.path.join(args.output_dir, f'{video_name}_mask_{i:03d}.mp4')
                meta_output_path = os.path.join(args.output_dir, f'{video_name}_meta_{i:03d}.json')

                # resize
                masklet_bbox_xywh = xyxy2xywh(masklet_bbox) # L,4
                if np.all(visible == 0):
                    continue
                
                valid_seq_start, valid_seq_end = np.nonzero(visible)[0][0], np.nonzero(visible)[0][-1]

                if valid_seq_end - valid_seq_start < 5: # 5 帧都没有，就不保留了
                    continue

                visible = visible[valid_seq_start:valid_seq_end+1]
                masklet_bbox = masklet_bbox[valid_seq_start:valid_seq_end+1]
                masklet_bbox_xywh = masklet_bbox_xywh[valid_seq_start:valid_seq_end+1]
                max_w, max_h = masklet_bbox_xywh[:, 2].max(), masklet_bbox_xywh[:, 3].max()
                max_w, max_h = min(max_w, MAX_CROP_SIZE), min(max_h, MAX_CROP_SIZE)
                if max_w == 0 or max_h == 0:
                    continue

                meta_info = {
                    'video_frame_height': int(original_image_shape[0]),
                    'video_frame_width': int(original_image_shape[1]),
                    'visible': visible.tolist(),
                    'crop_bbox': masklet_bbox_xywh.tolist()
                }

                with open(meta_output_path, 'w') as f:
                    json.dump(meta_info, f)

                out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (max_w, max_h))
                out_mask = cv2.VideoWriter(mask_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (max_w, max_h), isColor=False)

                for j in range(valid_seq_start, valid_seq_end+1):
                    frame = resize(crop_with_masks[j], (max_h, max_w)).permute(1, 2, 0)
                    bgr_frame = frame[...,:3].contiguous().to(torch.uint8).cpu().numpy()
                    mask_frame = (frame[...,3] * 255).contiguous().to(torch.uint8).cpu().numpy()
                    out.write(bgr_frame)
                    out_mask.write(mask_frame)
                out.release()
                out_mask.release()
                torch.cuda.empty_cache()
            
            del frames
            del video_crops
            del video_masklet_bbox
            del video_visible
            del crop_with_masks
            torch.cuda.empty_cache()
        except:
            skip_videos.append(video_names[0])
            logging.warning(f'error in {video_names[0]}')

        if local_rank == 0:
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
    logging.warning(f'done with skip {len(skip_videos)} due to the large video resolutions or many masklets, use small sam_batch_size for these videos')
    for video_name in skip_videos:
        logging.info(f'{video_name} is skipped')
    

if __name__ == '__main__':
    dist.init_process_group(backend='nccl', init_method='env://')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/SA-V/sa_v_list.txt')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sam_batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='data/SA-V/crops')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--repvit_sam_checkpoint_path', type=str,
                        default='./third_party/Grounded-Segment-Anything/EfficientSAM/repvit_sam.pt')
    args = parser.parse_args()

    assert args.batch_size == 1, "batch_size must be 1"
    assert args.sam_batch_size % 4 == 0, "sam_batch_size must be divisible by 4"
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)

    dist.destroy_process_group() 
