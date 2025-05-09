import collections
import random
from typing import Optional, List

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from .datasets import BatchFeatureDataset, VariableVideoTextDataset, VideoTextDataset, VariableVideoAudioTextDataset, VideoAudioTextDataset
from .sampler import BatchDistributedSampler, StatefulDistributedSampler, VariableVideoBatchSampler, VariableVideoAudioBatchSampler


# Deterministic dataloader
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader(
    dataset,
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    prefetch_factor=None,
    sampler_kwargs={},
    **kwargs,
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, VariableVideoTextDataset):
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
            **sampler_kwargs
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VariableVideoAudioTextDataset):
        # simple copy from VariableVideoBatchSampler
        batch_sampler = VariableVideoAudioBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
            **sampler_kwargs
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VideoTextDataset) or isinstance(dataset, VideoAudioTextDataset):
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            **sampler_kwargs
        )
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    elif isinstance(dataset, BatchFeatureDataset):
        sampler = BatchDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            **sampler_kwargs
        )
        return (
            DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_batch,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def collate_fn_default(batch):
    def _padding(texts, masks):
        text_lens = []
        if isinstance(masks[0], int):
            text_lens = masks
            is_padding = False
        elif isinstance(masks[0], torch.Tensor):
            text_lens = [len(mask) for mask in masks]
            is_padding = True
        else:
            raise ValueError(f'Invalid mask type: {type(masks[0])}')

        if is_padding:
            max_len = max(text_lens)
            for i, l in enumerate(text_lens):
                delta = max_len - l
                pad_shape = list(texts[i].shape)
                pad_shape[-2] = delta
                pad_dtype = texts[i].dtype
                texts[i] = torch.cat([texts[i], torch.randn(pad_shape, dtype=pad_dtype)], dim=-2)
                pad_shape = (delta, )
                pad_dtype = masks[i].dtype
                masks[i] = torch.cat([masks[i], torch.zeros(pad_shape, dtype=pad_dtype)], dim=0)
            texts, masks = torch.stack(texts), torch.stack(masks)
        else:
            assert len(set(text_lens)) == 1
            texts = torch.cat(texts, dim=-2)  # TODO: check

        return texts, masks

    # filter out None
    batch = [x for x in batch if x is not None]
    assert len(batch) > 0
    # if len(batch) == 0:
    #     return {}

    # HACK: for loading text features
    use_mask, with_audio = False, False
    if "mask" in batch[0]:
        masks = [x.pop("mask") for x in batch]
        texts = [x.pop("text") for x in batch]
        texts, masks = _padding(texts, masks)
        use_mask = True

        if "audio_mask" in batch[0]:
            audio_masks = [x.pop("audio_mask") for x in batch]
            audio_texts = [x.pop("audio_text") for x in batch]
            audio_texts, audio_masks = _padding(audio_texts, audio_masks)
            if 'audio_text_2' in batch[0]:
                audio_texts_2 = [x.pop("audio_text_2") for x in batch]
                audio_texts_2 = torch.stack(audio_texts_2, dim=0)
            else:
                audio_texts_2 = None
            with_audio = True

    try:
        ret = torch.utils.data.default_collate(batch)
    except Exception as e:  # TODO: debug
        print(e)
        for i, x in enumerate(batch):
            for k, v in x.items():
                print(i, k, v.shape if isinstance(v, torch.Tensor) else v)
        exit()

    if use_mask:
        ret["mask"] = masks
        ret["text"] = texts
        if with_audio:
            ret["audio_mask"] = audio_masks
            ret["audio_text"] = audio_texts
            if audio_texts_2 is not None:
                ret["audio_text_2"] = audio_texts_2

    return ret


def collate_fn_batch(batch):
    """
    Used only with BatchDistributedSampler
    """
    # filter out None
    batch = [x for x in batch if x is not None]

    # TODO: stupid
    assert len(batch) == 1
    return batch[0]
    
    res = torch.utils.data.default_collate(batch)

    # squeeze the first dimension, which is due to torch.stack() in default_collate()
    if isinstance(res, collections.abc.Mapping):
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.squeeze(0)
    elif isinstance(res, collections.abc.Sequence):
        res = [x.squeeze(0) if isinstance(x, torch.Tensor) else x for x in res]
    elif isinstance(res, torch.Tensor):
        res = res.squeeze(0)
    else:
        raise TypeError

    return res
