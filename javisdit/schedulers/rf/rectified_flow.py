from typing import Dict
import warnings

import torch
from torch.distributions import LogisticNormal

from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from javisdit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            t = self.sample_timestep(x_start, model_kwargs)
            # if self.use_discrete_timesteps:
            #     t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            # elif self.sample_method == "uniform":
            #     t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            # elif self.sample_method == "logit-normal":
            #     t = self.sample_t(x_start) * self.num_timesteps

            # if self.use_timestep_transform:
            #     t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms
    
    def multimodal_training_losses(self, model, x_start: Dict[str, torch.Tensor], model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep for multi-modal diffusion (e.g., synchronized video and audio).
        Arguments format copied from javisdit/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        Note: mask is a unified temporal mask for all modalities, dict{modality: t_mask}
        """
        modal_keys = list(x_start.keys())  # e.g., ['video', 'audio']
        assert len(set([x.shape[0] for x in x_start.values()])) == 1, 'Unmatched batch size'
        B = x_start[modal_keys[0]].shape[0]
        if t is None:
            t = self.sample_timestep(x_start[modal_keys[0]], model_kwargs)
        if isinstance(mask, torch.Tensor):
            warnings.warn('Warning: t_mask is a tensor, which is automatically warpped as video t_mask')
            mask = {k: mask if k == 'video' else None}
        
        if model_kwargs is None:
            model_kwargs = {}
        audio_only = model_kwargs.get('audio_only', False)

        if noise is None:
            # TODO: joint sampling or not, maybe not
            noise = {k: torch.randn_like(x) for k, x in x_start.items()}
        
        x_t: Dict[str, torch.Tensor] = {}
        for k in modal_keys:
            assert noise[k].shape == x_start[k].shape
            x_t[k] = self.add_noise(x_start[k], noise[k], t)
            if mask is not None and mask.get(k) is not None:
                t0 = torch.zeros_like(t)
                x_t0 = self.add_noise(x_start[k], noise[k], t0)
                dim_ones = [1] * len(x_t[k].shape[3:])
                t_mask = mask[k].view(B, 1, -1, *dim_ones)
                x_t[k] = torch.where(t_mask, x_t[k], x_t0)

        terms = {}
        model_output: Dict[str, torch.Tensor] = model(x_t, t, **model_kwargs)
        if audio_only:
            for k in list(model_output.keys()):
                if k != 'audio':
                    del model_output[k]
        loss_total = 0.0
        for k, velocity_pred in model_output.items():
            velocity_label = x_start[k] - noise[k]
            if velocity_pred.shape[1] != velocity_label.shape[1]:  # TODO: don't know why
                assert velocity_pred.shape[1] == velocity_label.shape[1] * 2
                velocity_pred = velocity_pred.chunk(2, dim=1)[0]
            t_mask = mask[k]
            if weights is None:
                loss = mean_flat((velocity_pred - velocity_label).pow(2), mask=t_mask)
            else:
                weight = _extract_into_tensor(weights, t, x_start[k].shape)
                loss = mean_flat(weight * (velocity_pred - velocity_label).pow(2), mask=t_mask)
            terms[f"loss_{k}"] = loss.detach().mean().item()
            loss_total += loss
        terms["loss"] = loss_total

        return terms

    def sample_timestep(self, x_start, model_kwargs=None):
        if self.use_discrete_timesteps:
            t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        elif self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * self.num_timesteps

        if self.use_timestep_transform:
            t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)
        
        return t

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        # timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        target_dim = noise.shape[1:]
        timepoints = timepoints.view(-1, *([1] * len(target_dim)))
        timepoints = timepoints.repeat(1, *target_dim)

        return timepoints * original_samples + (1 - timepoints) * noise
