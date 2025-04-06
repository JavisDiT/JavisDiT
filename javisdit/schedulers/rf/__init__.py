import torch
from tqdm import tqdm
import numpy as np
import pdb

from javisdit.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        ## y: shape(bs,1,300,4096), last hidden state
        model_args = text_encoder.encode(prompts)
        ## y_null: randn(bs,1,300,4096), seems repeated; null prompt for classifier-free guidance
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        ## num_timesteps means diffusion/training steps; num_sampling_steps means denoising/inference steps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in enumerate(progress_wrap(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def multimodal_sample(
        self,
        model,
        text_encoder,
        latent_dict,
        prompts,
        device,
        additional_args=None,
        mask=None,
        prior_encoder=None,
        guidance_scale=None,
        progress=True,
        return_attn_map=False
    ):
        modal_keys = list(latent_dict.keys())  # e.g., ['video', 'audio']

        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = latent_dict[modal_keys[0]].shape[0]
        # text encoding
        ## y: shape(bs,1,300,4096), last hidden state;
        if isinstance(prompts, dict):
            model_args = prompts
            prompts = prompts.pop("prior_emb")  # compatible for prior_encoder.encode(prompts)
        else:
            model_args = text_encoder.encode(prompts)
        ## y_null: randn(bs,1,300,4096), seems repeated; null prompt for classifier-free guidance
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        # spatio-temporal prior encoding
        if prior_encoder is not None:
            prior_dict = prior_encoder.encode(prompts)
            # model_args.update({k: torch.cat((v, v), dim=0) for k, v in prior_dict.items()})
            spatial_prior, temporal_prior = prior_dict['spatial_prior'], prior_dict['temporal_prior']
            if getattr(prior_encoder.st_prior_embedder, 'y_embedding', None) is not None:
                null_spatial_prior, null_temporal_prior = prior_encoder.null(n)
            else:
                null_spatial_prior = torch.randn_like(spatial_prior)
                null_temporal_prior = torch.randn_like(temporal_prior)
            prior_null_dict = {
                'spatial_prior': null_spatial_prior, 
                'temporal_prior': null_temporal_prior, 
            }
            model_args.update({  # compatible for onset prediction
                k: torch.cat((v, prior_null_dict.get(k, torch.randn_like(v))), dim=0) \
                    for k, v in prior_dict.items()
            })
        if additional_args is not None:
            model_args.update(additional_args)
        model_args['num_timesteps'] = self.num_timesteps
        model_args['return_attn_map'] = return_attn_map

        # prepare timesteps
        ## num_timesteps means diffusion/training steps; num_sampling_steps means denoising/inference steps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * n, device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = {}
            for k, m in mask.items():
                noise_added[k] = torch.zeros_like(m, dtype=torch.bool)
                noise_added[k] = noise_added[k] | (m == 1)

        # pdb.set_trace()
        attn_map_dict = {}
        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in enumerate(progress_wrap(timesteps)):
            z_in = {}
            for k, z in latent_dict.items():
                # mask for adding noise
                if mask is not None:
                    mask_t = mask[k] * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                    mask_t_upper = mask_t >= t.unsqueeze(1)
                    model_args["x_mask" if k == 'video' else 'ax_mask'] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added[k]

                    if k == 'video':
                        mask_add_noise = mask_add_noise[:, None, :, None, None]
                    elif k == 'audio':
                        mask_add_noise = mask_add_noise[:, None, :, None]
                    assert len(mask_add_noise.shape) == len(x_noise.shape) == len(x0.shape)

                    z = torch.where(mask_add_noise, x_noise, x0)
                    noise_added[k] = mask_t_upper

                # classifier-free guidance
                z_in[k] = torch.cat([z, z], 0)
            t_in = torch.cat([t, t], 0)

            mm_pred = model(z_in, t_in, **model_args)
            attn_maps = mm_pred.pop('attn_maps', None)
            if i in np.linspace(0, len(timesteps)-1, 4, dtype=int):
                attn_map_dict[f'step{i}'] = attn_maps
            
            for k, pred in mm_pred.items():
                pred = pred.chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

                # update z
                z = latent_dict[k]
                x0 = z.clone()
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt.view(-1, *([1] * (len(z.shape) - 1)))

                if mask is not None:
                    if k == 'video':
                        mask_t_upper = noise_added[k][:, None, :, None, None]
                    elif k == 'audio':
                        mask_t_upper = noise_added[k][:, None, :, None]
                    assert len(mask_t_upper.shape) == len(z.shape) == len(x0.shape), f'{mask_t_upper.shape} {z.shape} {x0.shape}'
                    z = torch.where(mask_t_upper, z, x0)
                
                latent_dict[k] = z

        if return_attn_map:
            latent_dict['attn_map'] = attn_map_dict

        return latent_dict
    
    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def multimodal_training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.multimodal_training_losses(model, x_start, model_kwargs, noise, mask, weights, t)
