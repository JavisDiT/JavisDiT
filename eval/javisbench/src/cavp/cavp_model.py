"""
Based on the official implementation of DiffFoley (Apache-2.0 license)
https://github.com/luosiallen/Diff-Foley
"""


from .cavp_modules import ResNet3dSlowOnly, Cnn14
import torch.nn as nn 
import torch
import numpy as np
import torch.nn.functional as F

class CAVP_Inference(nn.Module):

    def __init__(
            self,
            video_encode,
            spec_encode,
            embed_dim: int,
            video_pretrained: bool = False,
            audio_pretrained: bool = False,
    ):
        super().__init__()

        self.video_encode = video_encode
        self.spec_encode = spec_encode


        # 1). Video Encoder:
        assert self.video_encode == "Slowonly_pool"
        self.video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None)    # Doesn't matter to set pretrained=None, since we will load CAVP weight outside.
        
        # Video Project & Pooling Head:
        self.video_project_head = nn.Linear(2048, embed_dim)
        self.video_pool = nn.MaxPool1d(kernel_size=16)


        # 2). Spec Encoder:
        assert self.spec_encode == "cnn14_pool"     # Pretrained
        self.spec_encoder = Cnn14(embed_dim=512)
        
        # Spec Project & Pooling Head:
        self.spec_project_head = nn.Identity()
        self.spec_pool = nn.MaxPool1d(kernel_size=16)

        # 3). Logit Scale:
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        

    def encode_video(self, video, normalize: bool = False, train=False, pool=True):

        # Video: B x T x 3 x H x W
        assert self.video_encode == "Slowonly_pool"
        video = video.permute(0, 2, 1, 3, 4)
        video_feat = self.video_encoder(video)
        bs, c, t, _, _ = video_feat.shape
        video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
        video_feat = self.video_project_head(video_feat)
        
        # Pooling:
        if pool:
            video_feat = self.video_pool(video_feat.permute(0,2,1)).squeeze(2)
        
        # Normalize:
        if normalize:
            video_feat = F.normalize(video_feat, dim=-1)

        return video_feat


    def encode_spec(self, spec, normalize: bool = False, pool=True):
        # spec: B x Mel_num x T
        assert self.spec_encode == "cnn14_pool"
        spec = spec.unsqueeze(1)                        # B x 1 x Mel x T
        spec = spec.permute(0, 1, 3, 2)                 # B x 1 x T x Mel
        spec_feat = self.spec_encoder(spec)             # B x T x C
        spec_feat = self.spec_project_head(spec_feat)
        
        # Pooling:
        if pool:
            spec_feat = self.spec_pool(spec_feat.permute(0, 2, 1)).squeeze(2)

        # Normalize:
        if normalize:
            spec_feat = F.normalize(spec_feat, dim=-1)
        
        return spec_feat


    def forward(self, video, spec, output_dict=True):
        video_features = self.encode_video(video, normalize=True)
        spec_features = self.encode_spec(spec, normalize=True)
        if output_dict:
            return {
                "video_features": video_features,
                "spec_features": spec_features,
                "logit_scale": self.logit_scale.exp()
            }
        return video_features, spec_features, self.logit_scale.exp()