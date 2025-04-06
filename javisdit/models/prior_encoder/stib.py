from typing import Optional, List, Dict, Literal, Union, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import logging

from javisdit.registry import MODELS
from javisdit.utils.misc import requires_grad
from javisdit.utils.ckpt_utils import load_checkpoint
from javisdit.models.prior_encoder.base_model import STPriorExtractor, STPriorExtractorConfig

from .ImageBind import imagebind_model, ModalityType
from .ImageBind import data as imagebind_data


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class STIBPriorExtractorConfig(STPriorExtractorConfig):
    model_type = "STIBPriorExtractor"

    def __init__(self, **kwargs):
        self.imagebind_ckpt_path = kwargs.get('imagebind_ckpt_path')
        super().__init__(**kwargs)


@MODELS.register_module("stib")
class STIBPriorExtractor(STPriorExtractor):
    """
    Spatio-Temporal ImageBind Extractor
    """
    def __init__(self, config: STIBPriorExtractorConfig):
        text_emb_dim, text_encoder = self.load_imagebind(config.imagebind_ckpt_path)
        config.text_emb_dim = text_emb_dim

        super().__init__(config)

        self.text_encoder = text_encoder
        
        logger.info('STIB initialized.')

    def load_imagebind(self, imagebind_ckpt_path):
        encoder, ib_dim = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        logger.info("Release unused modalities for ImageBind: VISION, AUDIO, THERMAL, DEPTH, IMU.")
        for modality_key in [ModalityType.VISION, ModalityType.AUDIO, ModalityType.THERMAL,
                             ModalityType.DEPTH, ModalityType.IMU]:
            del encoder.modality_preprocessors[modality_key]
            del encoder.modality_trunks[modality_key]
            del encoder.modality_heads[modality_key]
            del encoder.modality_postprocessors[modality_key]

        encoder.eval()
        requires_grad(encoder, False)

        return ib_dim, encoder
    
    def encode_text(self, text: List[str]):
        device = self.spatial_query_emb.device
        ib_text = imagebind_data.load_and_transform_text(text, device)
        inputs = {ModalityType.TEXT: ib_text}
        with torch.no_grad():
            ib_embeddings = self.text_encoder(inputs, return_hidden_states=True)
        # ib_text_emb = ib_embeddings[ModalityType.TEXT]  # shape(bs, 1024)
        ib_text_hidden = ib_embeddings[ModalityType.TEXT+ModalityType.HIDDEN_SUFFIX] 

        return ib_text_hidden  # shape(bs, 77, 1024)


@MODELS.register_module("STIBPrior")
def BaseSTPrior(from_pretrained=None, **kwargs):
    if from_pretrained is not None and not os.path.isfile(from_pretrained):
        model = STIBPriorExtractor.from_pretrained(from_pretrained, **kwargs)
    else:
        config = STIBPriorExtractorConfig(**kwargs)
        model = STIBPriorExtractor(config)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained, strict=True)
    return model
