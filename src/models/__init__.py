"""
Model architectures and PEFT adaptation wrappers.
"""

from .checkpoint_utils import extract_state_dict, checkpoint_targets
from .baselines import ConstantBaseline, TabularMLBaseline, SimpleUNet
from .decoder import DenseRegressionDecoder, NaiveBilinearDecoder
from .tta import predict_with_d4_tta

try:
    from .prithvi_backbone import PrithviBackboneWrapper, PrithviViTBlock, PrithviAttention
    from .multimodal_adapter import (
        LoRALinear, SARTXTEncoderCNN, GatedCrossModalFusion,
        LocalAuxiliaryEncoder, LocalResidualFusion,
        PrithviMultimodalAdapterModel
    )
except ImportError:
    PrithviBackboneWrapper = None
    PrithviViTBlock = None
    PrithviAttention = None
    LoRALinear = None
    SARTXTEncoderCNN = None
    GatedCrossModalFusion = None
    LocalAuxiliaryEncoder = None
    LocalResidualFusion = None
    PrithviMultimodalAdapterModel = None
