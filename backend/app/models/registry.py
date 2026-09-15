from typing import List, Dict, Any
from pydantic import BaseModel

class ModelMetadata(BaseModel):
    id: str
    name: str
    architecture: str
    domain: str
    default_dataset: str
    task: str
    auc_roc: float
    accuracy: float
    calibration_ece: float
    parameters: str
    status: str
    layer_hook: str
    is_active: bool = False

MODEL_REGISTRY: List[ModelMetadata] = [
    ModelMetadata(
        id="efficientnet-b4",
        name="EfficientNet-B4 (Dermoscopy)",
        architecture="EfficientNet-B4 (Tan & Le, ICML 2019)",
        domain="Dermoscopy",
        default_dataset="ISIC 2024 / HAM10000",
        task="Skin Lesion Multi-Class Diagnosis",
        auc_roc=0.934,
        accuracy=90.1,
        calibration_ece=0.038,
        parameters="19.3M",
        status="Active (Primary Backbone)",
        layer_hook="_blocks.31._project_conv",
        is_active=True
    ),
    ModelMetadata(
        id="densenet-121",
        name="DenseNet-121 (Radiology Backbone)",
        architecture="DenseNet-121 (Huang et al., CVPR 2017)",
        domain="Chest Radiograph (CXR)",
        default_dataset="CheXpert",
        task="Thoracic Multi-Label Classification",
        auc_roc=0.912,
        accuracy=88.4,
        calibration_ece=0.048,
        parameters="7.0M",
        status="Available (Research Baseline)",
        layer_hook="features.denseblock4.denselayer16.conv2"
    ),
    ModelMetadata(
        id="vit-base",
        name="Vision Transformer (ViT-B/16)",
        architecture="ViT-B/16 (Dosovitskiy et al., ICLR 2021)",
        domain="Chest Radiograph (CXR)",
        default_dataset="CheXpert",
        task="Thoracic Pathology Attribution",
        auc_roc=0.908,
        accuracy=87.9,
        calibration_ece=0.054,
        parameters="86.6M",
        status="Available",
        layer_hook="encoder.layers.encoder_layer_11"
    ),
    ModelMetadata(
        id="swin-transformer",
        name="Swin Transformer (Neuro-Oncology)",
        architecture="Swin-B (Liu et al., ICCV 2021)",
        domain="Brain MRI",
        default_dataset="BraTS 2023",
        task="Brain Tumor Multi-Region Characterization",
        auc_roc=0.925,
        accuracy=89.3,
        calibration_ece=0.042,
        parameters="88.0M",
        status="Active",
        layer_hook="layers.3.blocks.1"
    )
]
