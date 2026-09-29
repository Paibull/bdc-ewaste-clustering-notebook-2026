from pathlib import Path

ROOT = Path(".")
RESULTS = ROOT / "results"

BACKBONES = {
    "dinov3": {
        "kind": "timm",
        "model": "vit_large_patch16_dinov3.lvd1689m",
        "size": 224,
    },
    "radio": {
        "kind": "radio",
        "model": "nvidia/C-RADIOv4-SO400M",
        "size": 224,
    },
    "aimv2": {
        "kind": "timm",
        "model": "aimv2_large_patch14_224.apple_pt",
        "size": 224,
    },
    "siglip2": {
        "kind": "open_clip",
        "model": "timm/ViT-SO400M-16-SigLIP2-384",
        "size": 384,
    },
    "dinov2": {
        "kind": "timm",
        "model": "vit_large_patch14_reg4_dinov2.lvd142m",
        "size": 224,
    },
    "siglip": {
        "kind": "timm",
        "model": "vit_so400m_patch14_siglip_384",
        "size": 378,
    },
    "convnext": {
        "kind": "timm",
        "model": "convnextv2_large.fcmae_ft_in22k_in1k",
        "size": 224,
    },
    "eva02": {
        "kind": "timm",
        "model": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "size": 448,
    },
}

MODEL_PRESETS = {
    "fusion_8": tuple(BACKBONES),
    **{name: (name,) for name in BACKBONES},
}

ACTIVE_MODEL = "fusion_8"
# ACTIVE_MODEL = "dinov3"
# ACTIVE_MODEL = "radio"
# ACTIVE_MODEL = "aimv2"
# ACTIVE_MODEL = "siglip2"
# ACTIVE_MODEL = "dinov2"
# ACTIVE_MODEL = "siglip"
# ACTIVE_MODEL = "convnext"
# ACTIVE_MODEL = "eva02"

K = 12
SEED = 1
VIEW_DIM = 32
FUSION_DIM = 100
