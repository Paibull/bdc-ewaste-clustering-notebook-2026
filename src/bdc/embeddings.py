from pathlib import Path

import numpy as np
from PIL import Image

from .config import ACTIVE_MODEL, BACKBONES, MODEL_PRESETS
from .preprocessing import harmonize


class FrozenEmbeddingExtractor:
    def __init__(self, device=None):
        import torch

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _load(self, name):
        import timm
        from timm.data import create_transform, resolve_model_data_config

        item = BACKBONES[name]
        if item["kind"] == "radio":
            import torchvision.transforms as transforms
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                item["model"], trust_remote_code=True
            ).to(self.device).eval()
            transform = transforms.Compose([
                transforms.Resize(
                    (item["size"], item["size"]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ])
            forward = lambda current, batch: current(batch)[0]
        elif item["kind"] == "open_clip":
            import open_clip

            model, transform = open_clip.create_model_from_pretrained(
                f"hf-hub:{item['model']}"
            )
            model = model.to(self.device).eval()
            forward = lambda current, batch: current.encode_image(batch)
        else:
            options = {"dynamic_img_size": True} if item["model"].startswith("vit") else {}
            model = timm.create_model(
                item["model"], pretrained=True, num_classes=0, **options
            ).to(self.device).eval()
            config = resolve_model_data_config(model)
            config["input_size"] = (3, item["size"], item["size"])
            config["crop_pct"] = 1.0
            transform = create_transform(**config, is_training=False)
            forward = lambda current, batch: current(batch)
        return model, transform, forward

    def extract(self, paths, name, batch_size=64):
        torch = self.torch
        model, transform, forward = self._load(name)
        paths = [Path(path) for path in paths]
        output = []
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    image = harmonize(image.convert("RGB"))
                    batch.append(transform(image))
            batch = torch.stack(batch).to(self.device)
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                features = forward(model, batch).float()
            features = torch.nn.functional.normalize(features, dim=1)
            output.append(features.cpu().numpy())
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(output).astype(np.float16)

    def extract_selected(self, paths, preset=ACTIVE_MODEL, batch_size=64):
        return {
            name: self.extract(paths, name, batch_size)
            for name in MODEL_PRESETS[preset]
        }
