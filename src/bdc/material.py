import math

import numpy as np

MATERIALS = (
    "glass",
    "metal",
    "paper_cardboard",
    "plastic",
    "rubber",
    "other_unknown",
)
MATERIAL_MAP = {
    "Glass": "glass",
    "Mirror": "glass",
    "Metal": "metal",
    "Paper": "paper_cardboard",
    "Cardboard": "paper_cardboard",
    "Plastic, clear": "plastic",
    "Plastic, non-clear": "plastic",
    "Rubber/latex": "rubber",
}


class MaterialAnalyzer:
    def __init__(self, weights, class_names, device=None):
        import torch

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = torch.jit.load(str(weights), map_location="cpu")
        self.model = self.model.eval().to(self.device)
        self.groups = np.asarray([
            MATERIALS.index(MATERIAL_MAP.get(name, "other_unknown"))
            for name in class_names
        ])

    def _input(self, image):
        from PIL import Image

        width, height = image.size
        scale = min(512 / height, 512 / width)
        size = (math.ceil(width * scale), math.ceil(height * scale))
        array = np.asarray(
            image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
        tensor = self.torch.from_numpy(
            array.transpose(2, 0, 1).copy()
        ).float().unsqueeze(0).to(self.device)
        mean = self.torch.tensor(
            [123.675, 116.28, 103.53], device=self.device
        ).view(1, 3, 1, 1)
        std = self.torch.tensor(
            [58.395, 57.12, 57.375], device=self.device
        ).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def predict(self, image):
        with self.torch.inference_mode():
            output = self.model(self._input(image))
        prediction = output[0] if isinstance(output, (tuple, list)) else output
        if prediction.ndim == 4 and prediction.shape[1] == len(self.groups):
            prediction = prediction.argmax(dim=1)
        elif prediction.ndim == 4 and prediction.shape[-1] == len(self.groups):
            prediction = prediction.argmax(dim=-1)
        elif prediction.ndim == 4 and prediction.shape[1] == 1:
            prediction = prediction[:, 0]
        if prediction.ndim == 3:
            prediction = prediction[0]
        return prediction.detach().cpu().numpy().astype(np.int64)

    def mask(self, annotation, shape):
        height, width = shape
        roi = np.zeros((height, width), dtype=bool)
        for line in annotation.splitlines():
            parts = line.split()
            if len(parts) == 5:
                _, xc, yc, box_width, box_height = parts
                xc, yc, box_width, box_height = map(
                    float, (xc, yc, box_width, box_height)
                )
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                coordinates = list(map(float, parts[1:]))
                xs, ys = coordinates[0::2], coordinates[1::2]
                xc = (min(xs) + max(xs)) / 2
                yc = (min(ys) + max(ys)) / 2
                box_width = max(xs) - min(xs)
                box_height = max(ys) - min(ys)
            else:
                continue
            left = max(0, min(width, math.floor((xc - box_width / 2) * width)))
            right = max(0, min(width, math.ceil((xc + box_width / 2) * width)))
            top = max(0, min(height, math.floor((yc - box_height / 2) * height)))
            bottom = max(0, min(height, math.ceil((yc + box_height / 2) * height)))
            roi[top:bottom, left:right] = True
        return roi

    def profile(self, prediction, roi=None):
        values = prediction if roi is None else prediction[roi]
        groups = self.groups[values]
        fractions = np.bincount(groups.ravel(), minlength=len(MATERIALS)).astype(float)
        fractions /= fractions.sum()
        return dict(zip(MATERIALS, fractions.tolist()))

    def analyze(self, image, annotation=None):
        prediction = self.predict(image)
        roi = self.mask(annotation, prediction.shape) if annotation else None
        return prediction, self.profile(prediction, roi)
