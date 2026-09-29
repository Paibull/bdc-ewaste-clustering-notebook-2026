from io import BytesIO

from PIL import Image

IMAGE_SIZE = 150
JPEG_QUALITY = 75
JPEG_SUBSAMPLING = 2


def harmonize(image, size=IMAGE_SIZE, jpeg=True):
    width, height = image.size
    side = min(width, height)
    if width != height:
        left = (width - side) // 2
        top = (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    if side != size:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    if jpeg:
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image
