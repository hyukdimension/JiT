import numpy as np
from PIL import Image


def center_crop_arr(pil_image, crop_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * crop_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = crop_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - crop_size) // 2
    crop_x = (arr.shape[1] - crop_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + crop_size, crop_x: crop_x + crop_size])


def random_crop_arr(pil_image, crop_size=256):
    """
    512x512 이미지에서 crop_size x crop_size 영역을 랜덤하게 크롭.
    유효 영역(0 ~ image_size - crop_size) 내에서만 시작점을 선택.
    """
    arr = np.array(pil_image)          # (H, W) or (H, W, C)
    h, w = arr.shape[:2]

    # 시작점 범위: [0, h - crop_size] 이므로 끝점은 항상 이미지 내부
    max_y = h - crop_size
    max_x = w - crop_size

    assert max_y >= 0 and max_x >= 0, (
        f"이미지 크기({h}x{w})가 crop_size({crop_size})보다 작습니다."
    )

    crop_y = np.random.randint(0, max_y + 1)  # 0 ~ max_y 포함
    crop_x = np.random.randint(0, max_x + 1)

    cropped = arr[crop_y : crop_y + crop_size,
                  crop_x : crop_x + crop_size]

    return Image.fromarray(cropped)