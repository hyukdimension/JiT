import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from util.crop import center_crop_arr


def build_transform(img_size: int) -> transforms.Compose:
    """학습용 transform — center crop + random horizontal flip + tensor 변환"""
    return transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.PILToTensor(),
    ])

        #transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        #transforms.RandomHorizontalFlip(),
        #transforms.PILToTensor(),


class SingleFolderDataset(Dataset):
    """단일 폴더에서 TIFF 이미지만 로드하는 데이터셋"""

    VALID_EXTS = {'.tif', '.tiff'}

    def __init__(self, root: str, img_size: int, transform=None):
        self.transform = transform

        abs_root = os.path.abspath(root)
        if not os.path.exists(abs_root):
            raise ValueError(f"[Dataset Error] 디렉토리가 존재하지 않습니다: {abs_root}")

        self.samples = [
            os.path.join(abs_root, f)
            for f in sorted(os.listdir(abs_root))
            if os.path.splitext(f)[1].lower() in self.VALID_EXTS
        ]

        if len(self.samples) == 0:
            raise ValueError(f"[Dataset Error] TIFF 이미지가 없습니다: {abs_root}")

        # 이미지 크기 검증 — 모든 이미지가 img_size 이상이어야 함
        for path in self.samples:
            w, h = Image.open(path).size
            if h < img_size or w < img_size:
                raise ValueError(
                    f"[Dataset Error] 이미지 크기({w}x{h})가 img_size({img_size})보다 작습니다: {path}"
                )

        print(f"[Dataset] {len(self.samples)}개 TIFF 이미지 로드 완료: {abs_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert('L')  # 1채널 grayscale
        if self.transform:
            img = self.transform(img)
        # unconditional 생성이므로 label은 0으로 고정
        return img, 0