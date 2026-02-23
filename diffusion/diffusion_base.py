import torch
import torch.nn as nn
from models.model import JiT_models


class DiffusionBase(nn.Module):
    """학습/추론 공용 base 클래스.
    모델 구조, diffusion 공용 하이퍼파라미터를 보유.
    """

    def __init__(self, args):
        super().__init__()

        # 모델 구조
        self.net = JiT_models[args.model](
            input_size=args.img_size,
            in_channels=args.in_channels,
            num_classes=args.class_num,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
        )

        # 공용 diffusion 하이퍼파라미터
        self.img_size    = args.img_size
        self.in_channels = args.in_channels
        self.num_classes = args.class_num
        self.t_eps       = args.t_eps
        self.noise_scale = args.noise_scale

        # 공용 sampling 하이퍼파라미터 (학습 중 online_eval에서도 사용)
        self.cfg_scale    = args.cfg
        self.cfg_interval = (args.interval_min, args.interval_max)
        self.method       = args.sampling_method
        self.steps        = args.num_sampling_steps
