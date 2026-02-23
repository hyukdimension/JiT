import torch
import torch.nn as nn
from diffusion.diffusion_base import DiffusionBase


class DiffusionTrainer(DiffusionBase):
    """학습 전용 클래스.
    noise sampling, forward loss, EMA 업데이트를 담당.
    """

    def __init__(self, args):
        super().__init__(args)

        # 학습 전용 하이퍼파라미터
        self.label_drop_prob = args.label_drop_prob
        self.P_mean          = args.P_mean
        self.P_std           = args.P_std

        # EMA
        self.ema_decay1  = args.ema_decay1
        self.ema_decay2  = args.ema_decay2
        self.ema_params1 = None   # train.py에서 초기화
        self.ema_params2 = None   # train.py에서 초기화

    # ------------------------------------------------------------------
    # 학습 전용 메서드
    # ------------------------------------------------------------------
    def drop_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Classifier-free guidance를 위한 label drop"""
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def sample_t(self, n: int, device=None) -> torch.Tensor:
        """logit-normal 분포에서 timestep sampling"""
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """학습 loss 계산 (v-prediction + L2 loss)"""
        labels = self.drop_labels(labels)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        z      = t * x + (1 - t) * e
        v      = (x - z) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.net(z, t.flatten(), labels)
        v_pred = (x_pred - z) / (1 - t).clamp_min(self.t_eps)

        loss = (v - v_pred) ** 2
        return loss.mean(dim=(1, 2, 3)).mean()

    @torch.no_grad()
    def update_ema(self):
        """EMA 파라미터 업데이트"""
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)