import torch
import torch.nn as nn
from diffusion.diffusion_base import DiffusionBase


class DiffusionSampler(DiffusionBase):
    """추론 전용 클래스.
    ODE sampling (Euler / Heun), CFG를 담당.
    """

    def __init__(self, args):
        super().__init__(args)

    # ------------------------------------------------------------------
    # 추론 전용 메서드
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """이미지 생성 메인 메서드"""
        device = labels.device
        bsz    = labels.size(0)

        z = self.noise_scale * torch.randn(
            bsz, self.in_channels, self.img_size, self.img_size, device=device
        )
        timesteps = (
            torch.linspace(0.0, 1.0, self.steps + 1, device=device)
            .view(-1, *([1] * z.ndim))
            .expand(-1, bsz, -1, -1, -1)
        )

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError(f"지원하지 않는 sampling method: {self.method}")

        # ODE 적분
        for i in range(self.steps - 1):
            z = stepper(z, timesteps[i], timesteps[i + 1], labels)

        # 마지막 스텝은 항상 euler
        z = self._euler_step(z, timesteps[-2], timesteps[-1], labels)
        return z

    @torch.no_grad()
    def _predict(self, z: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """CFG 적용 velocity 예측"""
        # conditional
        x_cond   = self.net(z, t.flatten(), labels)
        v_cond   = (x_cond - z) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        x_uncond = self.net(z, t.flatten(), torch.full_like(labels, self.num_classes))
        v_uncond = (x_uncond - z) / (1.0 - t).clamp_min(self.t_eps)

        # CFG interval masking
        low, high        = self.cfg_interval
        interval_mask    = (t < high) & ((low == 0) | (t > low))
        cfg_scale_masked = torch.where(interval_mask, self.cfg_scale, 1.0)

        return v_uncond + cfg_scale_masked * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        v_pred = self._predict(z, t, labels)
        return z + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        v_pred_t      = self._predict(z, t, labels)
        z_euler       = z + (t_next - t) * v_pred_t
        v_pred_t_next = self._predict(z_euler, t_next, labels)
        v_pred        = 0.5 * (v_pred_t + v_pred_t_next)
        return z + (t_next - t) * v_pred
