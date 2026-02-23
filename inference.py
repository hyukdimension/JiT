import argparse
import os
import copy

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from util.config_loader import load_inference_config
import util.misc as misc

from diffusion.inference_diffusion import DiffusionSampler
from runners.inference_runner import run_inference


# --- 윈도우 단일 GPU 환경 dist 패치 ---
def _is_initialized(): return True
def _get_rank():        return 0
def _get_world_size():  return 1
def _barrier():         return None
def _all_reduce(x, **kwargs): return x
def _get_backend():     return 'gloo'

def setup_runtime_env():
    if not dist.is_initialized():
        dist.is_initialized = _is_initialized
        dist.get_rank       = _get_rank
        dist.get_world_size = _get_world_size
        dist.barrier        = _barrier
        dist.all_reduce     = _all_reduce
        dist.get_backend    = _get_backend
    misc.get_rank        = _get_rank
    misc.get_world_size  = _get_world_size
    misc.is_main_process = _is_initialized


def main(args):
    setup_runtime_env()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.output_dir) if misc.get_rank() == 0 else None

    # 모델 생성
    print(f"[*] Creating DiffusionSampler: {args.model}")
    denoiser_sampler = DiffusionSampler(args).to(device)

    # 체크포인트 로드 (EMA1 파라미터 우선 적용)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"[Error] 체크포인트를 찾을 수 없습니다: {args.checkpoint}")

    print(f"[*] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'model_ema1' in ckpt:
        # EMA1 파라미터를 sampler에 직접 로드
        print("[*] EMA1 파라미터 적용")
        base_state = ckpt.get('model', ckpt.get('model_base'))
        ema_state  = copy.deepcopy(base_state)
        e1         = ckpt['model_ema1']
        for name in ema_state:
            if name in e1:
                ema_state[name] = e1[name]
        denoiser_sampler.load_state_dict(ema_state)
    else:
        print("[*] EMA 없음 — 일반 파라미터 적용")
        denoiser_sampler.load_state_dict(ckpt.get('model', ckpt.get('model_base')))

    del ckpt
    torch.cuda.empty_cache()

    # 추론 실행
    print(f"\n{'='*60}")
    print(f"[*] Inference start")
    print(f"    Model      : {args.model}")
    print(f"    img_size   : {args.img_size}x{args.img_size}")
    print(f"    num_images : {args.num_images}")
    print(f"    method     : {args.sampling_method} / steps: {args.num_sampling_steps}")
    print(f"    cfg        : {args.cfg}")
    print(f"    Device     : {device}")
    print(f"{'='*60}\n")

    with torch.no_grad():
        run_inference(denoiser_sampler, args, epoch=0, log_writer=log_writer)

    print("[*] Inference finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('JiT inference')
    parser.add_argument('--common_config',    type=str, required=True)
    parser.add_argument('--inference_config', type=str, required=True)
    cli = parser.parse_args()

    args = load_inference_config(cli.common_config, cli.inference_config)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)