import argparse
import datetime
import numpy as np
import os
import time
import json
import copy
from pathlib import Path
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from util.crop import center_crop_arr
import util.misc as misc
import torch.distributed as dist

from engine_jit import train_one_epoch, evaluate
from denoiser import Denoiser

# --- [1] 윈도우 멀티프로세싱 호환 설정 (Pickle 에러 방지) ---
def _is_initialized(): return True
def _get_rank(): return 0
def _get_world_size(): return 1
def _barrier(): return None
def _all_reduce(x, **kwargs): return x
def _get_backend(): return 'gloo'

def setup_runtime_env():
    if not dist.is_initialized():
        dist.is_initialized = _is_initialized
        dist.get_rank = _get_rank
        dist.get_world_size = _get_world_size
        dist.barrier = _barrier
        dist.all_reduce = _all_reduce
        dist.get_backend = _get_backend
    misc.get_rank = _get_rank
    misc.get_world_size = _get_world_size
    misc.is_main_process = _is_initialized

GLOBAL_IMG_SIZE = 256
def center_crop_transform(img):
    return center_crop_arr(img, GLOBAL_IMG_SIZE)

# --- [2] Single Folder Dataset (TIFF 전용) ---
class SingleFolderDataset(torch.utils.data.Dataset):
    """단일 폴더에서 TIFF 이미지만 로드하는 데이터셋"""
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        
        abs_root = os.path.abspath(root)
        if not os.path.exists(abs_root):
            raise ValueError(f"[Error] Directory does not exist: {abs_root}")
        
        valid_exts = {'.tif', '.tiff'}
        
        # 단일 폴더에서 TIFF 이미지만 수집
        for f in sorted(os.listdir(abs_root)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                self.samples.append(os.path.join(abs_root, f))
        
        if len(self.samples) == 0:
            raise ValueError(f"[Error] No TIFF images found in {abs_root}")
        
        print(f"[*] Loaded {len(self.samples)} TIFF images from {abs_root}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # label은 0으로 통일 (unconditional 생성)
        return img, 0
	
# --- [3] Argument Parser ---
def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)
    parser.add_argument('--config', default='', type=str)
    
    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0, help='Projection dropout rate')

    # Training Hyperparameters
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
	
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--eval_freq', type=int, default=40,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate_gen', action='store_true')
    parser.add_argument('--gen_bsz', type=int, default=256,
                        help='Generation batch size')

    # dataset
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='Path to the dataset')
    parser.add_argument('--class_num', default=1, type=int)  # 200 → 1로 변경

    # checkpointing
    parser.add_argument('--output_dir', default='./output_dir',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=5,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')
    parser.add_argument('--local_rank', default=-1, type=int)

    return parser


def main(args):
    setup_runtime_env()
    global GLOBAL_IMG_SIZE
    GLOBAL_IMG_SIZE = args.img_size

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank() # <-- todo: 1) 재현성 최우선 고려. 2) 바꿔서 넣어보기
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True # <-- todo: 의미 알기. deterministic 관련. 조합 고려 (씨드 관련)

    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # evaluate_gen일 때는 데이터셋 로드 안 함
    print("args.evaluate_gen")
    print(args.evaluate_gen)
    if not args.evaluate_gen:
        print(f"[*] Loading dataset from: {args.data_path}")
        transform_train = transforms.Compose([
            transforms.Lambda(center_crop_transform),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor() 
        ])
        
        dataset_train = SingleFolderDataset(args.data_path, transform=transform_train)
        
        print(f"[*] Dataset size: {len(dataset_train)} images")
        print(f"[*] Batch size: {args.batch_size}")
        import math
        num_batches = math.ceil(len(dataset_train) / args.batch_size)
        print(f"[*] Number of batches per epoch: {num_batches}")
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, 
            sampler=torch.utils.data.RandomSampler(dataset_train),
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            pin_memory=args.pin_mem, 
            drop_last=False
        )
    else:
        print("[*] Inference mode: skipping dataset loading")
        data_loader_train = None

    # Create denoiser
    print(f"[*] Creating model: {args.model}")
    model = Denoiser(args).to(device)
    model_without_ddp = model

    # 2. Optimizer 설정
    if args.lr is None:
        args.lr = args.blr * (args.batch_size * misc.get_world_size()) / 256
    print(f"[*] Learning rate: {args.lr:.6f}")
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95)) # <-- 베타 의미.
    print(optimizer)

    # 3. EMA 초기화
    print("[*] Initializing EMA parameters")
    model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
    model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))

    # Resume from checkpoint if provided
    if args.resume: 
        checkpoint_path = args.resume
        if os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, "checkpoint-last.pth")
        if os.path.exists(checkpoint_path):
            print(f"[*] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint.get('model', checkpoint.get('model_base')))
            
            if 'model_ema1' in checkpoint and 'model_ema2' in checkpoint:
                e1, e2 = checkpoint['model_ema1'], checkpoint['model_ema2']
                model_without_ddp.ema_params1 = [e1[n].to(device) for n, _ in model_without_ddp.named_parameters()]
                model_without_ddp.ema_params2 = [e2[n].to(device) for n, _ in model_without_ddp.named_parameters()]
            
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint.get('epoch', -1) + 1
            del checkpoint
            torch.cuda.empty_cache()
            print(f"[*] Resumed from epoch {args.start_epoch}")

    if args.evaluate_gen:
        evaluate(model_without_ddp, args, args.start_epoch, batch_size=args.gen_bsz, log_writer=log_writer)
        return
	
    

    # 6. 학습 시작
    print(f"\n{'='*60}")
    print(f"[*] Training started: Epoch {args.start_epoch} to {args.epochs}")
    print(f"    - Model: {args.model}")
    print(f"    - Image size: {args.img_size}x{args.img_size}")
    print(f"    - Class num: {args.class_num}")
    print(f"    - Device: {device}")
    print(f"{'='*60}\n")
    
    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        
        
        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)
        
        # 저장 로직
        if (epoch + 1) % args.save_last_freq == 0 or (epoch + 1) == args.epochs:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name="last")
        if (epoch + 1) % 100 == 0:
            misc.save_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, epoch_name=None)

        # Perform online evaluation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            torch.cuda.empty_cache()
            with torch.no_grad():
                evaluate(model_without_ddp, args, epoch, batch_size=args.gen_bsz, log_writer=log_writer)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
            
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[*] Training finished!")
    print(f"    - Total time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for k, v in config_dict.items():
                if hasattr(args, k):
                    val = float(v) if k in ['lr', 'blr', 'weight_decay', 'interval_min', 'interval_max'] and isinstance(v, str) else v
                    setattr(args, k, val)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)