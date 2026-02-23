import math
import sys
import copy

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched

# 정규화 상수 — train_runner/inference_runner 양쪽이 이 값을 참조
PIXEL_MEAN = 0.5   # (x / 255) * 2 - 1  →  [-1, 1]
PIXEL_STD  = 0.5


def normalize(x: torch.Tensor) -> torch.Tensor:
    """uint8 tensor [0,255] → float32 [-1, 1]"""
    return x.to(torch.float32).div_(255).mul_(2.0).sub_(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """float32 [-1, 1] → [0, 1]"""
    return (x + 1.0) / 2.0


def apply_ema_to_sampler(denoiser_trainer, denoiser_sampler):
    """Trainer의 EMA1 파라미터를 Sampler net에 적용.
    평가 시 호출하고, 평가 후 원복은 호출자가 책임진다.
    """
    ema_state = {}
    for i, (name, _) in enumerate(denoiser_trainer.named_parameters()):
        ema_state[name] = denoiser_trainer.ema_params1[i]
    denoiser_sampler.load_state_dict(ema_state)


def train_one_epoch(
    denoiser_trainer,
    data_loader,
    optimizer,
    device,
    epoch,
    log_writer=None,
    args=None
):
    denoiser_trainer.train(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header     = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # iteration 단위 lr 스케줄
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x      = normalize(x.to(device, non_blocking=True))
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = denoiser_trainer(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        denoiser_trainer.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)