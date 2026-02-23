import math
import sys
import os
import shutil

import torch
import numpy as np
import cv2

import util.misc as misc
import util.lr_sched as lr_sched
import torch_fidelity
import copy


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):



        p_model = next(model.parameters())
        p_optim = optimizer.param_groups[0]['params'][0]
        print(f"Is same object?: {p_model is p_optim}") # 이게 False면 백날 돌려도 안 변합니다.



        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        # loss.backward() 바로 다음 줄
        print("\n" + "="*50)
        print("GRADIENT FLOW AUDIT")
        print("="*50)

        has_grad_count = 0
        no_grad_count = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_abs_mean = param.grad.abs().mean().item()
                    if grad_abs_mean > 0:
                        print(f"[OK] {name:<40} | Grad: {grad_abs_mean:.10f}")
                        has_grad_count += 1
                    else:
                        print(f"[ZERO] {name:<40} | Grad is EXACTLY 0.0000")
                        no_grad_count += 1
                else:
                    print(f"[!! NONE !!] {name:<40} | No Gradient at all")
                    no_grad_count += 1
            else:
                print(f"[FROZEN] {name:<40} | requires_grad=False")

        print("="*50)
        print(f"Summary: Flowing: {has_grad_count} | Broken/Zero: {no_grad_count}")
        print("="*50 + "\n")


        # loss.backward() 바로 다음 줄
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"Found Gradient in {name}! Mean: {param.grad.abs().mean().item()}")
                break
        else:
            print("STILL NONE: No parameters received gradients.")

        print(f"Does loss have grad_fn?: {loss.grad_fn}")

        # 모니터링 블록 내부
        first_param = next(model_without_ddp.net.parameters())
        if first_param.grad is not None:
            print(f"[Grad Check] Grad Abs Mean: {first_param.grad.abs().mean().item():.10f}")
        else:
            print("[Grad Check] Gradient is NONE!")

        optimizer.step()


        param = next(model.parameters())
        # 1. 가중치에 강제로 1.0을 더해버립니다 (무조건 변해야 함)
        param.add_(1.0) 
        print(f"!!! FORCED UPDATE TEST !!! Mean: {param.mean().item():.6f}")



        # 1. 모니터링 연산 (데이터 오염 및 스케일 확인)
        if data_iter_step % 100 == 0:
            with torch.no_grad():
                # 1. 특정 레이어(예: 첫 번째 컨볼루션)의 가중치 분포 확인
                # 'net'은 Denoiser 내부의 JiT 모델입니다. 
                # 레이어 이름은 모델 구조에 따라 다를 수 있으니 확인이 필요합니다.
                first_layer_w = next(model_without_ddp.net.parameters())
                
                w_max = first_layer_w.max().item()
                w_min = first_layer_w.min().item()
                w_std = first_layer_w.std().item()

                print(f"\n[Weight Check] Max: {w_max:.6f}, Min: {w_min:.6f}, Std: {w_std:.6f}")
                
                # 2. Gradient가 흐르고 있는지 확인 (학습 직후에만 유효)
                if first_layer_w.grad is not None:
                    g_std = first_layer_w.grad.std().item()
                    print(f"[Grad Check] Gradient Std: {g_std:.8f}")

        torch.cuda.synchronize()

        #model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps = args.num_images // (batch_size * world_size) + 1

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_num = args.class_num
    assert args.num_images % class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        # engine_jit.py 내 evaluate 함수 수정
        # ...
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            
            # [1, H, W] -> [H, W] 로 변환 (squeeze 사용)
            img_tensor = sampled_images[b_id].numpy().squeeze()




            # B 모델 전용: [-1, 1] -> [0, 255] 복원 로직
            img_rescaled = (img_tensor + 1.0) / 2.0
            # (img_tensor + 1.0) / 2.0  =>  [-1, 1]을 [0, 1]로 변환
            # 그 후 * 255 수행
            gen_img = np.round(np.clip(img_rescaled * 255, 0, 255)).astype(np.uint8)



            
            # 이제 G 모델과 동일한 스케일(0~255)에서 로그를 출력합니다.
            print(f"B model Final Image - Min: {gen_img.min():.4f}, Max: {gen_img.max():.4f}, Mean: {gen_img.mean():.4f}")
            
            # 이미지 저장
            cv2.imwrite(os.path.join(save_folder, f'{img_id:05d}.png'), gen_img)

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)

    # compute FID and IS
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder,
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
        #shutil.rmtree(save_folder)

    torch.distributed.barrier()