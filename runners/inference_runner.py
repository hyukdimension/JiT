import os
import copy

import torch
import numpy as np
import cv2

import util.misc as misc
import torch_fidelity
from runners.train_runner import denormalize


def run_inference(
    denoiser_sampler,
    args,
    epoch=0,
    log_writer=None
):
    """이미지 생성 및 FID/IS 계산"""
    denoiser_sampler.eval()

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    num_steps  = args.num_images // (args.gen_bsz * world_size) + 1

    # 저장 폴더
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            denoiser_sampler.method,
            denoiser_sampler.steps,
            denoiser_sampler.cfg_scale,
            denoiser_sampler.cfg_interval[0],
            denoiser_sampler.cfg_interval[1],
            args.num_images,
            args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # class label 생성
    class_num             = args.class_num
    assert args.num_images % class_num == 0, "num_images는 class_num의 배수여야 합니다"
    class_label_gen_world = np.arange(0, class_num).repeat(args.num_images // class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print("Generation step {}/{}".format(i, num_steps))

        start_idx  = world_size * args.gen_bsz * i + local_rank * args.gen_bsz
        end_idx    = start_idx + args.gen_bsz
        labels_gen = torch.Tensor(class_label_gen_world[start_idx:end_idx]).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = denoiser_sampler.sample(labels_gen)

        torch.distributed.barrier()

        # denormalize: [-1,1] → [0,1]
        sampled_images = denormalize(sampled_images).detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id  = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            # 3채널: BGR 변환, 1채널: 그대로
            gen_img = gen_img.astype(np.uint8)
            if gen_img.shape[2] == 3:
                gen_img = gen_img[:, :, ::-1]
            print(f"Raw Tensor - Min: {gen_img.min():.4f}, Max: {gen_img.max():.4f}, Mean: {gen_img.mean():.4f}")
            cv2.imwrite(os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5))), gen_img)

    torch.distributed.barrier()

    # FID / IS 계산
    if log_writer is not None:
        if args.img_size == 256:
            fid_statistics_file = 'fid_stats/jit_in256_stats.npz'
        elif args.img_size == 512:
            fid_statistics_file = 'fid_stats/jit_in512_stats.npz'
        else:
            raise NotImplementedError(f"지원하지 않는 img_size: {args.img_size}")

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
        fid             = metrics_dict['frechet_inception_distance']
        inception_score = metrics_dict['inception_score_mean']
        postfix         = "_cfg{}_res{}".format(denoiser_sampler.cfg_scale, args.img_size)
        log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
        log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
        print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))

    torch.distributed.barrier()
