import argparse
import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch
from adem.build import create_model
from util.coco_karpathy_dataset import coco_karpathy_train
from util.datasets import InstrcutDataSet, ScienceQADataSet
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='sqa')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--clip', type=str, default='ViT-L/14')
    parser.add_argument('--clip_root', type=str, default='./clip')
    parser.add_argument('--llm_model', type=str, default='7B')
    parser.add_argument('--output_dir', type=str, default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', type=str, default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE', 'Q-A', 'QM-A', 'Q-AL', 'QM-EA'
                        ],
                        help='prompt format template')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--caption_file', type=str, default='./data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # model
    parser.add_argument('--adapter_dim', type=int, default=8, help='the dims of adapter layer')
    parser.add_argument('--hidden_proj', type=int, default=128,
                        help='the visual adapter dim')
    parser.add_argument('--max_seq_len', type=int, default=512, help='the maximum sequence length')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--down_sample_num', type=int, nargs='+', default=[256, 64])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--drop_ratio', type=float, default=0.1)
    parser.add_argument('--no_cls', action='store_true')

    # optim
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=float, default=2,
                        help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='clips gradient norm of an iterable of parameters')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--cpu_load', action='store_true', help='load the model on cpu and avoid OOM on gpu')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='saving memory costs via gradient_checkpointing')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    random.seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    args.is_train = True

    llama_model_path = os.path.join(args.data_root, "weights/")
    if args.dataset == 'sqa':
        dataset_train = ScienceQADataSet(args, 'train', llama_model_path, args.max_seq_len)
    elif args.dataset == 'coco_caption':
        dataset_train = coco_karpathy_train(image_root=os.path.join(args.data_root, 'images'),
                                            ann_root=os.path.join(args.data_root, 'coco_caption'),
                                            model_root=llama_model_path,
                                            prompt='a picture of ')
    elif args.dataset == 'instruction':
        dataset_train = InstrcutDataSet(args, 'all', llama_model_path, args.max_seq_len)
    else:
        raise RuntimeError

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        generator=g,
    )

    # define the model
    model = create_model(args)
    model.to(device)

    # for debug.   print the data type.
    # for name, param in model.named_parameters():
    #     print(name, param.dtype)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    # mixed precision scaler
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
