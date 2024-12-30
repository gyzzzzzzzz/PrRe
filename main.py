
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn


from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from dataloader.datasets import bulid_fscil_dataloader,bulid_domain_dataloader
from engine import *
import models
import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args,config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    #在每次卷积操作时，CUDNN 会在一系列可用的卷积实现中寻找最快的算法，并且在内存消耗与算法速度之间作出权衡选择

    if config.startswith("fscil"):
        data_loader, class_mask = bulid_fscil_dataloader(args)
    elif config.startswith("domain"):
        data_loader, class_mask = bulid_domain_dataloader(args)
    else:
        raise ValueError('Dataset not found.')
    
    print(f"Creating original model: {args.model}")
    original_model = create_model( 
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    print(f"Creating model: {args.model}")
    model = new_func(args)
    original_model.to(device)
    model.to(device)

    if args.freeze:
        # 冻结整个预训练模型的参数，使其不可训练
         for p in original_model.parameters():
            p.requires_grad = False
        
        # 冻结指定部分（blocks, patch_embed, cls_token）的参数，使其不可训练
         for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):  # 检查参数名是否以指定的标志开头
                p.requires_grad = False
    
    print(args)

    if args.eval:   # 检查是否需要进行模型评估
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  # 创建一个零矩阵用于存储模型评估结果
        # 构建任务特定的checkpoint路径
        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            if os.path.exists(checkpoint_path):# 如果任务特定的checkpoint存在
                print('Loading checkpoint from:', checkpoint_path) # 打印加载checkpoint的信息
                checkpoint = torch.load(checkpoint_path)# 从checkpoint中加载模型参数
                model.load_state_dict(checkpoint['model'])# 将加载的模型参数设置到模型中
            else:
                print('No checkpoint found at:', checkpoint_path)# 如果checkpoint不存在，打印未找到checkpoint的信息
                return
             # 执行模型评估
            _ = evaluate_till_now(model, original_model, data_loader, device, 
                                            task_id, class_mask, acc_matrix, args,)
        
        return
    #它们指向相同的内存地址
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)#模型总共的可训练参数数量
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    print('number of params:', n_parameters)

    if args.unscale_lr: # 检查是否使用非缩放的学习率
        global_batch_size = args.batch_size # 如果是，则使用本地批量大小作为全局批量大小
    else:
        global_batch_size = args.batch_size * args.world_size# 如果未使用非缩放学习率，则计算全局批大小
    args.lr = args.lr * global_batch_size / 256.0# 根据全局批大小调整学习率

    optimizer = create_optimizer(args, model_without_ddp)# 使用给定参数和模型创建优化器
    
    if args.sched != 'constant':  # 如果传入的参数args中的sched不等于'constant'
        lr_scheduler, _ = create_scheduler(args, optimizer)  # 调用create_scheduler函数创建学习率调度器和一个未命名的对象
    elif args.sched == 'constant':  # 如果传入的参数args中的sched等于'constant'
        lr_scheduler = None  # 设置学习率调度器为None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    # 调用train_and_evaluate函数，用神经网络模型进行训练和评估
    acc_matrix = train_and_evaluate(model, model_without_ddp, original_model,
                    criterion, data_loader, optimizer, lr_scheduler,
                    device, class_mask, args)

    total_time = time.time() - start_time   # 计算总训练时间
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))   # 将总训练时间转化为可读性更好的时间格式
    print(f"Total training time: {total_time_str}")  # 打印总训练时间

def new_func(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        f_prompt_length=args.length, 
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        usf_prompt_mask=args.usf_prompt_mask,
        use_d_prompt=args.use_d_prompt,
        d_prompt_length=args.d_prompt_length,
        d_prompt_layer_idx=args.d_prompt_layer_idx,
        use_prefix_tune_for_d_prompt=args.use_prefix_tune_for_d_prompt,
        use_f_prompt=args.use_f_prompt,
        f_prompt_layer_idx=args.f_prompt_layer_idx,
        use_prefix_tune_for_f_prompt=args.use_prefix_tune_for_f_prompt,
        same_key_value=args.same_key_value,
    )
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prompt training and evaluation configs')

    config = parser.parse_known_args()[-1][0] 

    subparser = parser.add_subparsers(dest='subparser_name')
    # 根据config选择对应的参数解析器
    if config == 'fscil_cifar100':
        from configs.fscil_cifar100 import get_args_parser
        config_parser = subparser.add_parser('fscil_cifar100', help='FSCIL configs cifar100')
    elif config == 'fscil_cub200':
        from configs.fscil_cub200 import get_args_parser
        config_parser = subparser.add_parser('fscil_cub200', help='FSCIL configs cub200')
    elif config == 'fscil_miniImageNet':
        from configs.fscil_miniImageNet import get_args_parser
        config_parser = subparser.add_parser('fscil_miniImageNet', help='FSCIL configs miniImageNet')
    elif config == 'domain_cifar10':
        from configs.domain_cifar10 import get_args_parser
        config_parser = subparser.add_parser('domain_cifar10', help='domain_cifar10 configs')
    elif config == 'domain_STL10':
        from configs.domain_STL10 import get_args_parser
        config_parser = subparser.add_parser('domain_STL10', help='domain_STL10')
    elif config == 'domain_Caltech256':
        from configs.domain_Caltech256 import get_args_parser
        config_parser = subparser.add_parser('domain_Caltech256', help='domain_Caltech256')
    elif config == 'domain_Flower102':
        from configs.domain_Flower102 import get_args_parser
        config_parser = subparser.add_parser('domain_Flower102', help='domain_Flower102')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser) 

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
        main(args,config)
    
    sys.exit(0)