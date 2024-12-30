
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from argparse import Namespace
import utils


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)  # 将模型设置为训练模式
    original_model.eval()  # 将原始模型设置为评估模式
     
    if args.distributed and utils.get_world_size() > 1: 
        data_loader.sampler.set_epoch(epoch)    # 对于分布式训练，设置数据加载器的epoch

    metric_logger = utils.MetricLogger(delimiter="  ")     # 创建度量记录器用于跟踪训练统计信息
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加学习率的度量
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))   # 添加损失的度量
    header = f'Task:{task_id+1} Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'   # 创建进度显示的标题

   
    # 遍历数据加载器，并在指定频率下记录度量
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)  # 将输入数据移动到设备上
        target = target.to(device, non_blocking=True)   # 将目标数据移动到设备上

        with torch.no_grad():
            if original_model is not None:
                output,_,_,_ = original_model(input)
                cls_features = output['pre_logits'] 
            else:
                cls_features = None
        # output:key['similarity','prompt_idx','selected_key','prompt_key_norm','x_embed_norm','reduce_sim','batched_prompt', 'x', 'pre_logits', 'logits'])
        # 模型进行前向传播
        output,d_prompt,f_prompt,pos_embed= model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        k = output['selected_key'] 
        logits = output['logits'] 
        total_reg_loss = 0
        for i in range(3):
            f_prompt_slice=f_prompt[i]
            product = torch.matmul(d_prompt,f_prompt_slice.permute(0, 1, 2, 4,3))
            reg_loss = torch.norm(product, p='fro')
            total_reg_loss += reg_loss
        alpha=0.001
        # 在训练模型时，根据给定的类别掩码，将特定任务不关联的类别的逻辑输出（logits）修改为非常小的值（负无穷），从而在后续的softmax操作和损失计算中，实际上忽略这些类别
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask) 
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf')) 
        # 计算损失（使用交叉熵损失函数）
        loss_ce = criterion(logits, target) # base criterion (CrossEntropyLoss)   # 应用拉动约束（如果指定）
        def euclidean_distance(a, b):
            return ((a - b)**2).sum().sqrt()
        loss_dist=euclidean_distance(k,cls_features)
        𝜆 = 0.005 # 你需要根据问题来设置合适的权重
        total_loss = loss_ce+𝜆*loss_dist+alpha*total_reg_loss
        if args.pull_constraint and 'reduce_sim' in output:
            total_loss = total_loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))   # 计算top-1和top-5准确率

        if not math.isfinite(total_loss.item()):
            print("Loss is {}, stopping training".format(total_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()   # 梯度清零
        total_loss.backward()     # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)   # 对梯度进行裁剪
        optimizer.step()  # 更新模型参
        torch.cuda.synchronize()   # 同步CUDA流
        metric_logger.update(Loss=total_loss.item())   # 更新损失
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])  #更新学习率
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])   # 更新top-1准确率
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])   # 更新top-5准确率
        
     # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)  # 打印平均统计信息
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}  # 返回全局平均统计信息

@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):  # 打印平均统计信息
    criterion = torch.nn.CrossEntropyLoss()   # 返回全局平均统计信息

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad() 
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) 

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    
    print(result_str)

    return test_stats
# 定义了一个 train_and_evaluate 的函数，用于训练和评估模型
def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # 创建矩阵以保存任务结束时的准确度 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))  
    all_logits =None
    all_targets =None
    prototype = utils.ClassPrototypes(num_class=args.nb_classes,dim=args.nb_classes,device=device) 
    
    #创建一个用于存储类别原型的对象，并且指定了类别的数量、维度和计算设备
         # 循环每个任务
    for task_id in range(args.num_tasks): 
        
        # Transfer previous learned prompt params to the new prompt
         # 如果当前任务需要共享先前学习的提示参数到新的提示
        if args.prompt_pool and args.shared_prompt_pool and args.use_f_prompt:   # 进行一些条件检查后，将先前学习的提示参数传递到新的提示中
            if task_id > 0: 
                prev_start = (task_id - 1) * args.top_k 
                prev_end = task_id * args.top_k 

                cur_start = prev_end 
                cur_end = (task_id + 1) * args.top_k 

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_f_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad(): 
                        if args.distributed:
                            model.module.f_prompt.prompt.grad.zero_()
                            model.module.f_prompt.prompt[cur_idx] = model.module.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.f_prompt.prompt.grad.zero_()
                            model.f_prompt.prompt[cur_idx] = model.f_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        #如果当前任务需要共享先前学习的提示参数键到新的提示
        if args.prompt_pool and args.shared_prompt_key and args.use_f_prompt: # 进行一些条件检查后，将先前学习的提示参数键传递到新的提示中
            if task_id > 0: 
                with torch.no_grad():
                    if args.distributed:
                        model.module.f_prompt.prompt_key.grad.zero_()
                        model.module.f_prompt.prompt_key[cur_idx] = model.module.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.f_prompt.prompt_key.grad.zero_()
                        model.f_prompt.prompt_key[cur_idx] = model.f_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()

        # 如果需要为每个任务创建新的优化器以清除优化器状态
        if task_id > 0 and args.reinit_optimizer: # 如果条件满足，则创建一个新的优化器
            optimizer = create_optimizer(args, model) 
        # 如果需要更新任务的训练轮次
        if task_id > 0:
            try: 
                args.epochs = args.inc_epochs
            except:
                pass

        # 循环每个任务的训练轮次
        for epoch in range(args.epochs):       # 对模型进行一轮训练     
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            # 如果存在学习率调度器，则调整学习率
            if lr_scheduler:
                lr_scheduler.step(epoch)
       
        logits,targets=utils.logits_and_targets(model=model,original_model=original_model,
                                data_loader=data_loader[task_id]['train'],task_id=task_id,device=device,)
        if task_id==0:
            all_logits=logits
            all_targets=targets
        else:
        # 之后的迭代，将新结果插入到all_logits和all_targets的头部
            all_logits = torch.cat((all_logits, logits), dim=0)
            all_targets = torch.cat((all_targets, targets), dim=0)
        # Prototype Variable
        
        # 更新任务的原型并计算准确度矩阵
        prototype = utils.prototype_updata(prototype=prototype, model=model,original_model=original_model,device=device, data_loader=data_loader[task_id]['train'],task_id=task_id,)
        acc_matrix = utils.prototype_evaluate_till_now(prototype=prototype,all_logits=all_logits,all_targets=all_targets, model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args,)
        
        # MLP Classifier
        # test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
        #                             task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)

        # if args.output_dir and utils.is_main_process():
        #     Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

        #     checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            
        #     args_to_save = Namespace(**{k: v for k, v in vars(args).items() if k != "Dataset"}) 
        #     state_dict = {
        #             'model': model_without_ddp.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'epoch': epoch,
        #             'args': args_to_save,
        #         }
        #     if args.sched is not None and args.sched != 'constant': 
        #         state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
        #     utils.save_on_master(state_dict, checkpoint_path)

        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
        #     **{f'test_{k}': v for k, v in test_stats.items()},
        #     'epoch': epoch,}

        # if args.output_dir and utils.is_main_process():
        #     with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
        #         f.write(json.dumps(log_stats) + '\n')

     # 计算每个任务的平均准确度
    column_mean_values = []
    for i in range(args.num_tasks):
        selected_elements = acc_matrix[:i+1, i]
        mean_value = np.mean(selected_elements)
        column_mean_values.append(mean_value)
# 打印准确度矩阵和每个任务的平均准确度值
    print(acc_matrix)
    print(column_mean_values)
    
    return acc_matrix
