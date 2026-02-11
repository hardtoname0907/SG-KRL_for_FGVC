#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, weight_decay, \
    proposalN, set, channels,gpu_ids,CUDA_VISIBLE_DEVICES
from utils.train_model import train
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet
from torch.nn import DataParallel

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def main():

    #加载数据
    trainloader, testloader = read_dataset(input_size, batch_size, root, set)

    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()

    #加载checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        start_epoch, lr = auto_load_resume(model, save_path, status='train')
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    # model = model.cuda()  # 部署在GPU
    # 多卡训练
    if len(gpu_ids) == 1:
        print(f"\tUse number {gpu_ids} GPU")
    else:
        print(f"\tUse number {gpu_ids} GPUs")
        model.to(torch.device("cuda"))
        model = DataParallel(model, device_ids=gpu_ids)  # 指定多GPU

    torch.cuda.set_device(gpu_ids[0])
    model.cuda(gpu_ids[0])  # 将模型移动到指定的CUDA设备
    
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    # 开始训练
    train(model=model,
          trainloader=trainloader,
          testloader=testloader,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          save_interval=save_interval)


if __name__ == '__main__':
    main()