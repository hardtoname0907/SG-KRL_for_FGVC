#coding=utf-8
"""
本测试适用于datasets路径下的数据集的批量测试
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels
from utils.read_dataset import read_dataset
from utils.auto_laod_resume import auto_load_resume
from networks.model import MainNet

CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

# dataset
set = 'LUX'
sub_cls = 'maotai'  # 'lingbiao' or 'maotai'
if set == 'CUB':
    root = './datasets/CUB_200_2011'  # dataset path
    # model path
    pth_path = "./checkpoint/cub/gcn_qkv101_mask1/best.pth"
    num_classes = 200
elif set == 'LUX':
    if sub_cls == 'lingbiao':
        root = '/data/user/zhuyueran/adv/attacks/MIMatt/out/res101/lingbiao' #'./datasets/gucci_lingbiao'
        pth_path = "./checkpoint/Luxury/gucci1202_1/best.pth"
        num_classes = 2
        batch_size = 10
    elif sub_cls == 'WJ':
        root = '/data/user/zhuyueran/adv/attacks/MIMatt/out/res101/wujin' # '/data/user/zhuyueran/datasets/LV55/wujin'
        pth_path = "/data/user/zhuyueran/models/GCN-MMAL2/checkpoint/Luxury/lvwujin_1/best.pth"
        num_classes = 8
        batch_size = 10
    elif sub_cls == 'YB':
        root = '/data/user/zhuyueran/datasets/LV55/yuanbiao'
        pth_path = "/data/user/zhuyueran/models/GCN-MMAL2/checkpoint/Luxury/lvyuanbiao_1/best.pth"
        num_classes = 6
        batch_size = 10
    elif sub_cls == 'maotai':
        root = '/data/user/zhuyueran/datasets/maotai/250812_latest_crop_fsp11' # './datasets/maotai_jiu'
        pth_path = "/data/user/zhuyueran/models/GCN-MMAL2/checkpoint/Luxury/t0812_jiuall_maug2gs/best.pth"
        num_classes = 8 # 领标为2，茅台为8
        batch_size = 1 # 茅台为4，领标为10
    elif sub_cls == 'cartier':
        root = '/data/user/zhuyueran/datasets/cartier_textsp'
        pth_path= "/data/user/zhuyueran/models/GCN-MMAL2/checkpoint/Luxury/cartier0506/best.pth"
        num_classes = 2 # 领标为2，茅台为8
        batch_size = 10 # 茅台为4，领标为10
    elif sub_cls == 'airsp':
        root = '/data/public/FGVC_Aircraft'
        num_classes = 100
        pth_path='/data/user/zhuyueran/models/GCN-MMAL2/checkpoint/aircraft/gcn_qkv50_mask/best.pth'
        batch_size = 7
#load dataset
_, testloader = read_dataset(input_size, batch_size, root, set)

# 定义模型
model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()

#加载checkpoint
if os.path.exists(pth_path):
    epoch = auto_load_resume(model, pth_path, status='test')
    print(f'Load {epoch}E pth successfully')
else:
    sys.exit('There is not a pth exist.')

print('Testing')
raw_correct = 0
object_correct = 0
gcn_correct = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(testloader)):
        if set == 'CUB':
            x, y, boxes, _ = data
        else:
            x, y = data
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        # # 获取当前实际 batch_size，测试最后1batch被drop的情况————已测试96.98%
        # current_batch_size = x.size(0)  # PyTorch Tensor
        # if current_batch_size < batch_size:
        #     break
        local_logits, local_imgs,gcn_logits = model(x, epoch, i, 'test', DEVICE)[-3:]
        # local
        pred = local_logits.max(1, keepdim=True)[1]
        object_correct += pred.eq(y.view_as(pred)).sum().item()
        # gcn
        gcn_pred = gcn_logits.max(1, keepdim=True)[1]
        gcn_correct += gcn_pred.eq(y.view_as(gcn_pred)).sum().item()

    print('\nObject branch accuracy: {}/{} ({:.2f}%)\n'.format(
            object_correct, len(testloader.dataset), 100. * object_correct / len(testloader.dataset)))
    print('\nGCN branch accuracy: {}/{} ({:.2f}%)\n'.format(
            gcn_correct, len(testloader.dataset), 100. * gcn_correct / len(testloader.dataset)))