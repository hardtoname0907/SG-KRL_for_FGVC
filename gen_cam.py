import os
print(os.getcwd())  # 打印当前工作目录
# os.chdir('path_to_directory')  # 更改工作目录，如果需要的话
os.chdir('/data/user/zhuyueran/models/SG-KRL_for19_0403') # 需要更改工作目录，绝对路径
print(os.getcwd())
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from networks.model import MainNet
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)  # C*H*W --> B*C*H*W
    return img


mean: list = [0.485, 0.456, 0.406]  # ImageNet中的均值和标准差
std: list = [0.229, 0.224, 0.225]


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # 反归一化操作
        return tensor


def fill_image(img, scale=448, crop_ratio=(0.1, 0.1, 0.8, 0.8)):
    # crop_ratio 为 (top, left, bottom, right) 的比例
    top_ratio, left_ratio, bottom_ratio, right_ratio = crop_ratio
    height, width, _ = img.shape

    # 计算裁剪坐标
    top = int(height * top_ratio)
    left = int(width * left_ratio)
    bottom = int(height * bottom_ratio)
    right = int(width * right_ratio)
    img = img[top:bottom, left:right]

    # 判断裁剪后的图像是否为正方形
    height, width, _ = img.shape
    if height != width:
        # 创建一个随机颜色的正方形
        new_size = max(height, width)
        fill_color = (128, 128, 128)
        new_img = np.ones((new_size, new_size, 3), dtype=np.uint8) * np.array(fill_color, dtype=np.uint8)

        # 计算粘贴坐标
        paste_x = (new_size - width) // 2
        paste_y = (new_size - height) // 2
        new_img[paste_y:paste_y + height, paste_x:paste_x + width] = img
        img = new_img

    # 缩放图像
    img = cv2.resize(img, (scale, scale))
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """

    img = img_in.copy()
    # img = fill_image(img)
    img = img[:, :, ::-1]  # BGR --> RGB
    transform = transforms.Compose([
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
        transforms.Normalize(mean, std)
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, strlabel, pre_class,picname):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = 0.5 * heatmap + 0.5 * np.float32(img)
    cam = cam / np.max(cam)

    # path_cam_img = os.path.join(out_dir, filename + "[" + pre_class + "]" + ".jpg")
    real_cls=os.path.basename(out_dir)
    path_cam_img = os.path.join(out_dir, f"{picname}_{strlabel}_{pre_class+1}.jpg")
    # path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    # cv2.imwrite(path_raw_img, np.uint8(255 * img))
    print("predict: {}".format(pre_class))


def comp_class_vec(ouput_vec, classes_len, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, classes_len).scatter_(1, index, 1)
    one_hot = one_hot.to(device)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (448, 448))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


def parse_dataset(file_path, root_dir):
    """
    从数据文件解析图像路径和类别编号。
    
    Args:
        file_path (str): 数据文件路径，如 train.txt 或 test.txt。
        root_dir (str): 存放图像的根目录。
        
    Returns:
        List[Tuple[str, int]]: 包含 (图像路径, 类别编号) 的列表。
    """
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                img_name, label = line.split()  # 假设文件内容是 "图片名 类别编号"
                img_path = os.path.join(root_dir, img_name)
                dataset.append((img_path, int(label)))
    return dataset

def parse_directory(root_dir):
    """
    从目录中解析图像路径和类别编号。
    
    Args:
        root_dir (str): 存放图像的根目录，如 train/ 或 test/。
        
    Returns:
        List[Tuple[str, int]]: 包含 (图像路径, 类别编号) 的列表。
    """
    dataset = []
    class_to_idx = {}  # 类别名到编号的映射
    for idx, class_name in enumerate(sorted(os.listdir(root_dir))):  # 确保类别编号按字母顺序排序
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            class_to_idx[class_name] = idx
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):
                    dataset.append((img_path, idx))  # (图像路径, 类别编号)
    return dataset

if __name__ == '__main__':
    # 读取数据
    daset='gen'
    if daset=='dog':
        root_dir = "/data/public/Standford_Dogs/Images"  # 图像根目录
        train_file = "/data/user/zhuyueran/models/MMAL/datasets/Stanford_Dogs/train.txt"
        test_file = "/data/user/zhuyueran/models/MMAL/datasets/Stanford_Dogs/test.txt"
        path_net = "./checkpoint/dog/dgcn_qkv50_mask/best.pth"  # /mnt1/zhuyueran/models/jianwei/mmal_penmo/checkpoints/zhlq/epoch5.pth
        output_dir = './heatmap/dog_50k3_heatmap'
        class_num=120
    elif daset=='air':
        root_dir = "/data/user/zhuyueran/models/MMAL/datasets/FGVC-aircraft/data/images"  # 图像根目录
        train_file = "/data/user/zhuyueran/models/MMAL/datasets/FGVC-aircraft/data/train.txt"
        test_file = "/data/user/zhuyueran/models/MMAL/datasets/FGVC-aircraft/data/test.txt"
        path_net = "./checkpoint/aircraft/gcn_qkv50_mask/best.pth"
        output_dir = './heatmap/air_50k3_heatmap'
        class_num=100
    
    elif daset=='lux':
        root_dir_train = "/data/user/zhuyueran/datasets/maotai/241112_latest_crop_fsp/train"
        root_dir_test = "/data/user/zhuyueran/datasets/maotai/241112_latest_crop_fsp/test"
        path_net = "./checkpoint/Luxury/maotai_1/best.pth"
        output_dir = './heatmap/lux_50k3_heatmap/maotai'
        class_num=8
    
    elif daset=='gen':
        root_dir_train = "/data/user/zhuyueran/datasets/fake_face_custom/train"
        root_dir_test = "/data/user/zhuyueran/datasets/fake_face_custom/test"
        path_net = "./checkpoint/Geners/fface_1/best.pth"
        output_dir = './heatmap/gen_50k3_heatmap/fface1'
        class_num=2
    
    os.makedirs(output_dir, exist_ok=True)

    # 读取训练集和测试集
    train_dataset = parse_directory(root_dir_train)
    test_dataset = parse_directory(root_dir_test)

    # 初始化模型
    net = MainNet(proposalN=6, num_classes=class_num, channels=2048)
    weights_dict = torch.load(path_net, map_location=device)
    net.load_state_dict(weights_dict['model_state_dict'])
    net = net.to(device)
    net.eval()

    # 注册 hook
    net.pretrained_model.layer4[2].conv3.register_forward_hook(farward_hook)
    net.pretrained_model.layer4[2].conv3.register_backward_hook(backward_hook)

    # 开始生成热力图
    for img_path, label in tqdm(test_dataset, desc="Generating Heatmaps"):
        fmap_block = []
        grad_block = []

        # 读取图片
        img = cv2.imread(img_path, 1)
        img_input = img_preprocess(img).to(device)
        un = UnNormalize(mean, std)

        # Forward pass
        output = net(img_input, 0, 0, 'test', device)[-3]
        idx = np.argmax(output.cpu().data.numpy())
        pre_class = idx

        # Backward pass
        net.zero_grad()
        class_loss = comp_class_vec(output, class_num)
        class_loss.backward()

        # 生成热力图
        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        # 保存热力图
        img_show = un(img_input).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)[..., ::-1]
        # img_name = os.path.basename(img_path)
        # 提取文件名，不含扩展名
        extract_name = os.path.splitext(img_path.split('/')[-1])[0]
        output_sub_dir = os.path.join(output_dir, f"class_{label}")
        os.makedirs(output_sub_dir, exist_ok=True)
        show_cam_on_image(img_show, cam, output_sub_dir, str(label), pre_class, extract_name)



