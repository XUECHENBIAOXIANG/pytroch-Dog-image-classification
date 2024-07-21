import scipy.io
import os
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt  # 导入Matplotlib库
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def process_image(image_path, annotation_path):
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 解析注释文件
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # 获取边界框信息
    bndbox = root.find('.//bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)

    # 裁剪图像
    cropped_image = image.crop((xmin, ymin, xmax, ymax))

    return cropped_image


class DogDataset(Dataset):
    def __init__(self, file_list, annotation_list, labels, image_dir, annotation_dir):
        self.file_list = file_list
        self.annotation_list = annotation_list
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_file = self.file_list[idx][0][0]
        annotation_file = self.annotation_list[idx][0][0]

        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.annotation_dir, annotation_file )

        image = process_image(image_path, annotation_path)
        label = self.labels[idx][0]

        return image, label


# 加载文件列表
train_list = scipy.io.loadmat('lists/train_list.mat')
file_list = train_list['file_list']
annotation_list = train_list['annotation_list']
labels = train_list['labels']

# 定义图像和注释文件夹路径
image_dir = 'images/Images'
annotation_dir = 'annotation/Annotation'

# 创建数据集实例
train_dataset = DogDataset(file_list[:10], annotation_list[:10], labels[:10], image_dir, annotation_dir)

# 显示前十个样本
plt.figure(figsize=(15, 6))
for i in range(10):
    image, label = train_dataset[i]

    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f'Label: {label}')
    plt.axis('off')

plt.tight_layout()
plt.show()
