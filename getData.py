import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# 基础数据变换（MNIST专用）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量并归一化到[0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST均值/标准差，提升训练稳定性
])

def get_mnist_dataloader(small_dataset=False, batch_size=32):
    """
    获取MNIST数据加载器（CNN用）
    :param small_dataset: 是否使用小数据集（True/False）
    :param batch_size: 批次大小
    :return: train_loader, test_loader
    """
    # 加载原始数据集
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    # 小数据集采样（按类别均衡抽取，避免类别失衡）
    if small_dataset:
        # 小数据集：训练集2000个，测试集1000个
        train_indices = []
        test_indices = []
        # 按类别（0-9）抽取
        for cls in range(10):
            # 训练集：每个类别抽200个（200*10=2000）
            cls_train_idx = np.where(np.array(train_dataset.targets) == cls)[0][:200]
            train_indices.extend(cls_train_idx)
            # 测试集：每个类别抽100个（100*10=1000）
            cls_test_idx = np.where(np.array(test_dataset.targets) == cls)[0][:100]
            test_indices.extend(cls_test_idx)
        # 构建子集
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # 构建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader

def get_knn_data(small_dataset=False):
    """
    获取KNN所需的扁平化数据（修复变量顺序问题）
    :param small_dataset: 是否使用小数据集
    :return: X_train, y_train, X_test, y_test
    """
    # 第一步：先获取数据加载器（batch_size设为全量，一次性加载）
    # 注意：先定义train_loader，再访问其属性，解决UnboundLocalError
    train_loader, test_loader = get_mnist_dataloader(
        small_dataset=small_dataset,
        batch_size=16  # 超大批次，确保一次性加载所有数据
    )

    # 第二步：提取训练集数据并扁平化（KNN需要二维特征：样本数×特征数）
    # 加载训练集
    train_data, train_labels = next(iter(train_loader))
    # 扁平化：(N, 1, 28, 28) → (N, 784)
    X_train = train_data.view(train_data.size(0), -1).numpy()
    y_train = train_labels.numpy()

    # 第三步：提取测试集数据并扁平化
    test_data, test_labels = next(iter(test_loader))
    X_test = test_data.view(test_data.size(0), -1).numpy()
    y_test = test_labels.numpy()

    return X_train, y_train, X_test, y_test