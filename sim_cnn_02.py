import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from getData import get_mnist_dataloader, get_knn_data  # 保留你的数据加载模块
import time


class SimpleCNN(nn.Module):
    def __init__(self, kernel_size=3, hidden_dim1=128, hidden_dim2=64):
        """
        新增第二个隐藏层的CNN模型
        :param kernel_size: 卷积核大小（默认3×3）
        :param hidden_dim1: 第一个隐藏层神经元数（默认128）
        :param hidden_dim2: 第二个隐藏层神经元数（默认64）
        """
        super(SimpleCNN, self).__init__()
        # 卷积层：32个指定大小卷积核，padding保证尺寸匹配
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        # 动态计算卷积后展平维度（28×28经卷积+池化后为14×14）
        self.conv_out_dim = 32 * 14 * 14  # 卷积层输出维度
        
        # 第一个隐藏层（原fc1）：卷积输出 → hidden_dim1
        self.fc1 = nn.Linear(self.conv_out_dim, hidden_dim1)
        # 第二个隐藏层（新增）：hidden_dim1 → hidden_dim2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        # 输出层：第二个隐藏层 → 10类（数字0-9）
        self.fc3 = nn.Linear(hidden_dim2, 10)

    def forward(self, x):
        # 卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))
        # 展平：4维张量 → 2维（样本数×特征数）
        x = x.view(-1, self.conv_out_dim)
        
        # 第一个隐藏层+激活
        x = F.relu(self.fc1(x))
        # 第二个隐藏层+激活（新增）
        x = F.relu(self.fc2(x))
        # 输出层（无激活，用CrossEntropyLoss时不需要Softmax）
        x = self.fc3(x)
        return x


def calculate_acc(model, dataloader, device):
    """通用准确率计算函数（训练集/测试集通用）"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total  # 返回百分比准确率


def train_cnn(model, train_loader, test_loader, epochs=50, lr=0.001, dataset_type="small"):
    """
    优化的CNN训练函数（修复打印/过拟合计算，增强日志）
    :param model: CNN模型实例
    :param train_loader: 训练数据加载器
    :param test_loader: 测试数据加载器
    :param epochs: 迭代次数
    :param lr: 学习率
    :param dataset_type: 数据集类型（small/full，仅用于日志）
    """
    # 基础配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 记录训练总时间
    total_start = time.time()

    print(f"\n===== 开始训练CNN（{dataset_type}数据集）=====")
    print(f"配置：Epochs={epochs}, LR={lr}, Device={device}")

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)  # 累计批次损失

        # 计算训练集/测试集准确率
        train_acc = calculate_acc(model, train_loader, device)
        test_acc = calculate_acc(model, test_loader, device)
        avg_train_loss = train_loss / len(train_loader.dataset)  # 平均损失
        overfitting = train_acc - test_acc  # 正确的过拟合程度（训练准确率-测试准确率）

        # 打印epoch级别结果（修复格式错误，信息更清晰）
        print(f"\nEpoch {epoch+1} 结果：")
        print(f"  训练平均损失：{avg_train_loss:.4f}")
        print(f"  训练集准确率：{train_acc:.2f}%")
        print(f"  测试集准确率：{test_acc:.2f}%")
        print(f"  过拟合程度：{overfitting:.2f}%")

    # 训练结束：汇总结果
    total_time = time.time() - total_start
    final_train_acc = calculate_acc(model, train_loader, device)
    final_test_acc = calculate_acc(model, test_loader, device)
    final_overfitting = final_train_acc - final_test_acc
    final_avg_loss = train_loss / len(train_loader.dataset)

    print("\n===== CNN训练完成 ======")
    print(f"最终训练平均损失：{final_avg_loss:.4f}")
    print(f"最终测试集准确率：{final_test_acc:.2f}%")
    print(f"最终过拟合程度：{final_overfitting:.2f}%")
    print(f"总训练时间：{total_time:.2f}秒（{total_time/60:.2f}分钟）")


def train_knn(X_train, y_train, X_test, y_test, n_neighbors=5, dataset_type="full"):
    """
    优化的KNN训练函数（增强日志，记录时间）
    :param X_train/X_test: 训练/测试特征（扁平化）
    :param y_train/y_test: 训练/测试标签
    :param n_neighbors: K值
    :param dataset_type: 数据集类型（small/full）
    :return: 训练好的KNN模型
    """
    print(f"\n===== 开始训练KNN（{dataset_type}数据集，k={n_neighbors}）=====")
    start_time = time.time()

    # 训练KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # 评估
    train_acc = 100 * knn.score(X_train, y_train)
    test_acc = 100 * knn.score(X_test, y_test)
    overfitting = train_acc - test_acc
    train_time = time.time() - start_time

    # 打印结果（格式统一，便于实验记录）
    print(f"KNN训练完成（耗时：{train_time:.2f}秒）：")
    print(f"  训练集准确率：{train_acc:.2f}%")
    print(f"  测试集准确率：{test_acc:.2f}%")
    print(f"  过拟合程度：{overfitting:.2f}%")

    return knn


# ===================== 实验运行入口 =====================
if __name__ == "__main__":
    # 1. 训练双隐藏层CNN（小数据集）
    print("---------- 实验：双隐藏层CNN（小数据集） ----------")
    train_loader_cnn_small, test_loader_cnn_small = get_mnist_dataloader(small_dataset=True)
    
    # 实例化双隐藏层CNN：第一个隐藏层128神经元，第二个64神经元（可自定义）
    cnn_model_double_hidden = SimpleCNN(
        kernel_size=5,    # 卷积核大小不变
        hidden_dim1=128,  # 第一个隐藏层（原fc1）
        hidden_dim2=64    # 第二个隐藏层（新增）
    )
    
    # 调用训练函数
    train_cnn(
        model=cnn_model_double_hidden,
        train_loader=train_loader_cnn_small,
        test_loader=test_loader_cnn_small,
        epochs=10,
        lr=0.001,
        # dataset_type="small"
    )