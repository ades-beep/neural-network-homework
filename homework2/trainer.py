import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
from network import mlp1, mlp2, mlp3, AlexNet, VGG19, ResNet50

# ========= 选择模型，切换 mlp1() / mlp2() =========
model = ResNet50()
model_name = type(model).__name__

# ====================== GPU设备设置 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model = model.to(device) # 模型移到GPU

# ============ 基准路径，强制所有文件夹生成在 homework2 内部 ========
base_path = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(base_path, model_name)
weight_dir = os.path.join(root_dir, f"{model_name}_weights")
plt_dir = os.path.join(root_dir, f"{model_name}_plt")

os.makedirs(weight_dir, exist_ok=True)
os.makedirs(plt_dir, exist_ok=True)

# ========= 1. 数据加载 =========
transform = transforms.Compose([transforms.ToTensor()])
# ========= 1.1. 使用CNN需要把数据集扩展为三通道，resize成224*224 ============
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3), # 1通道复制成3通道
    transforms.ToTensor(),
])

data_root = os.path.join(base_path, "../data")
train_data = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_transform)
test_data = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True) #64
test_loader = DataLoader(test_data, batch_size=64, shuffle=False) #64

# ========= 2. 模型、损失函数、优化器 =========
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(), lr=1e-2)

# csv日志放在plt文件夹下
csv_path = os.path.join(plt_dir, f"{model_name}_log.csv")
# 写入表头
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "test_loss", "test_acc"])

epochs = 20
best_acc = 0

for epoch in range(epochs):
    # 训练
    model.train()
    total_train_loss = 0
    for imgs, labels in train_loader:
        # ===== 数据送入GPU =====
        imgs, labels = imgs.to(device), labels.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # 测试评估
    model.eval()
    total_test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            # ===== 数据送入GPU =====
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, labels)
            total_test_loss += loss.item()
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == labels).sum().item()
    avg_test_loss = total_test_loss / len(test_loader)
    acc = correct / len(test_data)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss:{avg_train_loss:.4f} | "
          f"Test Loss:{avg_test_loss:.4f} | "
          f"Test Acc:{acc:.4f}")

    # 追加写入日志到csv
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_test_loss, acc])

    # 保存最优权重到weights文件夹
    if acc > best_acc:
        best_acc = acc
        best_save_path = os.path.join(weight_dir, f"{model_name}_best.pth")
        torch.save(model.state_dict(), best_save_path)
        print(f"====> 更新最优模型! epoch{epoch+1}, best acc={best_acc:.4f}")

    # 每5epoch保存断点权重
    if (epoch + 1) % 5 == 0:
        save_path = os.path.join(weight_dir, f"{model_name}_epoch_{epoch+1}_acc_{acc:.3f}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"====> 保存断点权重: {save_path}")
