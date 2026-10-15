import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network import mlp1, mlp2, mlp3, AlexNet, VGG19, ResNet50

# ====================== 选择模型 ========================
model_name = "VGG19"   # 换成其他模型名字切换模型：mlp1/mlp2/mlp3/AlexNet/VGG19/ResNet50
model = VGG19()
# =======================================================

# ====================== GPU推理 ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) # 模型放到GPU

# 获取当前脚本所在目录（homework2）
base_path = os.path.dirname(os.path.abspath(__file__))
# 权重路径：homework2/mlp1/mlp1_weights/mlp1_best.pth
weight_folder = f"{model_name}_weights"
best_weight_path = os.path.join(base_path, model_name, weight_folder, f"{model_name}_best.pth")

# 加载权重并评估模式
model.load_state_dict(torch.load(best_weight_path, map_location=device))
model.eval()

# FashionMNIST类别名称
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 加载测试数据集
transform = transforms.Compose([transforms.ToTensor()])
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3), # 1通道复制成3通道
    transforms.ToTensor(),
])
data_root = os.path.join(base_path, "../data")
test_data = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=train_transform)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# ========= 可视化图片+预测结果，图片保存到 xxx_plt =========
with torch.no_grad():
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    pred = model(imgs)
    pred_class = torch.argmax(pred, dim=1)

plt.figure(figsize=(12, 12))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    img_tensor = imgs[i].cpu()
    if img_tensor.shape[0] == 1:
        # MLP单通道情况
        img_np = img_tensor.squeeze().numpy()
    else:
        # VGG/AlexNet/ResNet三通道情况 (C,H,W)->(H,W,C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
    plt.imshow(img_np, cmap="gray")
    plt.title(f"True:{class_names[labels[i]]}\nPred:{class_names[pred_class[i]]}")
    plt.axis("off")
plt.tight_layout()

# 样例图自动保存到对应模型的plt文件夹
img_save_path = os.path.join(base_path, model_name, f"{model_name}_plt", f"{model_name}_pred_sample.png")
plt.savefig(img_save_path, dpi=300)
plt.show()
print(f"✅ 预测样例图片保存至：{img_save_path}")

# ========= 计算测试集整体准确率 =========
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        pred = model(imgs)
        pred_class = torch.argmax(pred, dim=1)
        total += labels.size(0)
        correct += (pred_class == labels).sum().item()

print(f"\n==== 模型：{model_name} ====")
print(f"测试集准确率: {correct / total:.4f}")
print(f"✅ 加载权重路径：{best_weight_path}")
