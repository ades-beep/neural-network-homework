import torch
import matplotlib.pyplot as plt

# 生成x数据
x = torch.linspace(-10, 10, 1000)

# 计算各个激活函数
y_sigmoid = torch.sigmoid(x)
y_tanh = torch.tanh(x)
y_relu = torch.relu(x)
y_softplus = torch.nn.functional.softplus(x)

# 创建画布
plt.figure(figsize=(10, 8))

# 子图1 Sigmoid
plt.subplot(2, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)

# 子图2 Tanh
plt.subplot(2, 2, 2)
plt.plot(x, y_tanh)
plt.title('Tanh')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)

# 子图3 ReLU
plt.subplot(2, 2, 3)
plt.plot(x, y_relu)
plt.title('ReLU')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)

# 子图4 Softplus
plt.subplot(2, 2, 4)
plt.plot(x, y_softplus)
plt.title('Softplus')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout() #自动调整整张画布里面所有子图、标题、坐标轴标签的位置
plt.savefig("images/activation.png", dpi=300, bbox_inches='tight')
plt.show()