import csv
import os

import matplotlib
matplotlib.use('Agg')  # 无图形界面环境也能保存图片
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'result')

# 网络名 -> 训练日志 csv 的相对路径
# 6 条线用 6 种不同颜色（matplotlib tab10 色板的前 6 个）
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

NETWORKS = [
    ('mlp1', 'mlp1/mlp1_plt/mlp1_log.csv'),
    ('mlp2', 'mlp2/mlp2_plt/mlp2_log.csv'),
    ('mlp3', 'mlp3/mlp3_plt/mlp3_log.csv'),
    ('AlexNet', 'AlexNet/AlexNet_plt/AlexNet_log.csv'),
    ('VGG19', 'VGG19/VGG19_plt/VGG19_log.csv'),
    ('ResNet50', 'ResNet50/ResNet50_plt/ResNet50_log.csv'),
]


def load_log(path):
    """读取训练日志 csv，返回 (epoch, train_loss, test_loss, test_acc) 四个列表。"""
    epochs, train_loss, test_loss, test_acc = [], [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train_loss.append(float(row['train_loss']))
            test_loss.append(float(row['test_loss']))
            test_acc.append(float(row['test_acc']))
    return epochs, train_loss, test_loss, test_acc


def plot_loss_curves(logs):
    """图1：2x3 子图，每个网络一张，画它的 Train loss 和 Test loss。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, (name, log) in zip(axes.ravel(), logs):
        epochs, train_loss, test_loss, _ = log
        ax.plot(epochs, train_loss, label='Train loss', color='tab:blue')
        ax.plot(epochs, test_loss, label='Test loss', color='tab:orange')
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(0, 2.5)  # 纵轴统一固定 0~2.5
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, 'loss_curves.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {save_path}')


def plot_test_accuracy(logs):
    """图2：单张图，6 个网络的 Test accuracy 画在同一坐标系里。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, log) in enumerate(logs):
        epochs, _, _, test_acc = log
        ax.plot(epochs, test_acc, label=name, color=COLORS[i], linewidth=1.5)

    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    fig.tight_layout()
    save_path = os.path.join(OUT_DIR, 'test_accuracy.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {save_path}')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    logs = []
    for name, rel_path in NETWORKS:
        logs.append((name, load_log(os.path.join(ROOT, rel_path))))

    plot_loss_curves(logs)
    plot_test_accuracy(logs)
