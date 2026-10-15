import os
import matplotlib.pyplot as plt
import pandas as pd

# ===================== 只需要修改这一行：填写你的csv文件名 =====================
csv_filename = "AlexNet_log.csv"
# 切换mlp2就改成 csv_filename = "mlp2_log.csv"
# ============================================================================

# 获取当前plot.py所在目录（homework2）
base_path = os.path.dirname(os.path.abspath(__file__))

# 自动从csv文件名解析模型名称
model_name = csv_filename.split("_log.csv")[0]

# 拼接csv完整路径： homework2/mlp1/mlp1_plt/mlp1_log.csv
plt_folder_name = f"{model_name}_plt"
csv_full_path = os.path.join(base_path, model_name, plt_folder_name, csv_filename)

# 读取csv
df = pd.read_csv(csv_full_path)
epoch_list = df["epoch"]
train_loss = df["train_loss"]
test_loss = df["test_loss"]
test_acc = df["test_acc"]

# 绘图
plt.figure(figsize=(12, 4))

# 子图1：loss曲线
plt.subplot(1, 2, 1)
plt.plot(epoch_list, train_loss, label="Train Loss")
plt.plot(epoch_list, test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_name} Loss Curve")
plt.legend()  #生成图例标签
plt.grid(True)
plt.ylim(0, 2.5)

# 子图2：测试准确率
plt.subplot(1, 2, 2)
plt.plot(epoch_list, test_acc, color="orange", label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"{model_name} Accuracy Curve")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)

plt.tight_layout() #自动调整整张画布上所有元素的位置，防止文字重叠、被裁掉

# 图片保存到 xxx_plt 文件夹
img_save_path = os.path.join(base_path, model_name, plt_folder_name, f"{model_name}_curve.png")
plt.savefig(img_save_path, dpi=300)
plt.show()

print(f"✅ 绘图完成！")
print(f"📄 读取csv：{csv_full_path}")
print(f"🖼️ 图片保存位置：{img_save_path}")