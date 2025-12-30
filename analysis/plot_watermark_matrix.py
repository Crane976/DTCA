# analysis/plot_watermark_matrix.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# --- 路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
figures_dir = os.path.join(project_root, 'figures')
os.makedirs(figures_dir, exist_ok=True)

# --- 全局设置 ---
sns.set(style="white")
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    pass
plt.rcParams['font.size'] = 14


def plot_watermark_matrix():
    # 数据来源于你的实验结果
    # Row 0: Actual Decoy (伪装流量)
    # Row 1: Actual Real (真实流量)
    # Col 0: Predicted Decoy (检出水印)
    # Col 1: Predicted Real (未检出水印)

    # TPR = 100%, FNR = 0%
    # FPR = 0.94%, TNR = 100 - 0.94 = 99.06%

    matrix_data = np.array([
        [100.00, 0.00],  # Decoy
        [0.94, 99.06]  # Real
    ])

    # 标签定义
    group_names = ['True Positive\n(Identified)', 'False Negative\n(Missed)',
                   'False Positive\n(Misidentified)', 'True Negative\n(Clean)']

    group_counts = ["{0:0.2f}%".format(value) for value in matrix_data.flatten()]

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))

    # 使用蓝色调，符合论文风格
    sns.heatmap(matrix_data, annot=labels, fmt='', cmap='Blues', cbar=True,
                xticklabels=['Watermark Detected', 'No Watermark'],
                yticklabels=['Decoy Traffic', 'Real Traffic'],
                annot_kws={"size": 14, "weight": "bold"},
                vmin=0, vmax=100)

    # 调整轴标签
    ax.set_ylabel('Actual Source', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel('Verification Result', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Traceability Performance (Key=97)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'traceability_matrix.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 溯源混淆矩阵已保存: {save_path}")


if __name__ == "__main__":
    plot_watermark_matrix()