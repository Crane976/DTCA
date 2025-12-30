# analysis/plot_ablation_study.py
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
sns.set(style="whitegrid")
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    pass
plt.rcParams['font.size'] = 12


def plot_ablation_decoy_rate():
    models = ['1D-CNN', 'MLP', 'XGBoost', 'KNN']

    # 数据录入 (Decoy Rate %)
    # 顺序对应上面的 models 列表
    # Variant B (最差基准)
    no_constraint = [0.00, 25.44, 30.14, 40.48]
    # Variant A
    no_cluster = [27.21, 37.13, 47.08, 59.41]
    # Final
    proposed = [40.00, 42.12, 36.17, 40.48]

    x = np.arange(len(models))
    width = 0.25  # 柱子宽度

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制三组柱子
    rects1 = ax.bar(x - width, no_constraint, width, label='w/o Hard Constraints', color='#95a5a6', edgecolor='white')
    rects2 = ax.bar(x, no_cluster, width, label='w/o Cluster Focus', color='#3498db', edgecolor='white')
    rects3 = ax.bar(x + width, proposed, width, label='Proposed Framework', color='#e74c3c', edgecolor='white',
                    hatch='//')

    ax.set_ylabel('Decoy Success Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Ablation Study: Contribution of Components', fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(loc='upper left', fontsize=10)

    ax.set_ylim(0, 70)

    # 标注数值 (只标注 Final 和 0.00 的特殊情况)
    def autolabel(rects, is_final=False):
        for rect in rects:
            height = rect.get_height()
            if is_final or height == 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)  # 标注 No Constraint
    autolabel(rects3, is_final=True)  # 标注 Final

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'ablation_study.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 消融实验对比图已保存: {save_path}")


if __name__ == "__main__":
    plot_ablation_decoy_rate()