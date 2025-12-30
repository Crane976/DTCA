# analysis/plot_radar_capability.py
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 路径 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
figures_dir = os.path.join(project_root, 'figures')
os.makedirs(figures_dir, exist_ok=True)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14


def plot_radar_chart():
    # 数据 (基于你之前 prove_hunter_capability.py 的结果)
    # 顺序: Precision, Recall, F1, Accuracy
    categories = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    N = len(categories)

    # 数据集 (1:1 Balanced Test)
    # MLP: P=0.98, R=0.98, F1=0.98, Acc=0.98
    values_mlp = [0.9838, 0.9834, 0.9834, 0.9834]
    # XGB: P=0.99, R=0.98, F1=0.98, Acc=0.98
    values_xgb = [0.9850, 0.9847, 0.9847, 0.9847]
    # CNN: P=0.99, R=0.99, F1=0.99, Acc=0.99
    values_cnn = [0.9862, 0.9860, 0.9860, 0.9860]
    # KNN: P=0.93, R=0.92, F1=0.92, Acc=0.92
    values_knn = [0.9253, 0.9184, 0.9180, 0.9184]

    # 闭合曲线
    values_mlp += values_mlp[:1]
    values_xgb += values_xgb[:1]
    values_cnn += values_cnn[:1]
    values_knn += values_knn[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # 绘制线条
    ax.plot(angles, values_mlp, linewidth=2, linestyle='solid', label='MLP Hunter')
    ax.fill(angles, values_mlp, 'b', alpha=0.1)

    ax.plot(angles, values_xgb, linewidth=2, linestyle='dashed', label='XGBoost Hunter')

    ax.plot(angles, values_cnn, linewidth=2, linestyle='dotted', label='1D-CNN Hunter')

    ax.plot(angles, values_knn, linewidth=2, linestyle='dashdot', label='KNN Hunter')

    # 标签
    plt.xticks(angles[:-1], categories, color='black', size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.05)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Baseline Capability of Hunters (Balanced Test)', y=1.08, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'hunter_capability_radar.png'), dpi=300)
    print("✅ 雷达图已保存。")


if __name__ == "__main__":
    plot_radar_chart()