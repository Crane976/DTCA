# analysis/plot_results_final.py (UPDATED WITH FINAL DATA)
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
# 尝试设置论文标准字体
try:
    plt.rcParams['font.family'] = 'Times New Roman'
except:
    pass
plt.rcParams['font.size'] = 14  # 稍微调大字体，适合论文阅读


def plot_precision_collapse():
    """绘制图表 3: 精确率断崖 (Final Version)"""
    models = ['1D-CNN', 'XGBoost', 'KNN', 'MLP']

    # ✅ 数据更新：基于最终一轮跑出的结果
    # Base Precision (攻击前)
    base_prec = [100.00, 96.22, 91.70, 98.75]
    # Decayed Precision (攻击后)
    decayed_prec = [1.46, 1.56, 1.41, 1.39]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制柱状图
    rects1 = ax.bar(x - width / 2, base_prec, width, label='Base Precision (No Attack)', color='#4c72b0',
                    edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width / 2, decayed_prec, width, label='Decayed Precision (Under Attack)', color='#c44e52',
                    edgecolor='black', linewidth=1)

    ax.set_ylabel('Precision (%)', fontweight='bold')
    ax.set_title('Impact of Decoy Traffic on Detection Precision', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12, loc='upper right')

    # 设置Y轴上限，留出空间
    ax.set_ylim(0, 115)

    # 标签函数
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'precision_collapse_final.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 精确率断崖图已保存: {save_path}")


def plot_alert_composition():
    """绘制图表 4: 警报成分分析 (以表现最均衡的 MLP 为例)"""
    # 基于 MLP 数据:
    # Decoy Rate 42.12% -> 伪装Bot产生的警报 = 40000 * 0.4212 = 16848
    # Recall 60.46% -> 真实Bot产生的警报 = 392 * 0.6046 = 237
    # FP -> 3
    # 总警报 = 16848 + 237 + 3 = 17088

    # 计算百分比
    pct_decoy = 16848 / 17088 * 100  # ~98.6%
    pct_real = 237 / 17088 * 100  # ~1.39%
    pct_fp = 3 / 17088 * 100  # ~0.02%

    labels = [f'Real Target\n({pct_real:.1f}%)', f'False Positive\n(<0.1%)', f'Decoy / Fake Target\n({pct_decoy:.1f}%)']
    sizes = [pct_real, pct_fp, pct_decoy]
    colors = ['#55a868', '#f1c40f', '#c44e52']  # 绿(真), 黄(误), 红(假)

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, texts, autotexts = ax.pie(sizes, labels=None, autopct='',  # 暂时不自动显示标签，手动加
                                      startangle=160, colors=colors, pctdistance=0.85,
                                      explode=(0.1, 0.2, 0))  # 炸开小的部分

    # 画甜甜圈
    centre_circle = plt.Circle((0, 0), 0.65, fc='white')
    fig.gca().add_artist(centre_circle)

    # 添加图例
    ax.legend(wedges, labels,
              title="Target Source",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title('Target Composition (Attacker\'s View)', fontsize=16, fontweight='bold')

    # 在圆环中心写字
    ax.text(0, 0, f'DSR\n{pct_decoy:.1f}%', ha='center', va='center', fontsize=20, fontweight='bold', color='#c44e52')

    plt.tight_layout()
    save_path = os.path.join(figures_dir, 'Target_list_composition_final.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ 警报成分图已保存: {save_path}")


if __name__ == "__main__":
    plot_precision_collapse()
    plot_alert_composition()