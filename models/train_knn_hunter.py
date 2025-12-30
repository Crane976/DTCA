# models/train_knn_hunter.py (FINAL FIXED VERSION WITH DATA CLEANING)
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed

try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# --- 配置区 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MODELS_DIR = os.path.join(project_root, 'models')
FIGURES_DIR = os.path.join(project_root, 'figures')
HUNTER_MODEL_PATH = os.path.join(MODELS_DIR, 'knn_hunter.pkl')


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 开始训练 KNN Hunter (修复版: 含数据清洗)...")
    print("=" * 60)

    # --- 1. 加载数据 ---
    print("正在加载数据...")
    try:
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = scaler.feature_names_in_
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # --- ✅ 关键修复: 数据清洗 (直接丢弃 Inf/NaN) ---
    print("正在清洗数据 (去除 Inf/NaN)...")
    # 1. 替换 Inf
    df_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 丢弃脏数据
    len_before = len(df_train_full)
    df_train_full.dropna(subset=feature_names, inplace=True)
    print(f"   -> 训练集清洗掉 {len_before - len(df_train_full)} 条脏数据")

    len_test_before = len(df_test)
    df_test.dropna(subset=feature_names, inplace=True)
    print(f"   -> 测试集清洗掉 {len_test_before - len(df_test)} 条脏数据")

    # --- 2. 构建训练子集 ---
    # 注意: KNN计算量大，我们需要重采样
    print("\n[步骤1] 构建训练子集...")
    X_full = df_train_full[DEFENDER_SET]
    y_full = df_train_full['label']

    # 划分验证集 (保持真实比例 100:1)
    X_train_pool, X_val_natural, y_train_pool, y_val_natural = train_test_split(
        X_full, y_full, test_size=0.2, random_state=2025, stratify=y_full
    )

    # 训练集重采样
    df_pool = pd.concat([X_train_pool, y_train_pool], axis=1)
    df_bot = df_pool[df_pool['label'] == 1]
    df_benign = df_pool[df_pool['label'] == 0]

    n_bot = len(df_bot)
    # 🔥 策略调整: 尝试 1:1 采样以提升 Recall (如果还是低，这里是关键)
    # 为了对比，这里先保持你之前的逻辑，或者建议改为 n_bot * 1
    # --- 修改点 1: 增加良性样本比例至 1:10 ---
    # 之前是 n_bot * 5，现在改为 * 10
    # 目的：让模型见识更多样的良性样本，减少误报
    n_benign_sample = int(n_bot * 20)

    df_benign_sampled = df_benign.sample(n=n_benign_sample, random_state=2025)
    df_train_balanced = pd.concat([df_bot, df_benign_sampled])

    print(f"   -> 训练样本数: {len(df_train_balanced)} (Bot: {n_bot}, Benign: {n_benign_sample})")

    # --- 3. 缩放 ---
    print("正在使用Scaler转换数据...")
    X_train_final = scaler.transform(df_train_balanced[DEFENDER_SET])
    y_train_final = df_train_balanced['label']
    X_val_natural_scaled = scaler.transform(X_val_natural)
    X_test_scaled = scaler.transform(df_test[DEFENDER_SET])
    y_test = df_test['label']

    # --- 4. 训练 ---
    print("\n[步骤2] 训练 KNN (High Precision版)...")
    # --- 修改点 2: 增加 K 值到 51 ---
    # K越大，决策边界越平滑，越不容易误报
    knn_model = KNeighborsClassifier(n_neighbors=31, weights='distance', n_jobs=-1)

    with tqdm(total=1, desc="KNN Fitting") as pbar:
        knn_model.fit(X_train_final, y_train_final)
        pbar.update(1)

    # --- 5. 阈值寻优 ---
    print("\n[步骤3] 在【真实分布验证集】上寻找最佳决策阈值...")
    val_probs = knn_model.predict_proba(X_val_natural_scaled)[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    for thr in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y_val_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val_natural, y_val_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    print(f"✅ 最佳阈值: {best_threshold:.2f} (验证集 F1: {best_f1:.4f})")

    # --- 6. 评估 ---
    joblib.dump(knn_model, HUNTER_MODEL_PATH)

    print(f"\n--- 最终报告 (阈值={best_threshold:.2f}) ---")
    test_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (test_probs >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()