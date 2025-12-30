# models/STEP1_train_style_transfer_cae.py (FINAL 3-TIER BALANCED & CLEANED VERSION)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import os
import sys
import joblib

# ==========================================================
# --- 路径修正与模块导入 ---
# ==========================================================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.style_transfer_cae import ConditionalAutoencoder
from config import ATTACKER_KNOWLEDGE_SET, set_seed

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MODELS_DIR = os.path.join(project_root, 'models')
CAE_MODEL_PATH = os.path.join(MODELS_DIR, 'style_transfer_cae.pt')

FEATURE_DIM = len(ATTACKER_KNOWLEDGE_SET)
LATENT_DIM = 5
NUM_CLASSES = 2
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================================
# --- 2. 主训练函数 ---
# ==========================================================
def main():
    set_seed(2025)
    print("==========================================================")
    print("🚀 STEP 1 (Final): 训练上下文提取CAE引擎 (均衡采样+数据清洗版)...")
    print(f"   >>> 攻击者认知边界 (输入维度): {FEATURE_DIM} 维 <<<")
    print("==========================================================")

    # --- 1. 加载数据和Scaler ---
    print("正在加载训练集和全局Scaler...")
    try:
        df_train_full = pd.read_csv(TRAIN_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到核心文件 - {e}");
        return

    # ==========================================================
    # 🧼 [关键修复] 数据净化模块
    # ==========================================================
    print("正在进行数据净化 (去除 Inf/NaN)...")
    original_len = len(df_train_full)

    # 1. 替换 inf 为 nan
    df_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 获取Scaler所需的特征列名
    full_feature_names = scaler.feature_names_in_

    # 3. 丢弃任何含有 NaN 的行 (仅检查涉及到的特征列)
    # 我们选择直接丢弃，因为含有 inf 的流量通常是无效的统计异常，不适合用于训练生成器
    df_train_full.dropna(subset=full_feature_names, inplace=True)

    new_len = len(df_train_full)
    if new_len < original_len:
        print(f"⚠️ 警告: 丢弃了 {original_len - new_len} 条包含无穷大/缺失值的脏数据。")
        print(f"   (剩余有效样本: {new_len})")
    # ==========================================================

    # --- 2. 准备特征(X)和条件标签(C) ---
    # 使用 DataFrame 进行 transform，可以保留特征名，消除 Warning
    X_full_scaled = scaler.transform(df_train_full[full_feature_names])
    df_full_scaled = pd.DataFrame(X_full_scaled, columns=full_feature_names)

    X_scaled = df_full_scaled[ATTACKER_KNOWLEDGE_SET].values
    y_labels = df_train_full['label'].values

    print("数据准备完毕。")

    C_one_hot = np.zeros((len(y_labels), NUM_CLASSES))
    C_one_hot[np.arange(len(y_labels)), y_labels] = 1

    # --- 3. 划分训练/验证集 ---
    X_train, X_val, C_train, C_val = train_test_split(
        X_scaled, C_one_hot, test_size=VALIDATION_SPLIT, random_state=2025,
        stratify=C_one_hot.argmax(axis=1)
    )

    # --- 4. 计算采样权重 (均衡采样) ---
    print("[关键] 正在计算 WeightedRandomSampler 权重...")
    train_targets = C_train.argmax(axis=1)
    class_counts = np.bincount(train_targets)

    # 防止某个类别被清洗没了导致除0错误
    if len(class_counts) < 2 or 0 in class_counts:
        print("错误: 数据清洗后某个类别的样本数为0，请检查数据源。")
        return

    print(f"   -> 训练集分布: Benign={class_counts[0]}, Bot={class_counts[1]}")

    class_weights = 1. / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    train_sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(C_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_c = torch.tensor(C_val, dtype=torch.float32).to(device)

    # --- 5. 训练模型 ---
    model = ConditionalAutoencoder(FEATURE_DIM, LATENT_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("\n开始训练CAE模型...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x_batch, c_batch in train_loader:
            x_batch, c_batch = x_batch.to(device), c_batch.to(device)
            recon, _ = model(x_batch, c_batch)
            loss = criterion(recon, x_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            recon_val, _ = model(val_tensor_x, val_tensor_c)
            val_loss = criterion(recon_val, val_tensor_x).item()
            if (epoch + 1) % 10 == 0:
                print(
                    f"  -> Epoch {epoch + 1:3d}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CAE_MODEL_PATH)

    print("\n--- 训练完成 ---")
    print(f"CAE模型已保存: {CAE_MODEL_PATH}")


if __name__ == "__main__":
    main()