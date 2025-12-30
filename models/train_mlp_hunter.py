# models/train_mlp_hunter.py (FINAL FIXED VERSION With Data Cleaning)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Path Setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier, FocalLoss

# Configuration
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MLP_HUNTER_MODEL_PATH = os.path.join(project_root, 'models', 'mlp_hunter.pt')

FEATURE_DIM = len(DEFENDER_SET)
EPOCHS = 100
BATCH_SIZE = 256
VALIDATION_SPLIT = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 2025
BEST_PARAMS = {'learning_rate': 0.0005}


def clean_data(df, feature_cols):
    """统一的数据清洗函数: 替换Inf并丢弃NaN"""
    # 1. 替换 Inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 2. 丢弃包含NaN的行 (仅检查特征列)
    df.dropna(subset=feature_cols, inplace=True)
    return df


def main():
    set_seed(RANDOM_SEED)
    print("=" * 60)
    print("🚀 开始训练 ResNet-MLP Hunter (修复版: 含数据清洗)...")
    print("=" * 60)

    # --- 1. 加载数据 ---
    df_train_full = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = scaler.feature_names_in_

    # --- ✅ 关键修复: 数据清洗 ---
    print("正在清洗数据 (去除 Inf/NaN)...")
    len_train_before = len(df_train_full)
    df_train_full = clean_data(df_train_full, feature_names)
    print(f"   -> 训练集清洗掉 {len_train_before - len(df_train_full)} 条脏数据")

    len_test_before = len(df_test)
    df_test = clean_data(df_test, feature_names)
    print(f"   -> 测试集清洗掉 {len_test_before - len(df_test)} 条脏数据")

    # --- 2. 转换与划分 ---
    X_test_scaled = scaler.transform(df_test[feature_names].values)
    y_test = df_test['label'].values

    X_train_full_scaled = scaler.transform(df_train_full[feature_names].values)
    y_train_full = df_train_full['label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_scaled, y_train_full, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED, stratify=y_train_full
    )

    # --- 3. 加权采样器 ---
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train.astype(int)])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False)

    val_tensor_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_tensor_y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # --- 4. 训练 ---
    benign_ratio = (y_train_full == 0).sum() / len(y_train_full)
    model = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)  # alpha调为0.5因为用了Balanced Sampler
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_PARAMS['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    print("\n[步骤1] 正在训练模型...")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for x_batch, y_batch in pbar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss = criterion(model(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor_x), val_tensor_y).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MLP_HUNTER_MODEL_PATH)

    print(f"\n✅ 训练完成，最佳验证损失: {best_val_loss:.6f}")

    # --- 5. 阈值与评估 ---
    print("\n[步骤2] 寻找最佳决策阈值...")
    final_model = MLP_Classifier(feature_dim=FEATURE_DIM).to(device)
    final_model.load_state_dict(torch.load(MLP_HUNTER_MODEL_PATH, map_location=device))
    final_model.eval()

    with torch.no_grad():
        val_probs = final_model.predict(val_tensor_x).cpu().numpy()

    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.01, 1.0, 0.01):
        y_val_pred = (val_probs > threshold).astype(int)
        current_f1 = f1_score(y_val, y_val_pred, pos_label=1)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f"✅ 最佳阈值: {best_threshold:.2f} (F1: {best_f1:.4f})")

    with torch.no_grad():
        test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        test_probs = final_model.predict(test_tensor_x).cpu().numpy()
        y_pred = (test_probs > best_threshold).astype(int)

    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()