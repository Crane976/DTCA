# models/train_xgboost_hunter.py (FINAL COMPLETE VERSION)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score  # 导入 f1_score
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed

# --- 配置区 ---
TRAIN_SET_PATH = os.path.join(project_root, 'data', 'splits', 'training_set.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')
MODELS_DIR = os.path.join(project_root, 'models')
HUNTER_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_hunter.pkl')


def main():
    set_seed(2025)
    print("=" * 60)
    print("🚀 训练 XGBoost Hunter (Recall 强化版 - 含最终评估)...")
    print("=" * 60)

    # 1. 加载与清洗
    df_train = pd.read_csv(TRAIN_SET_PATH)
    df_test = pd.read_csv(TEST_SET_PATH)  # 加载测试集
    scaler = joblib.load(SCALER_PATH)
    feature_names = scaler.feature_names_in_

    # 清洗
    df_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train.dropna(subset=feature_names, inplace=True)

    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.dropna(subset=feature_names, inplace=True)

    # 2. 划分与 1:1 采样
    X_train_full = df_train[feature_names]
    y_train_full = df_train['label']

    # 划分验证集
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=2025, stratify=y_train_full
    )

    # 构造训练集 (1:1)
    df_pool = pd.concat([X_train_split, y_train_split], axis=1)
    df_bot = df_pool[df_pool['label'] == 1]
    df_benign = df_pool[df_pool['label'] == 0]

    # 1:1 采样
    df_benign_sampled = df_benign.sample(n=len(df_bot), random_state=2025)
    df_train_balanced = pd.concat([df_bot, df_benign_sampled])

    print(f"   -> 训练样本 (Balanced 1:1): {len(df_train_balanced)}")

    # 3. 缩放
    X_train_final = scaler.transform(df_train_balanced[feature_names])
    y_train_final = df_train_balanced['label']
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(df_test[feature_names])  # 缩放测试集
    y_test = df_test['label'].values

    # 4. 参数搜索
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1,
        random_state=2025
    )

    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [6, 8, 10, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    print("\n[步骤2] 正在搜索最佳参数 (RandomizedSearch)...")
    search = RandomizedSearchCV(
        xgb_clf, param_dist, n_iter=10, scoring='f1', cv=3, verbose=1, n_jobs=-1, random_state=2025
    )
    search.fit(X_train_final, y_train_final)

    best_model = search.best_estimator_
    print(f"   -> 最佳参数: {search.best_params_}")

    # 5. 阈值寻优
    print("\n[步骤3] 寻找最佳阈值...")
    val_probs = best_model.predict_proba(X_val_scaled)[:, 1]
    best_thr, best_f1 = 0.5, 0

    for thr in np.arange(0.1, 0.99, 0.01):
        y_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val_split, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"✅ 最佳阈值: {best_thr:.2f} (Val F1: {best_f1:.4f})")

    joblib.dump(best_model, HUNTER_MODEL_PATH)
    print("✅ 模型已保存")

    # --- 6. 最终测试集评估 (新增) ---
    print(f"\n--- 'XGBoost Hunter'在【留出测试集】上的真实性能报告 (阈值={best_thr:.2f}) ---")
    test_probs = best_model.predict_proba(X_test_scaled)[:, 1]
    y_test_pred = (test_probs >= best_thr).astype(int)

    print(classification_report(y_test, y_test_pred, target_names=['Benign (0)', 'Bot (1)'], digits=4))


if __name__ == "__main__":
    main()