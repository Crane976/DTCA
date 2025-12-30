# evaluation/prove_hunter_capability.py (FINAL: ALL HUNTERS WITH BEST THRESHOLDS)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
from sklearn.metrics import classification_report

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed, LogMinMaxScaler
from models.mlp_architecture import MLP_Classifier
# ✅ 1. 导入 CNN
from models.cnn_architecture import CNN_Classifier

# --- 配置区 ---
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

# ✅ 2. 模型路径 (RF -> CNN)
MLP_PATH = os.path.join(project_root, 'models', 'mlp_hunter.pt')
XGB_PATH = os.path.join(project_root, 'models', 'xgboost_hunter.pkl')
KNN_PATH = os.path.join(project_root, 'models', 'knn_hunter.pkl')
CNN_PATH = os.path.join(project_root, 'models', 'cnn_hunter.pt')

# ✅ 3. 默认阈值配置 (Hardcoded)
THRESHOLDS = {
    "KNN": 0.50,
    "CNN": 0.50,
    "XGB": 0.50,
    "MLP": 0.50
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    set_seed(2025)
    print("🚀 开始 Hunter 能力自证评估 (Balanced Test Environment)...")
    print(f"   👉 使用默认阈值配置: {THRESHOLDS}")

    # 1. 加载原始测试集与 Scaler
    df_test = pd.read_csv(TEST_SET_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 2. 数据清洗
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.dropna(subset=scaler.feature_names_in_, inplace=True)

    # 3. 构造 1:1 均衡测试集
    print("\n[步骤1] 构造均衡测试集 (1:1 Sampling)...")
    df_bot = df_test[df_test['label'] == 1]
    n_bot = len(df_bot)

    # 随机抽取同等数量的 Benign
    df_benign = df_test[df_test['label'] == 0].sample(n=n_bot, random_state=2025)

    df_balanced = pd.concat([df_bot, df_benign])
    X_balanced = scaler.transform(df_balanced[scaler.feature_names_in_])
    y_balanced = df_balanced['label'].values

    print(f"   -> Bot样本: {n_bot}, Benign样本: {n_bot}, 总计: {len(df_balanced)}")

    # ==================================================
    # 4. 评估 MLP Hunter
    # ==================================================
    print("\n" + "=" * 40)
    print(f"🔬 评估 MLP Hunter (阈值: {THRESHOLDS['MLP']})")
    try:
        mlp = MLP_Classifier(len(scaler.feature_names_in_)).to(device)
        mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
        mlp.eval()

        with torch.no_grad():
            probs = mlp.predict(torch.tensor(X_balanced, dtype=torch.float32).to(device)).cpu().numpy()
            y_pred = (probs > THRESHOLDS['MLP']).astype(int)

        print(classification_report(y_balanced, y_pred, target_names=['Benign', 'Bot'], digits=4))
    except Exception as e:
        print(f"❌ MLP 评估失败: {e}")

    # ==================================================
    # 5. 评估 1D-CNN Hunter (新增)
    # ==================================================
    print("\n" + "=" * 40)
    print(f"🔬 评估 1D-CNN Hunter (阈值: {THRESHOLDS['CNN']})")
    try:
        cnn = CNN_Classifier(len(scaler.feature_names_in_)).to(device)
        cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
        cnn.eval()

        with torch.no_grad():
            probs = cnn.predict(torch.tensor(X_balanced, dtype=torch.float32).to(device)).cpu().numpy()
            y_pred = (probs > THRESHOLDS['CNN']).astype(int)

        print(classification_report(y_balanced, y_pred, target_names=['Benign', 'Bot'], digits=4))
    except Exception as e:
        print(f"❌ CNN 评估失败: {e}")

    # ==================================================
    # 6. 评估 XGBoost Hunter
    # ==================================================
    print("\n" + "=" * 40)
    print(f"🔬 评估 XGBoost Hunter (阈值: {THRESHOLDS['XGB']})")
    try:
        xgb_model = joblib.load(XGB_PATH)
        # ✅ 关键修改: 使用 predict_proba + 硬阈值
        probs = xgb_model.predict_proba(X_balanced)[:, 1]
        y_pred = (probs >= THRESHOLDS['XGB']).astype(int)

        print(classification_report(y_balanced, y_pred, target_names=['Benign', 'Bot'], digits=4))
    except Exception as e:
        print(f"❌ XGBoost 评估失败: {e}")

    # ==================================================
    # 7. 评估 KNN Hunter
    # ==================================================
    print("\n" + "=" * 40)
    print(f"🔬 评估 KNN Hunter (阈值: {THRESHOLDS['KNN']})")
    try:
        knn_model = joblib.load(KNN_PATH)
        # ✅ 关键修改: 使用 predict_proba + 硬阈值
        probs = knn_model.predict_proba(X_balanced)[:, 1]
        y_pred = (probs >= THRESHOLDS['KNN']).astype(int)

        print(classification_report(y_balanced, y_pred, target_names=['Benign', 'Bot'], digits=4))
    except Exception as e:
        print(f"❌ KNN 评估失败: {e}")


if __name__ == "__main__":
    main()