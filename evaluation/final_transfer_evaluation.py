# evaluation/final_transfer_evaluation.py (FINAL: CNN REPLACES RF)
import pandas as pd
import numpy as np
import os
import sys
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.append(project_root)

from config import DEFENDER_SET, set_seed
from models.mlp_architecture import MLP_Classifier
# ✅ 新增导入 CNN
from models.cnn_architecture import CNN_Classifier

# ==============================================================================
# 🎯 最佳阈值配置 (Hardcoded based on training logs)
# ==============================================================================
MODEL_THRESHOLDS = {
    "KNN Hunter": 0.90,
    "1D-CNN Hunter": 0.90,  # ✅ 新增 CNN 阈值
    "XGBoost Hunter": 0.96,
    "MLP Hunter": 0.76
}


# ------------------------------------------------------------------------------
# 2. 核心评估函数
# ------------------------------------------------------------------------------
def evaluate_hunter(hunter_name, hunter_model, X_cam_scaled, X_benign_test, X_bot_test, y_bot_test, device,
                    threshold=0.5):
    print("\n" + "=" * 50)
    print(f"--- 正在评估对抗: {hunter_name} ---")
    print(f"    👉 使用最佳决策阈值: {threshold:.2f}")

    # --- 统一预测接口 ---

    # A. PyTorch 模型 (MLP & CNN)
    if isinstance(hunter_model, nn.Module):
        hunter_model.eval()
        with torch.no_grad():
            # 转换为Tensor
            # 注意: CNN 需要 input shape (N, Features), 它内部会unsqueeze
            t_cam = torch.tensor(X_cam_scaled, dtype=torch.float32).to(device)
            t_benign = torch.tensor(X_benign_test, dtype=torch.float32).to(device)
            t_bot = torch.tensor(X_bot_test, dtype=torch.float32).to(device)

            # 获取概率并应用阈值
            preds_cam = (hunter_model.predict(t_cam) > threshold).int().cpu().numpy().flatten()
            preds_benign = (hunter_model.predict(t_benign) > threshold).int().cpu().numpy().flatten()
            preds_bot = (hunter_model.predict(t_bot) > threshold).int().cpu().numpy().flatten()

    # B. Sklearn/XGBoost 模型 (XGB, KNN)
    else:
        def batch_predict_with_threshold(model, data, thr, batch_size=5000):
            n_samples = len(data)
            preds = []
            for i in range(0, n_samples, batch_size):
                batch = data[i:i + batch_size]
                probs = model.predict_proba(batch)[:, 1]
                batch_preds = (probs >= thr).astype(int)
                preds.extend(batch_preds)
            return np.array(preds)

        preds_cam = batch_predict_with_threshold(hunter_model, X_cam_scaled, threshold)
        preds_benign = batch_predict_with_threshold(hunter_model, X_benign_test, threshold)
        preds_bot = batch_predict_with_threshold(hunter_model, X_bot_test, threshold)

    # --- 计算指标 ---
    deceived_count = np.sum(preds_cam == 0)
    deception_rate = deceived_count / len(X_cam_scaled) * 100

    base_tp = np.sum(preds_bot == 1)
    base_fn = len(y_bot_test) - base_tp
    recall = base_tp / (base_tp + base_fn) * 100

    base_fp = np.sum(preds_benign == 1)

    failed_deception_count = len(X_cam_scaled) - deceived_count
    base_alerts = base_fp + base_tp
    mix_alerts = base_alerts + failed_deception_count

    dsr = (failed_deception_count / mix_alerts) * 100 if mix_alerts > 0 else 0
    base_precision = (base_tp / base_alerts) * 100 if base_alerts > 0 else 0
    hunter_precision_decayed = (base_tp / mix_alerts) * 100 if mix_alerts > 0 else 0

    print(f"  - 伪装Bot被判为Benign (隐身): {deceived_count} / {len(X_cam_scaled)} ({deception_rate:.2f}%)")
    print(f"  - 伪装Bot被判为Bot (诱饵): {failed_deception_count} / {len(X_cam_scaled)} ({100 - deception_rate:.2f}%)")
    print(f"  - 真实Bot捕获率 (Recall): {recall:.2f}%")
    print(f"  - 误报数 (Benign -> Bot): {base_fp}")
    print("---------------------------------------------")
    print(f"  🎯 警报污染率 (DSR): {dsr:.2f}%")
    print(f"  📉 精确率从 {base_precision:.2f}% 衰减为: {hunter_precision_decayed:.2f}%")

    return {
        "Hunter": hunter_name,
        "Threshold": threshold,
        "Evasion Rate (%)": deception_rate,
        "Decoy Rate (%)": 100 - deception_rate,
        "Recall (%)": recall,
        "Base Precision (%)": base_precision,
        "Decayed Precision (%)": hunter_precision_decayed,
        "DSR (Pollution) (%)": dsr
    }


# ------------------------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------------------------
def main():
    set_seed(2025)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置路径
    CAMOUFLAGE_BOT_PATH = os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot_hard_constrained.csv')
    TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')
    SCALER_PATH = os.path.join(project_root, 'models', 'global_scaler.pkl')

    # ✅ 模型路径字典 (RF -> CNN)
    MODEL_PATHS = {
        "1D-CNN Hunter": os.path.join(project_root, 'models', 'cnn_hunter.pt'),  # 新增
        "XGBoost Hunter": os.path.join(project_root, 'models', 'xgboost_hunter.pkl'),
        "KNN Hunter": os.path.join(project_root, 'models', 'knn_hunter.pkl'),
        "MLP Hunter": os.path.join(project_root, 'models', 'mlp_hunter.pt'),
    }

    print("=" * 50);
    print("🚀 最终迁移攻击评估 (含 1D-CNN)...");
    print("=" * 50)

    # 1. 加载数据
    print("\n[步骤1] 正在加载数据...")
    try:
        df_cam = pd.read_csv(CAMOUFLAGE_BOT_PATH)
        df_test = pd.read_csv(TEST_SET_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"错误: {e}");
        return

    # 2. 准备数据
    df_cam.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cam.dropna(subset=DEFENDER_SET, inplace=True)
    df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test.dropna(subset=DEFENDER_SET, inplace=True)

    print(f"使用 {len(DEFENDER_SET)} 维特征进行评估...")
    X_cam_scaled = scaler.transform(df_cam[DEFENDER_SET])
    X_benign_scaled = scaler.transform(df_test[df_test['label'] == 0][DEFENDER_SET])
    X_bot_scaled = scaler.transform(df_test[df_test['label'] == 1][DEFENDER_SET])
    y_bot_numpy = df_test[df_test['label'] == 1]['label'].values

    # 3. 加载模型并评估
    print("\n[步骤2] 开始评估...")
    results_list = []

    for name, path in MODEL_PATHS.items():
        try:
            threshold = MODEL_THRESHOLDS.get(name, 0.5)

            if name == "MLP Hunter":
                model = MLP_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            elif name == "1D-CNN Hunter":  # ✅ 新增 CNN 处理逻辑
                model = CNN_Classifier(feature_dim=len(DEFENDER_SET)).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            else:
                # Sklearn/XGB
                model = joblib.load(path)
                result = evaluate_hunter(name, model, X_cam_scaled, X_benign_scaled, X_bot_scaled, y_bot_numpy, device,
                                         threshold)

            results_list.append(result)
        except Exception as e:
            print(f"⚠️ 无法加载或评估 {name}: {e}")

    # 4. 汇总
    print("\n\n" + "=" * 100)
    print("--- 最终评估汇总报告 (Final Results) ---")
    print("=" * 100)
    if results_list:
        results_df = pd.DataFrame(results_list).set_index("Hunter")
        print(results_df.to_string(float_format="%.2f"))
    else:
        print("无结果。")


if __name__ == "__main__":
    main()