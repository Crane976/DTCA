# models/train_xgboost_hunter.py (Final Confirmed Version)
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from config import UNIFIED_FEATURE_SET  # 导入我们唯一的标准

# ==========================================================
# --- 中文显示配置 ---
# ==========================================================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    print("已设置字体为 SimHei。")
except Exception:
    print("警告: 未找到SimHei字体，中文可能无法显示。")

# ==========================================================
# --- 1. 配置区 ---
# ==========================================================
DATA_DIR = r'D:\DTCA\data\preprocessed'  # 输入输出都在preprocessed
MODELS_DIR = r'D:\DTCA\models'
FIGURES_DIR = r'D:\DTCA\figures'

benign_processed_path = os.path.join(DATA_DIR, 'benign_traffic_processed.csv')
bot_processed_path = os.path.join(DATA_DIR, 'bot_traffic_processed.csv')

hunter_model_path = os.path.join(MODELS_DIR, 'xgboost_hunter.pkl')
test_set_path = os.path.join(DATA_DIR, 'evaluation_test_set.csv')


# ==========================================================
# --- 2. 主训练函数 ---
# ==========================================================
def main():
    print("=============================================")
    print("🚀 开始训练'最强猎手' (XGBoost Classifier)...")
    print("=============================================")

    print("正在加载和准备数据...")
    try:
        df_benign = pd.read_csv(benign_processed_path)
        df_bot = pd.read_csv(bot_processed_path)
    except FileNotFoundError as e:
        print(f"错误: 找不到输入文件 - {e}");
        return

    # ✅ 关键修正：强制所有数据都遵循统一的特征标准
    # 这可以防止因step2脚本输出列顺序不同等意外情况导致的错误
    df_benign = df_benign[UNIFIED_FEATURE_SET]
    df_bot = df_bot[UNIFIED_FEATURE_SET]

    df_benign['label'] = 0
    df_bot['label'] = 1

    # 合并前打乱一下，增加随机性
    df_full = pd.concat([df_benign, df_bot], ignore_index=True).sample(frac=1, random_state=42)

    X = df_full[UNIFIED_FEATURE_SET]  # 明确使用统一特征集
    y = df_full['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    print(f"训练集中Bot样本比例: {y_train.mean():.2%}")
    print(f"测试集中Bot样本比例: {y_test.mean():.2%}")

    df_test = pd.concat([X_test, y_test], axis=1)
    df_test.to_csv(test_set_path, index=False)
    print(f"\n✅ 独立的测试集已保存到: {test_set_path}")

    print("\n正在训练XGBoost模型...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    hunter_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, n_estimators=200, max_depth=6,
        learning_rate=0.1, n_jobs=-1, random_state=42
    )
    hunter_model.fit(X_train, y_train)

    joblib.dump(hunter_model, hunter_model_path)
    print(f"✅ '猎手'模型已保存到: {hunter_model_path}")

    print("\n--- '猎手'在独立测试集上的基线性能报告 ---")
    y_pred = hunter_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Bot']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Bot'], yticklabels=['Benign', 'Bot'])
    plt.title("'猎手'模型在独立测试集上的混淆矩阵")
    plt.xlabel('预测标签');
    plt.ylabel('真实标签')
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "hunter_baseline_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print(f"✅ 混淆矩阵已保存到: {cm_path}")
    plt.show()


if __name__ == "__main__":
    main()