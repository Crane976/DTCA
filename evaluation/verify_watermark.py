import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 配置
GENERATED_PATH = os.path.join(project_root, 'data', 'generated', 'final_camouflage_bot_hard_constrained.csv')
TEST_SET_PATH = os.path.join(project_root, 'data', 'splits', 'holdout_test_set.csv')

WATERMARK_KEY = 97
WATERMARK_FEATURE = 'Flow Duration'


def verify(df, name):
    print(f"\n--- 验证数据集: {name} ---")
    if WATERMARK_FEATURE not in df.columns:
        print("❌ 特征缺失，无法验证")
        return

    values = df[WATERMARK_FEATURE].values.astype(int)
    # 提取逻辑: 余数为 0 即为我方流量
    matches = (values % WATERMARK_KEY == 0)

    accuracy = np.mean(matches)
    print(f"   水印检出率 (Extraction Rate): {accuracy * 100:.2f}%")
    return accuracy


def main():
    print(f"🔐 开始水印溯源验证 (Key={WATERMARK_KEY})...")

    # 1. 验证伪装流量 (应该接近 100%)
    df_gen = pd.read_csv(GENERATED_PATH)
    acc_gen = verify(df_gen, "伪装诱饵流量 (Self)")

    # 2. 验证真实流量 (应该接近 1/Key, 极低)
    # 这代表了"误伤率"，即把真实流量误认为是我方诱饵的概率
    df_test = pd.read_csv(TEST_SET_PATH)
    acc_test = verify(df_test, "真实背景流量 (Others)")

    print("\n" + "=" * 40)
    print(f"📊 溯源性能总结:")
    print(f"   - 自身识别率 (TPR): {acc_gen * 100:.2f}% (越高越好)")
    print(f"   - 误伤率 (FPR):     {acc_test * 100:.2f}% (理论值约 {100 / WATERMARK_KEY:.2f}%)")
    print("=" * 40)


if __name__ == "__main__":
    main()