# config.py (THE FINAL ROBUST HYBRID VERSION - LOGICALLY CONSISTENT & LOG-SCALED)
import pandas as pd
import torch
import numpy as np
import random
import os
from sklearn.preprocessing import MinMaxScaler  # ✅ 新增导入


def set_seed(seed_value=2025):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"✅ 全局随机种子已固定为: {seed_value}")


# =================================================================
# --- 自定义 Log-MinMax Scaler (解决长尾分布问题) ---
# =================================================================
# config.py 中的 LogMinMaxScaler 类 (修复版)

class LogMinMaxScaler:
    """
    自定义缩放器：先进行 Log1p 变换，再进行 MinMax 缩放。
    解决网络流量特征（如 Duration, Bytes）跨度过大导致的长尾分布问题。
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # 记录列名 (如果是DataFrame)
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns

        # ✅ 核心修复: 强制将数据截断为非负数 (处理脏数据中的负值)
        # 将 DataFrame 或 Numpy 数组中的负数全部置为 0
        X_safe = np.maximum(X, 0)

        # 1. Log变换: log(1 + x)
        X_log = np.log1p(X_safe)

        # 再次清洗: 万一 log 产生了 inf (虽然 max(0) 后不太可能，但为了稳健)
        if isinstance(X_log, pd.DataFrame):
            X_log.replace([np.inf, -np.inf], 0, inplace=True)
            X_log.fillna(0, inplace=True)
        else:
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)

        # 2. MinMax fit
        self.scaler.fit(X_log)
        return self

    def transform(self, X):
        # ✅ 核心修复: 同样在 transform 时强制非负
        X_safe = np.maximum(X, 0)

        # 1. Log变换
        X_log = np.log1p(X_safe)

        # 清洗潜在的 inf
        if isinstance(X_log, pd.DataFrame):
            X_log.replace([np.inf, -np.inf], 0, inplace=True)
            X_log.fillna(0, inplace=True)
        else:
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)

        # 2. MinMax transform
        return self.scaler.transform(X_log)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        # 1. MinMax inverse
        X_log = self.scaler.inverse_transform(X_scaled)
        # 2. Log inverse: exp(x) - 1
        X_original = np.expm1(X_log)
        # 3. 强制非负
        return np.maximum(X_original, 0)


print("✅ LogMinMaxScaler 类已加载 (Config内嵌版)")

# =================================================================
# --- 最终特征体系：逻辑自洽版 (Hard Constraints Ready) ---
# =================================================================

# ✅ 1. 行动集 (ATTACKER_ACTION_SET) - 核心预测目标
# 这些是模型(LSTM/CAE)直接修改或生成的变量。必须是相互独立的。
ATTACKER_ACTION_SET = sorted([
    # --- 空间域 (独立变量) ---
    'Total Fwd Packets',
    'Total Backward Packets',  # 注意：Bwd Packets 也是独立的，应该预测
    'Average Packet Size',  # 预测平均包大小，而不是总长度（更易学习）

    # --- 时间域 (独立变量) ---
    'Flow Duration',
    'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std',
    'Active Mean', 'Idle Mean',
])

# ✅ 2. 可计算集 (CALCULABLE_SET)
# 这些变量将通过数学公式强制计算得出，绝不让神经网络预测！
# 这样可以保证 100% 的数学逻辑自洽，攻击者无法抓到把柄。
CALCULABLE_SET = sorted([
    'Total Length of Fwd Packets',  # = Total Fwd Pkts * Avg Pkt Size (近似)
    'Total Length of Bwd Packets',  # = Total Bwd Pkts * Avg Pkt Size (近似)
    'Flow Bytes/s',
    'Flow Packets/s',
    'Packet Length Mean',
    'Down/Up Ratio',
    # 如果原本有 'Total Length'，在这里算
])

# ✅ 3. 复杂关联集 (COMPLEX_SET)
# 这些是难以通过简单公式计算的统计特征（如极值、方差）。
# 依然交给 LSTM Predictor (TIER 3) 去预测。
COMPLEX_SET = sorted([
    # 包长统计细节
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Std',
    'Packet Length Std', 'Packet Length Variance',

    # 时间极值
    'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Max', 'Bwd IAT Min'
])

# ✅ 4. 防御者集 (DEFENDER_SET)
DEFENDER_SET = sorted(list(set(ATTACKER_ACTION_SET) | set(CALCULABLE_SET) | set(COMPLEX_SET)))

# ✅ 5. 认知集 (ATTACKER_KNOWLEDGE_SET)
# CAE 输入。可以包含 CALCULABLE 的特征，因为输入时是看真实数据的。
ATTACKER_KNOWLEDGE_SET = sorted(list(set(ATTACKER_ACTION_SET) | {
    'Flow Bytes/s', 'Flow Packets/s',
    'Packet Length Mean',
    'Flow IAT Max', 'Fwd Packet Length Max'
}))

print("特征体系加载完毕:")
print(f"  - ACTION_SET: {len(ATTACKER_ACTION_SET)} 维 (空间+时间)")
print(f"  - CALCULABLE_SET: {len(CALCULABLE_SET)} 维")
print(f"  - COMPLEX_SET: {len(COMPLEX_SET)} 维 (待预测)")
print(f"  - DEFENDER_SET: {len(DEFENDER_SET)} 维 (总目标)")
print(f"  - KNOWLEDGE_SET: {len(ATTACKER_KNOWLEDGE_SET)} 维 (CAE输入)")

# --- 交叉验证 ---
assert set(ATTACKER_ACTION_SET).issubset(set(ATTACKER_KNOWLEDGE_SET)), "行动集必须是认知集的子集!"
assert set(ATTACKER_KNOWLEDGE_SET).issubset(set(DEFENDER_SET)), "认知集必须是防御者集的子集!"
print("✅ 特征集逻辑自洽性通过验证。")