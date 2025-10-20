# config.py (v4 - Final & Collinearity Fixed)

# ==============================================================================
# --- 统一特征集定义 (唯一真实来源) ---
# ==============================================================================

# 1. 定义一个包含所有可能相关特征的原始并集
_UNIFIED_FEATURE_SET_RAW = sorted(list(set([
    # 来自旧的 BENIGN_FEATURE_SET
    'Flow Duration', 'Fwd IAT Total', 'Flow IAT Max', 'Idle Max',
    'Fwd IAT Std', 'Flow IAT Std', 'Idle Mean', 'Flow IAT Mean', 'Fwd IAT Mean',
    'Fwd Header Length', 'Total Fwd Packets', 'Subflow Fwd Packets',
    'act_data_pkt_fwd', 'Idle Min', 'Idle Std',
    # 来自我们分析出的 BOT_FEATURE_SET
    'Bwd IAT Total', 'Active Max', 'Active Min', 'Active Std', 'Active Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Mean'
])))

# 2. ✅ 核心修正: 从原始并集中，移除已知的冗余/共线性特征
# 我们根据之前的分析，'Subflow Fwd Packets' 与 'Total Fwd Packets' 高度共线，移除前者。
# 另外，我们有两个 'Fwd IAT Max'，这是笔误，我们也清理掉。
# 确保列表中的特征名是唯一的。
_temp_set = set(_UNIFIED_FEATURE_SET_RAW)
_temp_set.discard('Subflow Fwd Packets') # 移除共线性特征

# 最终的、干净的、数学上稳健的统一特征集
UNIFIED_FEATURE_SET = sorted(list(_temp_set))

# 打印最终结果以供验证
print(f"Config loaded: Final Unified Feature Set ({len(UNIFIED_FEATURE_SET)} features).")


# ==============================================================================
# --- LSTM 预测目标定义 ---
# ==============================================================================

TARGET_FIELDS_FOR_LSTM = [
    'Flow Duration',
    'Flow IAT Mean',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Active Mean',
    'Idle Mean'
]

# 检查目标特征是否都在统一特征集中
for f in TARGET_FIELDS_FOR_LSTM:
    if f not in UNIFIED_FEATURE_SET:
        print(f"警告: 目标特征 '{f}' 不在统一特征集中！")