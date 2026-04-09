import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

# ---------------------- 1. 基础配置 ----------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 核心污染指标
POLLUTION_INDICATORS = [
    "Cd",
    "Hg",
    "As",
    "Pb",
    "Cr",
    "Cu",
    "Ni",
    "Zn",
    "BHC",
    "DDT",
    "BAP",
]
AGG_POLLUTION_COL = "Pollution status"  # 聚合列名称（强制固定）
POLLUTION_SCORE_PREFIX = "Pollution score_"

# AHP权重配置
BASIC_AHP_WEIGHTS = {
    "effective soil layer thickness": 10.2155 / 100,
    "pH": 5.1237 / 100,
    "soil bulk density": 2.9883 / 100,
    "biodiversity": 2.1970 / 100,
    "salinization degree": 1.4998 / 100,
    "total nitrogen": 1.2416 / 100,
    "available phosphorus": 0.7815 / 100,
    "available potassium": 0.6577 / 100,
    "cation exchange capacity": 0.5392 / 100,
    "Light-temperature production potential": 5.1822 / 100,
    "terrain gradient": 0.5517 / 100,
    "the capacity of irrigation and drainage": 1.4554 / 100,
    "groundwater depth": 0.8947 / 100,
    "surface soil texture": 4.2909 / 100,
    "carbon pool variation factor(arable land)": 1.1443 / 100,
    "soil organic carbon content": 2.3023 / 100,
    "profile configuration": 2.3023 / 100,
}

POLLUTION_AHP_WEIGHTS = {
    "Hg": 16.0745 / 100,
    "As": 10.5259 / 100,
    "Cd": 7.0476 / 100,
    "Pb": 3.4975 / 100,
    "Cr": 4.8605 / 100,
    "BAP": 5.2397 / 100,
    "DDT": 3.3283 / 100,
    "BHC": 2.5968 / 100,
    "Ni": 1.5412 / 100,
    "Cu": 1.1053 / 100,
    "Zn": 0.8248 / 100,
}

# 初始权重（聚合时动态更新污染程度权重）
AHP_WEIGHTS_DICT = {**BASIC_AHP_WEIGHTS, **POLLUTION_AHP_WEIGHTS}

# 编码映射
ENCODING_MAPPING = {
    "salinization degree": ["None to mild", "Mild to moderate", "Moderate to severe", "Unknowns"],
    "biodiversity": ["rich", "Moderate", "Not rich", "Unknowns"],
    "the capacity of irrigation and drainage": ["Fully satisfaction or satisfaction", "Satisfaction or basic satisfaction", "Basic satisfaction or dissatisfaction", "Unknowns"],
    "surface soil texture": ["loam", "clay", "sandy soil", "Gravely soil", "Unknowns"],
    "carbon pool variation factor(arable land)": [
        "Temperate/Northern Temperate arid areas",
        "Temperate/Northern Temperate humid areas",
        "Tropical mountainous areas",
        "Tropical dry regions",
        "Tropical humid/wetland areas",
        "Unknowns",
    ],
    "profile configuration": [
        "Whole body loam、loam/sand/loam",
        "loam/clay/loam",
        "sand/clay/sand、loam/clay/clay、loam/sand/sand",
        "sand/clay/clay",
        "clay/sand/clay、Whole body clay、clay/sand/sand",
        "Whole body sandy",
        "Whole body gravel",
        "Unknowns",
    ],
    "soil bulk density": ["Moderate", "Slightly light", "heavy", "Unknowns"],
}

# 污染物阈值
two_level_pollutants = ["Cd", "Hg", "As", "Pb", "Cr"]
POLLUTANT_THRESHOLDS = {
    "Hg": {
        (0, 5.5): [1.3, 2],
        (5.5, 6.5): [1.8, 2.5],
        (6.5, 7.5): [2.4, 4],
        (7.5, 14): [3.4, 6],
    },
    "As": {
        (0, 5.5): [40, 200],
        (5.5, 6.5): [40, 150],
        (6.5, 7.5): [30, 120],
        (7.5, 14): [25, 100],
    },
    "Cd": {
        (0, 5.5): [0.3, 1.5],
        (5.5, 6.5): [0.3, 2.0],
        (6.5, 7.5): [0.3, 3],
        (7.5, 14): [0.6, 4],
    },
    "Pb": {
        (0, 5.5): [70, 400],
        (5.5, 6.5): [90, 500],
        (6.5, 7.5): [120, 700],
        (7.5, 14): [170, 1000],
    },
    "Cr": {
        (0, 5.5): [150, 800],
        (5.5, 6.5): [150, 850],
        (6.5, 7.5): [200, 1000],
        (7.5, 14): [250, 1300],
    },
    "Cu": {(0, 5.5): [50], (5.5, 6.5): [50], (6.5, 7.5): [100], (7.5, 14): [100]},
    "Ni": {(0, 5.5): [60], (5.5, 6.5): [70], (6.5, 7.5): [100], (7.5, 14): [190]},
    "Zn": {(0, 5.5): [200], (5.5, 6.5): [200], (6.5, 7.5): [250], (7.5, 14): [300]},
    "BHC": {(0, 14): [0.1]},
    "DDT": {(0, 14): [0.1]},
    "BAP": {(0, 14): [0.55]},
}


# ---------------------- 2. 核心工具函数 ----------------------
def clean_and_convert(df):
    """数据清洗"""
    df_clean = df.copy(deep=True)
    site_col = "location" if "location" in df_clean.columns else None

    for col in df_clean.columns:
        if col == site_col:
            continue
        df_clean[col] = df_clean[col].replace(
            ["none", "NA", "-", "——", "null", ""], np.nan
        )
        try:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="raise").astype(
                np.float64
            )
        except:
            df_clean[col] = df_clean[col].fillna("Unknowns").astype(str).str.strip()
            if col in ENCODING_MAPPING:
                cat_map = {v: i for i, v in enumerate(ENCODING_MAPPING[col])}
            else:
                unique_vals = df_clean[col].unique().tolist()
                if "Unknowns" not in unique_vals:
                    unique_vals.append("Unknowns")
                cat_map = {v: i for i, v in enumerate(unique_vals)}
            df_clean[col] = (
                df_clean[col].map(cat_map).fillna(len(cat_map)).astype(np.float64)
            )

    for col in df_clean.columns:
        if col != site_col:
            df_clean[col] = df_clean[col].fillna(0.0).astype(np.float64)

    return df_clean


def map_pollution_cols(cols, df_cols, is_aggregated):
    """映射列（聚合场景强制加入污染程度列）"""
    mapped_cols = []
    # 第一步：加入非污染指标
    for col in cols:
        if col not in POLLUTION_INDICATORS:
            mapped_cols.append(col)
    # 第二步：聚合场景 → 强制加入污染程度列（如果存在）
    if is_aggregated and AGG_POLLUTION_COL in df_cols:
        if AGG_POLLUTION_COL not in mapped_cols:
            mapped_cols.append(AGG_POLLUTION_COL)
            print(
                f"📌 Aggregate Scenario Mandatory Supplementary Columns：{AGG_POLLUTION_COL}（The original mapping logic was missing and has been fixed）"
            )
    # 第三步：未聚合场景 → 加入原污染指标
    else:
        for col in cols:
            if col in POLLUTION_INDICATORS and col not in mapped_cols:
                mapped_cols.append(col)
    # 去重
    mapped_cols = list(dict.fromkeys(mapped_cols))
    print(
        f"🔍 Column mapping results: input{len(cols)}row → output{len(mapped_cols)}row | include pollution status：{AGG_POLLUTION_COL in mapped_cols}"
    )
    return mapped_cols


def calculate_pollutant_score(val, ph, pollutant):
    """污染物赋分"""
    if pd.isna(val) or val < 0:
        return 60

    ph_ranges = list(POLLUTANT_THRESHOLDS[pollutant].keys())
    target_range = None
    for ph_min, ph_max in ph_ranges:
        if ph_min <= ph <= ph_max:
            target_range = (ph_min, ph_max)
            break
    if target_range is None:
        target_range = ph_ranges[0] if ph_ranges else (0, 14)

    thresholds = POLLUTANT_THRESHOLDS[pollutant][target_range]

    if pollutant in two_level_pollutants:
        if val <= thresholds[0]:
            return 100
        elif val <= thresholds[1]:
            return 50
        else:
            return 10
    else:
        if val <= thresholds[0]:
            return 100
        else:
            return 50


def aggregate_pollution_indicators(df):
    """聚合污染指标（计算被聚合指标权重和）"""
    df_agg = df.copy(deep=True)
    site_col = "location" if "location" in df_agg.columns else None
    need_aggregate = False
    missing_details = {"missing columns": [], "Value missing": [], "Effective indicators": []}
    aggregated_pollution_cols = []

    # 第一步：逐一审视污染指标
    for col in POLLUTION_INDICATORS:
        if col not in df_agg.columns:
            need_aggregate = True
            missing_details["missing columns"].append(col)
        else:
            col_vals = df_agg[col].values
            has_valid = np.any(~np.isnan(col_vals)) and np.any(col_vals != 0.0)
            if not has_valid:
                need_aggregate = True
                missing_details["Value missing"].append(col)
            else:
                missing_details["Effective indicators"].append(col)
                aggregated_pollution_cols.append(col)

    # 第二步：强制聚合
    if not need_aggregate and len(missing_details["Effective indicators"]) < len(
        POLLUTION_INDICATORS
    ):
        need_aggregate = True
        print(
            f"⚠️  Partial pollution indicators are effective（{len(missing_details['Effective indicators'])}/{len(POLLUTION_INDICATORS)}），Force triggering aggregation"
        )

    # 第三步：未聚合 → 保留原指标
    if not need_aggregate:
        cache_cols = [
            col for col in df_agg.columns if col.startswith(POLLUTION_SCORE_PREFIX)
        ]
        df_agg = df_agg.drop(columns=cache_cols + [AGG_POLLUTION_COL], errors="ignore")
        AHP_WEIGHTS_DICT[AGG_POLLUTION_COL] = 0.0

        print(f"\n📊 Aggregation judgment result of pollution indicators: No aggregation required")
        print(f"   ✅ All{len(POLLUTION_INDICATORS)}pollution indicators are complete and have effective values")
        print(f"   📌 Retain the original pollution indicators：{', '.join(POLLUTION_INDICATORS)}")
        return df_agg, need_aggregate, aggregated_pollution_cols

    # 第四步：需要聚合 → 计算赋分缓存，生成污染程度列
    existing_pollution_cols = missing_details["Effective indicators"]
    aggregated_pollution_cols = existing_pollution_cols.copy()
    ph_vals = (
        df_agg["pH"].fillna(7.0).values
        if "pH" in df_agg.columns
        else np.ones(len(df_agg)) * 7.0
    )

    # 计算赋分缓存
    cache_cols = []
    for col in existing_pollution_cols:
        score_col = f"{POLLUTION_SCORE_PREFIX}{col}"
        df_agg[score_col] = [
            calculate_pollutant_score(val, ph, col)
            for val, ph in zip(df_agg[col].values, ph_vals)
        ]
        cache_cols.append(score_col)

    # 计算污染程度值
    if len(existing_pollution_cols) > 0:
        pollution_weights = np.array(
            [POLLUTION_AHP_WEIGHTS.get(col, 0.0) for col in existing_pollution_cols]
        )
        pollution_weights = (
            pollution_weights / np.sum(pollution_weights)
            if np.sum(pollution_weights) > 0
            else np.ones_like(pollution_weights) / len(existing_pollution_cols)
        )
        w_max = np.max(pollution_weights) if len(pollution_weights) > 0 else 0.0
        pollution_vals = df_agg[existing_pollution_cols].values
        pollution_vals = np.nan_to_num(pollution_vals, 0.0)
        c_max = np.max(pollution_vals, axis=1)
    else:
        w_max = 0.0
        c_max = np.zeros(len(df_agg))

    pollution_degree = np.sqrt(np.square(w_max) + np.square(c_max))

    # 移除原污染指标，添加聚合列
    df_agg = df_agg.drop(columns=POLLUTION_INDICATORS, errors="ignore")
    df_agg[AGG_POLLUTION_COL] = pollution_degree.astype(np.float64)

    # 计算被聚合指标权重和（核心：用于后续污染程度权重赋值）
    total_pollution_weight = sum(
        [POLLUTION_AHP_WEIGHTS.get(col, 0.0) for col in aggregated_pollution_cols]
    )
    AHP_WEIGHTS_DICT[AGG_POLLUTION_COL] = total_pollution_weight

    # 详细日志
    print(f"\n📊 Aggregation judgment result of pollution indicators: aggregation required (mandatory trigger）")
    print(
        f"   ❌ Columns missing pollution indicators：{', '.join(missing_details['missing columns']) if missing_details['missing columns'] else 'none'}"
    )
    print(
        f"   ❌ Value missing pollution indicators：{', '.join(missing_details['Value missing']) if missing_details['Value missing'] else 'none'}"
    )
    print(
        f"   ✅ Effective pollution indicators：{', '.join(missing_details['Effective indicators']) if missing_details['Effective indicators'] else 'none'}"
    )
    print(
        f"   📌 Weight of aggregated indicators：{total_pollution_weight:.6f} | Related indicators：{', '.join(aggregated_pollution_cols)}"
    )
    print(
        f"   📌 Original indicator weight details：{[(col, POLLUTION_AHP_WEIGHTS.get(col, 0.0)) for col in aggregated_pollution_cols]}"
    )

    return df_agg, need_aggregate, aggregated_pollution_cols


def df_to_safe_array(df):
    """转数组"""
    df_numeric = df.drop(columns=["location"], errors="ignore")
    df_numeric = df_numeric.drop(
        columns=[
            col for col in df_numeric.columns if col.startswith(POLLUTION_SCORE_PREFIX)
        ],
        errors="ignore",
    )
    safe_array = df_numeric.astype(np.float64).values.copy()
    return safe_array


def fill_nan_with_mean(arr):
    """填充空值"""
    arr_filled = arr.copy()
    for i in range(arr_filled.shape[1]):
        col = arr_filled[:, i]
        mean_val = np.nanmean(col) if not np.isnan(np.nanmean(col)) else 0.0
        arr_filled[np.isnan(col), i] = mean_val
        if np.var(col) < 1e-6:
            arr_filled[:, i] += np.random.rand(arr_filled.shape[0]) * 0.001
    return arr_filled


def normalize_weights(weights):
    """权重归一化"""
    weights = weights.copy().astype(np.float64)
    total = np.sum(weights)

    if total < 1e-8:
        normalized = np.ones_like(weights) / len(weights)
        print(f"⚠️  The total weight is 0, automatically divided equally: the first 3 weights={normalized[:3]}...")
        return normalized

    normalized = weights / total
    normalized = normalized / np.sum(normalized)  # 强制修正浮点误差

    print(
        f"🔍 Weight normalization: original sum={total:.8f} → Normalized Sum={np.sum(normalized):.8f}（The first three weights={normalized[:3]}）"
    )
    return normalized


# ---------------------- 3. 权重计算函数（核心修改：污染程度权重统一为被聚合指标权重和） ----------------------
def calculate_entropy_weight(safe_arr, valid_cols, agg_col_idx, total_pollution_weight):
    """熵值法（强制污染程度权重=被聚合指标权重和）"""
    arr = fill_nan_with_mean(safe_arr)
    arr_max = np.max(arr, axis=0)
    arr_min = np.min(arr, axis=0)
    arr_pos = (arr_max - arr) / (arr_max - arr_min + 1e-8)
    arr_pos = (arr_pos - np.min(arr_pos, axis=0)) / (
        np.max(arr_pos, axis=0) - np.min(arr_pos, axis=0) + 1e-8
    )
    arr_pos += 1e-10

    p = arr_pos / (np.sum(arr_pos, axis=0) + 1e-8)
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(arr_pos.shape[0] + 1e-8)
    entropy_weight = (1 - entropy) / (np.sum(1 - entropy) + 1e-8)

    # 核心修改：强制污染程度的熵值权重=被聚合指标权重和
    if agg_col_idx != -1:
        entropy_weight[agg_col_idx] = total_pollution_weight
        print(
            f"📌 Mandatory Entropy Method Weight: Pollution status={total_pollution_weight:.6f}（Weight of aggregated indicators）"
        )

    entropy_weight = normalize_weights(entropy_weight)
    return entropy_weight


def calculate_critic_weight(safe_arr, valid_cols, agg_col_idx, total_pollution_weight):
    """Critic法（强制污染程度权重=被聚合指标权重和）"""
    arr = fill_nan_with_mean(safe_arr)
    arr_norm = (arr - np.min(arr, axis=0)) / (
        np.max(arr, axis=0) - np.min(arr, axis=0) + 1e-8
    )
    std_vals = np.std(arr_norm, axis=0) + 1e-10
    corr_mat = np.corrcoef(arr_norm.T)
    corr_mat = np.nan_to_num(corr_mat, 0)
    conflict_mat = 1 - np.abs(corr_mat)
    critic_val = std_vals * np.sum(conflict_mat, axis=1)

    # 核心修改：强制污染程度的Critic权重=被聚合指标权重和
    if agg_col_idx != -1:
        critic_val[agg_col_idx] = total_pollution_weight
        print(
            f"📌 Mandatory assignment of Critic weight: pollution status={total_pollution_weight:.6f}（Weight of aggregated indicators）"
        )

    critic_weight = normalize_weights(critic_val)
    return critic_weight


def calculate_pca_weight(
    safe_arr, valid_cols, agg_col_idx, total_pollution_weight, var_threshold=0.85
):
    """PCA法（强制污染程度权重=被聚合指标权重和）"""
    arr = fill_nan_with_mean(safe_arr)
    arr_norm = (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + 1e-8)

    if arr_norm.shape[1] < 2:
        pca_weights = np.ones(arr_norm.shape[1]) / arr_norm.shape[1]
    else:
        pca = PCA(n_components=None)
        pca.fit(arr_norm)
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        n_components = (
            np.argmax(cumulative_var >= var_threshold) + 1
            if np.max(cumulative_var) >= var_threshold
            else arr_norm.shape[1]
        )
        pca_weights = np.sum(
            np.abs(pca.components_[:n_components])
            * explained_var[:n_components, np.newaxis],
            axis=0,
        )

    # 核心修改：强制污染程度的PCA权重=被聚合指标权重和
    if agg_col_idx != -1:
        pca_weights[agg_col_idx] = total_pollution_weight
        print(
            f"📌 Mandatory assignment of PCA weights: pollution status={total_pollution_weight:.6f}（Weight of aggregated indicators）"
        )

    pca_weights = normalize_weights(pca_weights)
    return pca_weights


def game_theory_combination(weight_list, agg_col_idx, total_pollution_weight):
    """博弈论组合（强制综合客观权重=被聚合指标权重和）"""
    weight_mat = np.array(weight_list).T.astype(np.float64)
    weight_mat = fill_nan_with_mean(weight_mat)

    A = weight_mat
    ATA = np.dot(A.T, A) + 1e-8 * np.eye(A.shape[1])
    inv_ATA = np.linalg.inv(ATA)
    ones_vec = np.ones(A.shape[1])
    coeffs = np.dot(inv_ATA, ones_vec) / (
        np.dot(np.dot(ones_vec.T, inv_ATA), ones_vec) + 1e-8
    )
    combined_weight = np.dot(A, coeffs)

    # 核心修改：强制污染程度的综合客观权重=被聚合指标权重和
    if agg_col_idx != -1:
        combined_weight[agg_col_idx] = total_pollution_weight
        print(
            f"📌 Mandatory assignment of comprehensive objective weights: pollution status={total_pollution_weight:.6f}（Weight of aggregated indicators）"
        )

    combined_weight = normalize_weights(combined_weight)
    return combined_weight


# ---------------------- 4. 数据读取函数 ----------------------
def read_data(file_path):
    """读取数据"""
    try:
        if file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            raise ValueError("Only supports Excel(.xlsx) and CSV(.csv)formats")
    except Exception as e:
        raise FileNotFoundError(f"fail to read file：{str(e)}")

    df_clean = clean_and_convert(df)
    df_clean, is_aggregated, aggregated_pollution_cols = aggregate_pollution_indicators(
        df_clean
    )

    # 强制校验污染程度列
    if is_aggregated and AGG_POLLUTION_COL not in df_clean.columns:
        df_clean[AGG_POLLUTION_COL] = 0.0
        print(f"⚠️  Not generated in aggregation scenario{AGG_POLLUTION_COL}row，Mandatory supplementation (value 0)")

    print(f"\n📊 Data reading&processing completed：")
    print(f"  sample size：{len(df_clean)} | Total number of columns：{len(df_clean.columns)}")
    print(f"  Point Ranking：{'existence' if 'location' in df_clean.columns else 'does not exist'}")
    print(
        f"  Pollution status columns：{'existence' if AGG_POLLUTION_COL in df_clean.columns else 'does not exist'}（aggregation state：{is_aggregated}）"
    )
    print(
        f"  all columns：{', '.join(df_clean.columns[:10])}{'...' if len(df_clean.columns) > 10 else ''}"
    )

    return df_clean, is_aggregated, aggregated_pollution_cols


# ---------------------- 5. 权重整合函数（核心修改：统一污染程度权重） ----------------------
def get_weights(df, selected_cols, is_aggregated, aggregated_pollution_cols):
    """权重整合（强制污染程度所有权重=被聚合指标权重和）"""
    # 第一步：映射列
    mapped_selected_cols = map_pollution_cols(selected_cols, df.columns, is_aggregated)

    # 第二步：强制补充污染程度列
    if (
        is_aggregated
        and AGG_POLLUTION_COL not in mapped_selected_cols
        and AGG_POLLUTION_COL in df.columns
    ):
        mapped_selected_cols.append(AGG_POLLUTION_COL)
        print(f"📌 Mandatory supplementary column in weight calculation stage：{AGG_POLLUTION_COL}（Aggregation scenarios must include）")

    # 第三步：筛选有效列
    valid_cols = [col for col in mapped_selected_cols if col in df.columns]
    if not valid_cols:
        raise ValueError(
            f"Mapped indicators{', '.join(mapped_selected_cols)}None of them exist in the data"
        )

    # 第四步：定位污染程度列索引 & 计算被聚合指标权重和
    agg_col_in_valid = AGG_POLLUTION_COL in valid_cols
    agg_col_idx = valid_cols.index(AGG_POLLUTION_COL) if agg_col_in_valid else -1
    total_pollution_weight = sum(
        [POLLUTION_AHP_WEIGHTS.get(col, 0.0) for col in aggregated_pollution_cols]
    )  # 核心值

    # 校验聚合场景
    if is_aggregated and not agg_col_in_valid:
        raise ValueError(f"Under the aggregation scenario，{AGG_POLLUTION_COL}Column not included in weight calculation！")

    # 第五步：准备数据
    df_selected = df[valid_cols].copy()
    safe_arr = df_to_safe_array(df_selected)

    # 第六步：计算客观权重（全部强制赋值为被聚合指标权重和）
    print("\n📈 Start calculating objective weights（Weight of mandatory pollution status=Weight of aggregated indicators）：")
    entropy_w = calculate_entropy_weight(
        safe_arr, valid_cols, agg_col_idx, total_pollution_weight
    )
    critic_w = calculate_critic_weight(
        safe_arr, valid_cols, agg_col_idx, total_pollution_weight
    )
    pca_w = calculate_pca_weight(
        safe_arr, valid_cols, agg_col_idx, total_pollution_weight
    )

    # 第七步：计算综合客观权重（强制赋值）
    print("\n📈 Calculate the objective weights of game theory combinations（Weight of mandatory pollution status=Weight of aggregated indicators）：")
    obj_w = game_theory_combination(
        [entropy_w, critic_w, pca_w], agg_col_idx, total_pollution_weight
    )

    # 第八步：计算AHP主观权重（强制污染程度=被聚合指标权重和）
    print("\n📈 Calculate the subjective weights of AHP（Weight of mandatory pollution status=Weight of aggregated indicators）：")
    ahp_w = []
    for i, col in enumerate(valid_cols):
        if col == AGG_POLLUTION_COL:
            ahp_w.append(total_pollution_weight)
            print(
                f"   {col}：Force assignment={total_pollution_weight:.6f}（Weight of aggregated indicators）"
            )
        else:
            w = AHP_WEIGHTS_DICT.get(col, 0.0)
            ahp_w.append(w)
            print(f"   {col}：Original weight={w:.6f}")
    ahp_w = np.array(ahp_w)
    ahp_w = normalize_weights(ahp_w)

    # 第九步：计算最终综合权重
    print("\n🏁 Calculate the final comprehensive weight (objective 50%+subjective 50%)：")
    final_w = 0.5 * obj_w + 0.5 * ahp_w
    final_w = normalize_weights(final_w)

    # 第十步：专项校验（核心日志）
    print("\n🔴 Pollution status Weight Special Verification (Weight Integration Stage)：")
    if agg_col_in_valid:
        print(f"   Weight of aggregated indicators：{total_pollution_weight:.6f}")
        print(f"   Related indicators：{', '.join(aggregated_pollution_cols)}")
        print(f"   Entropy method weight：{entropy_w[agg_col_idx]:.8f}")
        print(f"   Critic method weight：{critic_w[agg_col_idx]:.8f}")
        print(f"   PCA method weight：{pca_w[agg_col_idx]:.8f}")
        print(f"   Comprehensive objective weight：{obj_w[agg_col_idx]:.8f}")
        print(f"   AHP subjective weights：{ahp_w[agg_col_idx]:.8f}")
        print(
            f"   Final comprehensive weight：{final_w[agg_col_idx]:.8f}（account for={final_w[agg_col_idx] * 100:.4f}%）"
        )
    else:
        print(f"   Not found{AGG_POLLUTION_COL}column（aggregation state：{is_aggregated}）")

    # 第十一步：全量日志
    print("\n=== Total weight summary ===")
    for i, col in enumerate(valid_cols):
        print(f"   {col}：")
        print(
            f"     entropy method：{entropy_w[i]:.8f} | Critic：{critic_w[i]:.8f} | PCA：{pca_w[i]:.8f}"
        )
        print(
            f"     Comprehensive objective：{obj_w[i]:.8f} | AHP：{ahp_w[i]:.8f} | final：{final_w[i]:.8f}"
        )
        if col == AGG_POLLUTION_COL:
            print(f"     Weight of aggregated indicators：{total_pollution_weight:.6f}")

    # 权重校验
    total_final_weight = np.sum(final_w)
    pollution_final_weight = final_w[agg_col_idx] if agg_col_in_valid else 0.0
    print(f"\n✅ Weight verification result：")
    print(f"   Final total weight：{total_final_weight:.8f}（应=1.0）")
    print(
        f"   Final weight of pollution status：{pollution_final_weight:.8f}（{pollution_final_weight * 100:.4f}%）"
    )
    print(f"   Total weight of other indicators：{total_final_weight - pollution_final_weight:.8f}")

    return {
        "entropy": entropy_w,
        "critic": critic_w,
        "pca": pca_w,
        "objective": obj_w,
        "ahp": ahp_w,
        "final": final_w,
        "selected_cols": valid_cols,
        "pollution_weight_ratio": pollution_final_weight,
        "aggregated_pollution_cols": aggregated_pollution_cols,
        "agg_col_idx": agg_col_idx,
        "total_pollution_weight": total_pollution_weight,  # 替换：agg_weight_sum → total_pollution_weight
    }


# ---------------------- 6. 业务赋分函数 ----------------------
def score_features(df, land_type, selected_cols, is_aggregated):
    """业务赋分"""
    # 第一步：映射列
    mapped_selected_cols = map_pollution_cols(selected_cols, df.columns, is_aggregated)

    # 第二步：强制补充污染程度列
    if (
        is_aggregated
        and AGG_POLLUTION_COL not in mapped_selected_cols
        and AGG_POLLUTION_COL in df.columns
    ):
        mapped_selected_cols.append(AGG_POLLUTION_COL)

    # 第三步：筛选有效列
    valid_cols = [col for col in mapped_selected_cols if col in df.columns]
    print(
        f"\n📝 Effective column in the scoring process：{len(valid_cols)}row | Including pollution status：{AGG_POLLUTION_COL in valid_cols}"
    )

    # 第四步：初始化赋分表
    df_score = (
        df[["location"] + valid_cols].copy(deep=True)
        if "location" in df.columns
        else df[valid_cols].copy()
    )
    cache_cols = [col for col in df.columns if col.startswith(POLLUTION_SCORE_PREFIX)]
    if cache_cols:
        df_score = pd.concat([df_score, df[cache_cols].copy()], axis=1)

    # 第五步：校验用地类型
    land_type = land_type if land_type in ["paddy", "others"] else "others"

    # 第六步：聚合污染程度赋分函数
    def get_aggregated_pollution_score(row):
        score_vals = []
        for cache_col in cache_cols:
            val = row[cache_col]
            if pd.notna(val) and val >= 0:
                score_vals.append(val)
        if not score_vals:
            return 60.0
        min_score = min(score_vals)
        other_scores = [s for s in score_vals if s != min_score]
        mean_other = min_score if not other_scores else np.mean(other_scores)
        final_score = min_score * 0.3 + mean_other * 0.7
        return max(0.0, min(100.0, final_score))

    # 第七步：逐行赋分
    ph_col = "pH" if "pH" in df_score.columns else None
    for idx, row in df_score.iterrows():
        ph_val = row[ph_col] if ph_col is not None else 7.0
        for col in valid_cols:
            val = row[col]
            # 污染程度列赋分
            if col == AGG_POLLUTION_COL:
                score = get_aggregated_pollution_score(row)
            # 原污染指标赋分
            elif col in POLLUTION_INDICATORS:
                score = calculate_pollutant_score(val, ph_val, col)
            # 基础指标赋分
            else:
                if pd.isna(val):
                    score = 60.0
                elif col == "effective soil layer thickness":
                    score = (
                        100
                        if val >= 150
                        else 90
                        if val >= 100
                        else 70
                        if val >= 60
                        else 50
                        if val >= 30
                        else 10
                    )
                elif col == "terrain gradient":
                    score = (
                        100
                        if val < 2
                        else 90
                        if val <= 5
                        else 80
                        if val <= 8
                        else 60
                        if val <= 15
                        else 30
                        if val <= 25
                        else 10
                    )
                elif col == "groundwater depth":
                    score = 100 if val >= 3 else 70 if val >= 2 else 30
                elif col == "soil organic carbon content":
                    score = (
                        100
                        if val >= 23.2
                        else 90
                        if val >= 17.4
                        else 80
                        if val >= 11.6
                        else 60
                        if val >= 5.8
                        else 40
                        if val >= 3.5
                        else 20
                    )
                elif col == "Light-temperature production potential":
                    score = (
                        100
                        if val >= 4000
                        else 80
                        if val >= 3000
                        else 60
                        if val >= 2000
                        else 40
                        if val >= 1000
                        else 20
                    )
                elif col == "pH":
                    score = (
                        100
                        if 6.0 <= val <= 7.9
                        else 90
                        if (5.5 < val < 6.0 or 7.9 < val < 8.5)
                        else 80
                        if (5.0 <= val <= 5.5 or 8.5 <= val <= 9.0)
                        else 60
                        if 4.0 < val < 5.0
                        else 30
                        if val < 4.0 or 9.0 < val < 9.5
                        else 10
                    )
                elif col == "salinization degree":
                    score = (
                        100 if val == 0 else 60 if val == 1 else 30 if val == 2 else 50
                    )
                elif col == "biodiversity":
                    score = (
                        100 if val == 0 else 80 if val == 1 else 50 if val == 2 else 60
                    )
                elif col == "the capacity of irrigation and drainage":
                    score = (
                        100 if val == 0 else 50 if val == 1 else 20 if val == 2 else 60
                    )
                elif col == "surface soil texture":
                    score = (
                        100
                        if val == 0
                        else 90
                        if val == 1
                        else 70
                        if val == 2
                        else 40
                        if val == 3
                        else 70
                    )
                elif col == "carbon pool variation factor(arable land)":
                    score = (
                        80
                        if val == 0
                        else 69
                        if val == 1
                        else 64
                        if val == 2
                        else 58
                        if val == 3
                        else 48
                        if val == 4
                        else 65
                    )
                elif col == "profile configuration":
                    score = (
                        100
                        if val == 0
                        else 90
                        if val == 1
                        else 70
                        if val == 2
                        else 60
                        if val == 3
                        else 50
                        if val == 4
                        else 40
                        if val == 5
                        else 30
                        if val == 6
                        else 60
                    )
                elif col == "soil bulk density":
                    score = 100 if val == 0 else 50 if val == 1 or val == 2 else 70
                elif col == "total nitrogen":
                    score = (
                        100
                        if (land_type == "paddy" and val >= 1.2)
                        or (land_type == "others" and val >= 1.0)
                        else 50
                        if (land_type == "paddy" and val >= 0.6)
                        or (land_type == "others" and val >= 0.5)
                        else 10
                    )
                elif col == "available phosphorus":
                    score = (
                        100
                        if (land_type == "paddy" and val >= 12.5)
                        or (land_type == "others" and val >= 7.5)
                        else 50
                        if (land_type == "paddy" and val >= 6.25)
                        or (land_type == "others" and val >= 3.75)
                        else 10
                    )
                elif col == "available potassium":
                    score = (
                        100
                        if (land_type == "paddy" and val >= 100)
                        or (land_type == "others" and val >= 80)
                        else 50
                        if (land_type == "paddy" and val >= 50)
                        or (land_type == "others" and val >= 40)
                        else 10
                    )
                elif col == "cation exchange capacity":
                    score = (
                        100
                        if (land_type == "paddy" and val >= 15)
                        or (land_type == "others" and val >= 12)
                        else 50
                        if (land_type == "paddy" and val >= 7.5)
                        or (land_type == "others" and val >= 6)
                        else 10
                    )
                else:
                    score = 60.0
            # 赋值
            df_score.loc[idx, col] = max(0.0, min(100.0, float(score)))

    # 第八步：清理缓存列
    df_score = df_score.drop(columns=cache_cols, errors="ignore")

    # 第九步：校验污染程度赋分
    if AGG_POLLUTION_COL in df_score.columns:
        agg_scores = df_score[AGG_POLLUTION_COL].values
        print(f"\n✅ Pollution status scoring completed：")
        print(f"   Scoring range：{np.min(agg_scores):.2f} ~ {np.max(agg_scores):.2f}")
        print(f"   Average rating：{np.mean(agg_scores):.2f}")
        print(f"   Assign points to the first three lines：{agg_scores[:3]}")
    else:
        print(f"\n⚠️  Pollution status column not assigned score: column does not exist")

    return df_score


# ---------------------- 7. 最终得分计算 ----------------------
def calculate_final_score(
    df_score, weights, selected_cols, is_aggregated, actual_pollution_cols=None
):
    """最终得分计算（修复版：统一归一化逻辑）"""
    # 第一步：获取基础信息
    original_valid_cols = weights["selected_cols"]
    agg_col_name = AGG_POLLUTION_COL

    # 【核心逻辑1】计算“污染程度”的原始权重值（即：已有污染物的权重之和）
    # 优先使用传入的实际列表，如果没有则尝试用weights里的缓存值
    if actual_pollution_cols and len(actual_pollution_cols) > 0:
        total_pollution_weight = sum(
            [POLLUTION_AHP_WEIGHTS.get(col, 0.0) for col in actual_pollution_cols]
        )
        print(f"\n🔄 [Weight calculation] Detected actual pollutants：{', '.join(actual_pollution_cols)}")
        print(f"   Original weight of pollution status（Sum）= {total_pollution_weight:.6f}")
    else:
        total_pollution_weight = weights.get("total_pollution_weight", 0.0)
        print(
            f"\n⚠️ No actual pollutant list detected, using default pollution weights：{total_pollution_weight:.6f}"
        )

    # 第二步：聚合场景下强制补充“污染程度”列
    if is_aggregated and agg_col_name not in df_score.columns:
        print(f"\n⚠️  Under the aggregation scenario{agg_col_name}missing columns，Mandatory supplementation (with a default value of 60 points)")
        df_score[agg_col_name] = 60.0

    # 第三步：准备权重映射表
    # 注意：col_weight_map 存储的是各个指标原本算出来的“原始权重份额”
    col_weight_map = dict(zip(weights["selected_cols"], weights["final"]))

    # 第四步：组装所有参与计算的列和它们的“原始权重”
    valid_cols = []
    raw_weights_list = []  # 暂存未归一化的权重

    # 遍历数据中实际存在的每一列
    for col in df_score.columns:
        if col == "location" or col.startswith(POLLUTION_SCORE_PREFIX):
            continue

        # 情况A：是污染程度列
        if col == agg_col_name:
            valid_cols.append(col)
            raw_weights_list.append(total_pollution_weight)  # 直接用算出来的Sum
            print(
                f"   -> Include in column：{col} | Original weight={total_pollution_weight:.6f} (pollutions Sum)"
            )

        # 情况B：是环境指标列（且在权重表中存在）
        elif col in col_weight_map:
            valid_cols.append(col)
            w = col_weight_map[col]
            raw_weights_list.append(w)
            # 只有前几个打日志，避免刷屏
            if len(valid_cols) < 5:
                print(f"   -> Include in column：{col} | Original weight={w:.6f}")

    # 防御性逻辑：如果聚合了但没找到污染程度列，强制加进去
    if is_aggregated and agg_col_name not in valid_cols:
        print(f"⚠️  {agg_col_name}Not in the column, forced to add calculation")
        valid_cols.append(agg_col_name)
        raw_weights_list.append(total_pollution_weight)

    if not valid_cols:
        raise ValueError("No valid columns are used to calculate the Final score！")

    # 第五步：【核心修改】统一归一化
    # 现在的逻辑是：[污染Sum, 环境1, 环境2...] 一起求和，然后大家一起除以这个和
    matched_weights = np.array(raw_weights_list)
    total_sum = np.sum(matched_weights)

    if total_sum == 0:
        print("❌ Error: The total weight is 0, unable to normalize")
        final_weights = np.ones_like(matched_weights) / len(matched_weights)
    else:
        final_weights = matched_weights / total_sum

    # 找到污染程度最终的权重，用于日志显示
    agg_weight_idx = (
        valid_cols.index(agg_col_name) if agg_col_name in valid_cols else -1
    )
    final_agg_weight = final_weights[agg_weight_idx] if agg_weight_idx != -1 else 0.0

    # 第六步：核心日志
    print("\n" + "=" * 80)
    print("🔴 Weight normalization result verification")
    print("=" * 80)
    print(f"1. Total number of indicators involved in the calculation：{len(valid_cols)}")
    print(f"2. Sum of original weights（denominator）：{total_sum:.6f}")
    if agg_weight_idx != -1:
        print(
            f"3. Pollution status: original weight={raw_weights_list[agg_weight_idx]:.6f} -> Normalized weights={final_agg_weight:.6f}"
        )
    print(f"4. Weight Check: Normalized Sum = {np.sum(final_weights):.6f}")
    print("=" * 80)

    # 第七步：计算加权得分
    df_weighted = df_score.copy()
    weighted_cols = []
    for i, col in enumerate(valid_cols):
        weight = final_weights[i]
        df_weighted[f"{col}_weight"] = weight
        df_weighted[f"{col}_Weighted score"] = df_weighted[col] * weight
        weighted_cols.append(f"{col}_Weighted score")

    # 第八步：计算最终得分
    df_weighted["Final Score"] = df_weighted[weighted_cols].sum(axis=1).round(4)

    # 第九步：输出统计
    print("\n📊 Final score statistics：")
    print(f"   Number of evaluation points：{len(df_weighted)}")
    print(f"   Average Final score：{df_weighted['Final Score'].mean():.4f}")
    print(
        f"   Highest score：{df_weighted['Final Score'].max():.4f} | Minimum score：{df_weighted['Final Score'].min():.4f}"
    )

    return df_weighted


# ---------------------- 8. 结果输出 ----------------------
def export_results(
    df_score, df_weighted, weights, selected_cols, land_type, is_aggregated
):
    """导出结果"""
    valid_cols = weights["selected_cols"]
    aggregated_pollution_cols = weights.get("aggregated_pollution_cols", [])
    total_pollution_weight = weights.get(
        "total_pollution_weight", 0.0
    )  # 替换：agg_weight_sum → total_pollution_weight

    # 权重汇总表
    weight_df = pd.DataFrame(
        {
            "indicator name": valid_cols,
            "Entropy method weight（normalization）": weights["entropy"],
            "Critic method weight（normalization）": weights["critic"],
            "PCA method weight（normalization）": weights["pca"],
            "Comprehensive objective weight（normalization）": weights["objective"],
            "AHP subjective weight（normalization）": weights["ahp"],
            "Final comprehensive weight（normalization）": weights["final"],
            "Final weight proportion(%)": [round(w * 100, 4) for w in weights["final"]],
            "Whether it is a pollution indicator": [
                1 if col in POLLUTION_INDICATORS + [AGG_POLLUTION_COL] else 0
                for col in valid_cols
            ],
            "Aggregated original pollution indicators": [
                ",".join(aggregated_pollution_cols) if col == AGG_POLLUTION_COL else ""
                for col in valid_cols
            ],
            "Weight of aggregated indicators": [
                total_pollution_weight if col == AGG_POLLUTION_COL else 0
                for col in valid_cols
            ],
        }
    )

    # 点位赋分表
    score_export = df_score.rename(columns={col: f"{col}_scoring" for col in valid_cols})

    # 加权得分表
    weighted_export = df_weighted.copy()
    weighted_export = weighted_export.rename(
        columns={
            col: f"{col}_Weighted score" if col.endswith("_Weighted score") else col
            for col in weighted_export.columns
        }
    )

    # 整体统计
    stats_df = pd.DataFrame(
        {
            "statistical indicators": [
                "Number of evaluation points",
                "Average score",
                "Highest score",
                "The lowest score",
                "Standard deviation",
                "Coefficient of variation",
                "Land use type",
                "Pollution weight proportion(%)",
                "Basic weight proportion(%)",
                "The final sum of weights",
                "Whether to aggregate pollution indicators",
                "Aggregation scoring rules",
                "Aggregated original pollution indicators",
                "Weight of aggregated indicators",
            ],
            "numerical value": [
                len(df_weighted),
                round(df_weighted["Final Score"].mean(), 4),
                round(df_weighted["Final Score"].max(), 4),
                round(df_weighted["Final Score"].min(), 4),
                round(df_weighted["Final Score"].std(), 4),
                round(df_weighted["Final Score"].std() / df_weighted["Final Score"].mean(), 6)
                if df_weighted["Final Score"].mean() != 0
                else 0,
                land_type,
                round(weights["pollution_weight_ratio"] * 100, 4),
                round((1 - weights["pollution_weight_ratio"]) * 100, 4),
                round(np.sum(weights["final"]), 8),
                "yes" if is_aggregated else "no",
                "Minimum score x 30%+average score of other scores x 70%"
                if is_aggregated
                else "Original pollutant threshold scoring",
                ",".join(aggregated_pollution_cols) if is_aggregated else "",
                total_pollution_weight if is_aggregated else 0,
            ],
        }
    )

    # 写入Excel
    try:
        filename = "Assessment Results of Polluted Sites.xlsx"
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            weight_df.to_excel(writer, sheet_name="Weight Summary", index=False)
            score_export.to_excel(writer, sheet_name="location scoring", index=False)
            weighted_export.to_excel(writer, sheet_name="Weighted score", index=False)
            stats_df.to_excel(writer, sheet_name="Overall statistics", index=False)
        print(f"\n✅ The results have been exported to：{filename}")
    except PermissionError:
        print("⚠️  Unable to write to Excel file, it may have been opened！")
        return

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(
        f"Assessment results of contaminated sites（{land_type} | aggregation：{is_aggregated}）",
        fontsize=18,
        fontweight="bold",
    )

    # 1. 最终得分柱状图
    scores = df_weighted["Final Score"].values[:15]
    sites = (
        df_weighted["location"].values[:15]
        if "location" in df_weighted.columns
        else [f"location{i + 1}" for i in range(len(scores))]
    )
    axes[0, 0].bar(sites, scores, color="#1f77b4", alpha=0.8)
    axes[0, 0].set_title("Final score for the first 15 locations", fontsize=14)
    axes[0, 0].set_xticklabels(sites, rotation=45, ha="right")
    axes[0, 0].set_ylabel("score")

    # 2. 指标权重分布
    weight_vals = weights["final"][:15]
    weight_cols = valid_cols[:15]
    colors = [
        "#ff4444" if col == AGG_POLLUTION_COL else "#ff7f0e" for col in weight_cols
    ]
    axes[0, 1].barh(weight_cols, weight_vals, color=colors)
    axes[0, 1].set_title("The final weight of the first 15 indicators (red=pollution status)）", fontsize=14)
    axes[0, 1].set_xlabel("weight value")

    # 3. 得分分布
    axes[1, 0].hist(df_weighted["Final Score"], bins=10, color="#2ca02c", alpha=0.8)
    axes[1, 0].set_title("Final score distribution", fontsize=14)
    axes[1, 0].set_xlabel("score")
    axes[1, 0].set_ylabel("frequency")

    # 4. 污染程度权重占比
    if weights["pollution_weight_ratio"] > 0:
        pie_sizes = [
            weights["pollution_weight_ratio"],
            1 - weights["pollution_weight_ratio"],
        ]
        pie_labels = ["pollution status", "Other indicators"]
        axes[1, 1].pie(
            pie_sizes,
            labels=pie_labels,
            autopct="%1.2f%%",
            colors=["#ff4444", "#1f77b4"],
        )
        axes[1, 1].set_title("Proportion of pollution status weight", fontsize=14)
    else:
        axes[1, 1].text(
            0.5, 0.5, "The weight of pollution status is 0", ha="center", va="center", fontsize=16
        )
        axes[1, 1].set_title("Proportion of pollution status weight", fontsize=14)

    plt.tight_layout()
    plt.savefig("Visualization of Polluted Site Assessment.png", dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------- 9. 主函数 ----------------------
def main():
    """主函数"""
    try:
        print("=" * 80)
        print("📌 Contaminated Site Assessment Procedure ( Pollution status Weight=Weight of aggregated indicators）")
        print("=" * 80)

        # 第一步：配置文件路径
        weight_data_path = "Data weight calculation.xlsx"
        site_data_path = input("Please enter the path of the multi-point data file（Excel/CSV）：").strip()
        if not site_data_path:
            print("❌ The path cannot be empty！")
            return

        # 第二步：读取权重数据
        print(f"\n📥 Read weight calculation data：{weight_data_path}")
        weight_df, weight_is_aggregated, weight_agg_cols = read_data(weight_data_path)
        all_cols = [col for col in weight_df.columns if col != "location"]

        # 第三步：强制选择全部指标
        print(f"\n📋 Force selection of all indicators（{len(all_cols)}in total），Ensure pollution status column is included")
        selected_cols = all_cols

        # 第四步：选择用地类型
        land_type = input("\nPlease select the land type（paddy/others，Default others）：").strip() or "others"

        # 第五步：计算权重（核心：统一污染程度权重）
        print("\n🔢 Start to calculate the weight (pollution status weight=Weight of aggregated indicators)...")
        weights = get_weights(
            weight_df, selected_cols, weight_is_aggregated, weight_agg_cols
        )

        # 第六步：读取多点位数据
        print(f"\n📥 Read multi-point evaluation data：{site_data_path}")
        site_df, site_is_aggregated, site_agg_cols = read_data(site_data_path)

        # 第七步：赋分
        print("\n📝 Start indicator scoring...")
        score_df = score_features(site_df, land_type, selected_cols, site_is_aggregated)

        # 第八步：计算最终得分
        print("\n🏆 Calculate Final score...")
        weighted_df = calculate_final_score(
            score_df, weights, selected_cols, site_is_aggregated, site_agg_cols
        )

        # 第九步：导出结果前，精确计算并更新统计信息
        print("\n🔄 Update statistics and prepare for export...")

        # 1. 计算【环境指标】的原始权重和
        # 先找到当前数据中实际存在的环境指标列
        current_env_cols = [
            col
            for col in weighted_df.columns
            if col.endswith("_weight") and not col.startswith(AGG_POLLUTION_COL)
        ]
        # 注意：weighted_df里的列名是"xxx_权重"，我们需要数值。
        # 更直接的方法是利用我们在 calculate_final_score 里用的逻辑：
        # 重新获取环境指标的 map
        col_weight_map = dict(zip(weights["selected_cols"], weights["final"]))

        # 找出当前数据里有哪些环境指标，把它们的原始权重加起来
        current_env_weight_sum = 0.0
        for col in score_df.columns:
            if col in col_weight_map:
                current_env_weight_sum += col_weight_map[col]

        # 2. 计算【污染程度】的原始权重和 (被聚合指标权重和)
        if site_agg_cols:
            real_pollution_weight_sum = sum(
                [POLLUTION_AHP_WEIGHTS.get(col, 0.0) for col in site_agg_cols]
            )
        else:
            real_pollution_weight_sum = weights.get("total_pollution_weight", 0.0)

        # 3. 计算精确的归一化比例
        total_raw_sum = current_env_weight_sum + real_pollution_weight_sum
        if total_raw_sum > 0:
            final_pollution_ratio = real_pollution_weight_sum / total_raw_sum
        else:
            final_pollution_ratio = 0.0

        # 4. 更新 weights 字典（用于导出 Excel 和画图）
        # 这一步确保 Excel 里的“被聚合指标权重和”是正确的
        weights["total_pollution_weight"] = real_pollution_weight_sum
        weights["aggregated_pollution_cols"] = site_agg_cols

        # 这一步确保饼图（污染权重占比）是精确的
        weights["pollution_weight_ratio"] = final_pollution_ratio

        print(f"   ✅ Statistics corrected：")
        print(f"      - Weight of aggregated indicators（original）：{real_pollution_weight_sum:.6f}")
        print(f"      - Weight of environmental indicators and（original）：{current_env_weight_sum:.6f}")
        print(f"      - Final weight proportion of pollution status：{final_pollution_ratio * 100:.4f}%")

        # 第十步：导出结果
        print("\n💾 Export evaluation results...")
        export_results(
            score_df, weighted_df, weights, selected_cols, land_type, site_is_aggregated
        )

        # 最终总结
        print("\n🎉 Program running completed! Core conclusions：")
        print(f"1. Aggregate condition：{'Aggregated' if site_is_aggregated else 'Not aggregated'}")
        print(
            f"2. Pollution status column：{'existence' if AGG_POLLUTION_COL in weighted_df.columns else 'Does not exist'}"
        )
        print(
            f"3. Weight of aggregated indicators：{weights.get('total_pollution_weight', 0.0):.6f}"
        )  # 替换：agg_weight_sum → total_pollution_weight
        print(
            f"4. Final weight of pollution status：{weights['pollution_weight_ratio']:.8f}（{weights['pollution_weight_ratio'] * 100:.4f}%）"
        )
        print(f"5. Number of evaluation points：{len(weighted_df)}")
        print(f"6. Average Final score：{weighted_df['Final Score'].mean():.4f}")

    except Exception as e:
        print(f"\n❌ Program running error：{str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
