"""
这里提供一个完整的管线，用于基于因子的股票预测和指数增强回测。包括以下部分：
1) 超参数与数据范围的配置。
2) 数据加载和特征计算（示例性占位实现）。
3) 使用排名或标准化方法对数据进行预处理。
4) 适用于时间序列输入的自定义 PyTorch 数据集类。
5) 包含市场门控、特征提取和最终预测的 PyTorch 模型组件。
6) 使用自定义 Rank IC 损失来进行因子建模的训练流程。
7) 因子综合以及简化的回测框架。

可根据项目需求进一步扩展或替换此处的占位功能（如数据加载），以在生产环境中使用。
"""
# ----------------------------------------------------------------------------
# --- 0. 设置与导入 ---
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split # 或时间序列分割
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc # 垃圾回收

# 占位符：特征计算库（例如 qlib 或自定义实现）
# from feature_calculator import calculate_alpha158, calculate_moneyflow93, calculate_fundamental64

# 占位符：回测库
# from backtester import Backtester, PortfolioOptimizer

print(f"PyTorch 版本: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ----------------------------------------------------------------------------
# --- 1. 配置 ---
# ----------------------------------------------------------------------------
CONFIG = {
    # --- 数据与时间 ---
    "start_date": "2015-01-01", # 数据起始日期（需要历史数据以计算特征）
    "train_start_date": "2016-12-30", # 按报告回测起始日期
    "train_end_date": "2023-12-31", # 示例训练结束日期
    "test_start_date": "2024-01-01",
    "test_end_date": "2025-01-27", # 按报告回测结束日期
    "rolling_window_years": 8, # 示例：每期使用8年数据进行训练
    "prediction_horizon": 10, # 预测 T+10 收益
    "market_index_codes": ["000300.SH", "000905.SH", "000852.SH", "000985.CSI"], # HS300, CSI500, CSI1000, 全市场
    "target_index_code": "000852.SH", # CSI 1000 用于增强
    "stock_universe_source": "000985.CSI", # 使用 CSI 全市场成分股作为池

    # --- 特征 ---
    "feature_sets": ["alpha158", "moneyflow93", "fundamental64"], # 选择使用的基础特征
    "market_feature_lookback": [5, 10, 20, 30, 60], # 市场特征的回溯天数
    "use_market_gating": True,
    "gating_type": "attention", # 'linear' 或 'attention'
    "gating_beta": 3.0, # 线性门控的 Softmax 温度（报告发现1-3最优）

    # --- 模型 ---
    "feature_extractor_type": "attention", # 'gru' 或 'attention'（硕士论文的三层注意力）
    "d_model": 64, # 模型维度
    "n_heads": 4, # 注意力头数
    "num_gru_layers": 2, # GRU 提取器的层数
    "num_attention_layers": 3, # 硕士论文的注意力提取器层数
    "dropout_rate": 0.1,
    "time_series_length": 10, # 股票特征的输入序列长度 (T)

    # --- 训练 ---
    "batch_size": 1024, # 根据 GPU 内存调整
    "learning_rate": 1e-3,
    "weight_decay": 1e-4, # AdamW 权重衰减
    "epochs": 50, # 最大训练轮数
    "early_stopping_patience": 10, # 如果验证损失没有改善则停止

    # --- 因子综合 ---
    "synthesis_weights": "equal", # 'equal' 或未来可能的 'ml_based'
    "base_ai_factors": ["pv1", "pv5", "pv20"], # 来自报告的参考模型

    # --- 回测 ---
    "rebalance_freq": "W", # 每周再平衡
    "commission_rate": 0.0003, # 交易成本（3bps）
    "vwap_execution": True, # 假设 VWAP 执行（按报告）
    "turnover_limit": 0.20, # 每次再平衡的最大单向换手率（20%）
    "optimizer_constraints": { # 按报告
        "min_in_universe_weight": 0.80, # 目标指数成分股的最小权重为80%
        "industry_deviation": 0.02, # 行业偏离最大 +/- 2%（相对于基准）
        "style_deviation": { # 风格因子暴露偏离最大 +/- 0.3（大小，非线性大小）
            "size": 0.3,
            "nls": 0.3, # 非线性大小占位符
        },
        "max_stock_weight": 0.015, # 示例单只股票最大权重（未明确提及但常见）
        "max_stock_deviation": 0.01 # 示例相对于基准权重的最大偏离
    }
}

# ----------------------------------------------------------------------------
# --- 2. 数据加载 ---
# ----------------------------------------------------------------------------
def load_stock_data(start_date, end_date, universe_code):
    """加载每日股票 OHLCV、指数成分股等数据。"""
    print(f"加载股票数据，从 {start_date} 到 {end_date}，股票池为 {universe_code}...")
    # TODO: 实现实际数据加载（如 Wind、Tushare、JoinQuant、本地数据库等）
    # 需要：OHLCV、市值、行业分类、指数成分股、复权因子
    # 返回一个多索引 DataFrame（日期，资产）
    dummy_dates = pd.date_range(start_date, end_date)
    dummy_assets = [f"stock_{i:03d}.SZ" for i in range(5)]
    dummy_index = pd.MultiIndex.from_product([dummy_dates, dummy_assets], names=['date', 'asset'])
    dummy_data = pd.DataFrame(index=dummy_index)
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'market_cap', 'industry']:
        dummy_data[col] = np.random.rand(len(dummy_index))
    dummy_data['adj_factor'] = 1.0
    dummy_data['is_constituent'] = True # 假设所有股票都是成分股（占位实现）
    print("股票数据加载完成（占位实现）。")
    return dummy_data

def load_market_features(index_codes, start_date, end_date, lookbacks):
    """基于指数 OHLCV 计算并加载市场特征。"""
    print("加载市场特征...")
    market_features = {}
    # TODO: 加载实际指数 OHLCV 数据
    for index_code in index_codes:
         # 计算特征如 MR, MRAVG(d), MRSTD(d), MAAVG(d), MASTD(d) 等，d 为回溯天数
         # 返回一个以日期为索引的 DataFrame
        dummy_dates = pd.date_range(start_date, end_date)
        num_features = 1 + len(lookbacks) * 4 # MR + 4种类型 * 回溯天数
        market_features[index_code] = pd.DataFrame(
            np.random.randn(len(dummy_dates), num_features),
            index=dummy_dates,
            columns=[f"mkt_feat_{i}" for i in range(num_features)]
        )
    # 合并所有指数的特征（例如，列拼接）
    all_market_features = pd.concat(market_features.values(), axis=1)
    print(f"市场特征加载完成（占位实现），形状: {all_market_features.shape}")
    return all_market_features

# ----------------------------------------------------------------------------
# --- 3. 特征工程 ---
# ----------------------------------------------------------------------------
def calculate_features(stock_data, feature_set_name):
    """计算指定的特征集（Alpha158, MF93, Fund64）。"""
    print(f"计算特征集: {feature_set_name}...")
    # TODO: 使用 stock_data 实现实际特征计算
    # 这通常较复杂，通常依赖于库如 qlib 或自定义实现。
    if feature_set_name == "alpha158":
        num_features = 158
    elif feature_set_name == "moneyflow93":
        num_features = 93
    elif feature_set_name == "fundamental64":
        num_features = 64
    else:
        raise ValueError("未知特征集")

    features = pd.DataFrame(
        np.random.randn(len(stock_data), num_features),
        index=stock_data.index,
        columns=[f"{feature_set_name}_{i}" for i in range(num_features)]
    )
    print(f"{feature_set_name} 特征计算完成（占位实现），形状: {features.shape}")
    return features

def calculate_forward_returns(stock_data, horizon):
    """计算前瞻调整收益。"""
    print(f"计算 T+{horizon} 前瞻收益...")
    # TODO: 使用调整后的收盘价实现实际前瞻收益计算
    forward_returns = stock_data['close'].groupby(level='asset').pct_change(periods=-horizon).shift(horizon)
    # 如果需要，处理调整价格的潜在问题
    print("前瞻收益计算完成（占位实现）。")
    return forward_returns.rename('target_return')

# ----------------------------------------------------------------------------
# --- 4. 数据预处理 ---
# ----------------------------------------------------------------------------
def preprocess_data(features, market_features, target_returns):
    """在此函数中处理 NaN、归一化、标准化等操作。"""
    print("数据预处理...")
    data = features.join(market_features, on='date').join(target_returns)

    # --- 处理 NaN ---
    # 选项 1: 填充（例如，用中位数、均值、零） - 注意避免前瞻偏差
    # data = data.groupby(level='date').transform(lambda x: x.fillna(x.median())) # 横截面中位数填充
    # data = data.fillna(0) # 简单零填充

    # 选项 2: 删除含有过多 NaN 的股票/日期（或直接删除 NaN）
    data = data.dropna() # 最简单的方法，可能会丢失数据

    # --- 特征缩放 / 归一化 ---
    # 报告提到：排名归一化、标准化、行业/市值中性化
    # 应用横截面排名归一化是因子的常见做法
    # 应用时间序列归一化（如报告调查中提到的 RevIN）可以帮助解决漂移问题

    feature_cols = [col for col in data.columns if col not in ['target_return'] and not col.startswith('mkt_feat_')]
    market_feature_cols = [col for col in data.columns if col.startswith('mkt_feat_')]
    target_col = 'target_return'

    # 示例: 横截面排名归一化（QuantileTransformer 模拟排名）
    def rank_normalize_group(group):
        # 使用高斯输出以更好地适应线性模型/损失
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        group[feature_cols] = qt.fit_transform(group[feature_cols])
        # 对某些损失函数（如 Rank IC 损失）也进行目标排名归一化
        # group[target_col] = qt.fit_transform(group[[target_col]])[:, 0]
        return group

    # data = data.groupby(level='date', group_keys=False).apply(rank_normalize_group)
    print("排名归一化已应用（占位操作）。")


    # 示例: 标准化（减去均值，除以标准差） - 横截面应用
    def standardize_group(group):
        scaler_feat = StandardScaler()
        scaler_mkt = StandardScaler()
        scaler_target = StandardScaler()
        group[feature_cols] = scaler_feat.fit_transform(group[feature_cols])
        group[market_feature_cols] = scaler_mkt.fit_transform(group[market_feature_cols])
        # 也对目标进行标准化以稳定训练
        group[target_col] = scaler_target.fit_transform(group[[target_col]])[:, 0]
        return group

    data = data.groupby(level='date', group_keys=False).apply(standardize_group)
    print("标准化已应用。")

    # --- 中性化（可选但通常有帮助） ---
    # TODO: 如果需要，实施行业/市值中性化
    # 需要对特征进行行业虚拟变量/市值因子回归并取残差

    print(f"数据预处理完成。数据形状: {data.shape}")
    return data.dropna() # 在潜在的转换问题后再次删除 NaN


# --- 时间序列的自定义数据集 ---
class StockTimeSeriesDataset(Dataset):
    def __init__(self, data, stock_feature_cols, market_feature_cols, target_col, seq_len):
        self.data = data
        self.stock_feature_cols = stock_feature_cols
        self.market_feature_cols = market_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.unique_dates = data.index.get_level_values('date').unique().sort_values()
        self.date_to_idx = {date: i for i, date in enumerate(self.unique_dates)}
        self.num_stocks_per_date = data.groupby(level='date').size()

        # 预计算序列以加快访问速度（可能使用大量内存）
        self.sequences = self._precompute_sequences()

    def _precompute_sequences(self):
        print("预计算序列...")
        sequences = []
        all_assets = self.data.index.get_level_values('asset').unique()
        # 按资产分组数据以高效地进行时间序列切片
        grouped_by_asset = self.data.reset_index().set_index('date').groupby('asset')

        for date_idx, current_date in enumerate(tqdm(self.unique_dates)):
            if date_idx < self.seq_len - 1:
                continue # 历史数据不足

            start_date_idx = date_idx - self.seq_len + 1
            start_date = self.unique_dates[start_date_idx]

            # 获取当前日期的数据
            current_data_slice = self.data.loc[current_date]
            if isinstance(current_data_slice, pd.Series): # 处理单一股票情况
                 current_data_slice = current_data_slice.to_frame().T
            current_assets = current_data_slice.index

            # 获取序列的市场特征
            mkt_seq = self.data.loc[start_date:current_date, self.market_feature_cols].iloc[0:self.seq_len].values # 假设市场特征在股票间重复

            # 获取当前日期存在的每个资产的股票特征和目标
            for asset in current_assets:
                try:
                    asset_data = grouped_by_asset.get_group(asset)
                    stock_seq = asset_data.loc[start_date:current_date, self.stock_feature_cols]
                    target = asset_data.loc[current_date, self.target_col]

                    if len(stock_seq) == self.seq_len:
                        sequences.append({
                            'stock_features': torch.tensor(stock_seq.values, dtype=torch.float32),
                            'market_features': torch.tensor(mkt_seq, dtype=torch.float32),
                            'target': torch.tensor(target, dtype=torch.float32)
                        })
                except KeyError:
                    # 资产可能不存在完整的序列
                    continue
        print(f"预计算了 {len(sequences)} 个序列。")
        return sequences


    def __len__(self):
         return len(self.sequences)

    def __getitem__(self, idx):
         return self.sequences[idx]['stock_features'], self.sequences[idx]['market_features'], self.sequences[idx]['target']


# ----------------------------------------------------------------------------
# --- 5. 模型定义（关键组件） ---
# ----------------------------------------------------------------------------

class MarketGatingUnit(nn.Module):
    """对股票特征应用市场上下文门控。"""
    def __init__(self, d_market_features, d_stock_features, gating_type='attention', beta=1.0, n_heads=4):
        super().__init__()
        self.gating_type = gating_type
        self.beta = beta
        self.d_stock_features = d_stock_features

        if gating_type == 'linear':
            # 从市场特征到股票特征权重的线性投影
            self.market_to_weight = nn.Linear(d_market_features, d_stock_features)
            self.softmax = nn.Softmax(dim=-1)
        elif gating_type == 'attention':
            # 使用市场特征（在时间上平均？）查询股票特征
            # Q: 市场特征, K: 市场特征, V: 股票特征？或者 Q: 股票, K: 市场, V: 市场？
            # 按报告：市场特征 -> 权重生成
            # Q: 市场, K: Dummy, V: Dummy -> 获取权重；或者市场 -> 线性 -> 权重
            # 简化的注意力：市场特征生成 Q, K；股票特征为 V。
            # 这里，我们解释报告的图 24：市场 -> 线性 -> Softmax -> 权重
            # 实现 *这种* 注意力的想法稍有不同：
            # 使用市场上下文调制股票特征提取器中的注意力，
            # 或通过注意力显式生成门控权重：
            self.market_q = nn.Linear(d_market_features, d_stock_features) # 市场特征形成查询
            self.stock_k = nn.Linear(d_stock_features, d_stock_features) # 股票特征形成键
            self.attention = nn.MultiheadAttention(d_stock_features, n_heads, batch_first=True, dropout=0.1)
            # 注意力的输出可以是权重，也可以以其他方式使用。
            # 更接近图 24 的 *线性* 路径以清晰地实现门控单元：
            print("警告: MarketGatingUnit 注意力类型实现较复杂，目前使用线性实现。")
            self.gating_type = 'linear' # 回退到线性以简化示例框架
            self.market_to_weight = nn.Linear(d_market_features, d_stock_features)
            self.softmax = nn.Softmax(dim=-1) # 报告使用 Softmax(H_market / beta)
        else:
            raise ValueError("未知门控类型")

    def forward(self, stock_features, market_features_seq):
        # stock_features: (N, T, D_stock) 或 (N*T, D_stock) 如果展平
        # market_features_seq: (N, T, D_market) - 时间序列上的市场特征
        # N: 批量大小（给定日期的股票数量），T: 序列长度

        # 使用时间步的 *最后* 市场特征作为当前市场上下文
        market_context = market_features_seq[:, -1, :] # 形状: (N, D_market)

        if self.gating_type == 'linear':
            # 投影市场上下文到权重: (N, D_market) -> (N, D_stock)
            market_weights = self.market_to_weight(market_context)

            # 应用温度 beta 和 softmax
            # 重塑权重以广播: (N, 1, D_stock) 以与 (N, T, D_stock) 相乘
            # 将权重应用于 *最后* 股票特征表示？或整个序列？
            # 图 24 表示权重 (T x D_feature) 应用于股票特征 (N x T x D_feature)
            # 假设权重应用于 *最终* 特征表示（提取器之后）
            # 如果应用于 *提取器之前*：
            # 需要权重形状 (N, 1, D_stock) 或 (N, T, D_stock)
            # 尝试 (N, 1, D_stock) 应用于每个时间步 T。
            market_weights = self.softmax(market_weights / self.beta) # 形状: (N, D_stock)
            gating_weights = market_weights.unsqueeze(1) # 形状: (N, 1, D_stock)

            # 应用门控: 元素乘法
            # 确保 stock_features 形状为 (N, T, D_stock)
            gated_stock_features = stock_features * gating_weights

        elif self.gating_type == 'attention':
            # TODO: 实现基于注意力的门控
            # 查询（来自市场），键（来自股票？），值（股票）
            # 这需要根据报告的意图进行仔细设计。
            print("警告: 注意力门控在框架中未完全实现。")
            gated_stock_features = stock_features # 暂时通过

        else: # 无门控
             gated_stock_features = stock_features

        return gated_stock_features


class FeatureExtractor(nn.Module):
    """使用 GRU 或多层注意力提取特征。"""
    def __init__(self, input_dim, d_model, extractor_type='attention', num_gru_layers=2, num_attention_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.extractor_type = extractor_type
        self.d_model = d_model

        if extractor_type == 'gru':
            # 简单的 BiGRU 基线
            self.gru = nn.GRU(input_dim, d_model // 2 if num_gru_layers > 0 else d_model,
                              num_layers=num_gru_layers, batch_first=True,
                              bidirectional=(num_gru_layers > 0), dropout=dropout if num_gru_layers > 1 else 0)
            if num_gru_layers == 0: # 如果没有 GRU 层，则使用线性层
                 self.gru = nn.Linear(input_dim, d_model)

        elif extractor_type == 'attention':
            # 模拟硕士论文的三层注意力（概念性）
            # 层 1: 股票时间序列内的注意力（时间 T 上的自注意力）
            # 层 2: 股票间的横截面注意力（股票 N 上的自注意力） - 批量中较难！需要仔细的填充/掩码或替代方法。
            # 层 3: 时间序列聚合注意力（关注过去步骤以获得最终输出） - 按图 26
            print("警告: 多层注意力实现较复杂，使用简化的自注意力。")
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout) # 添加位置编码

            # 简化: 使用标准 Transformer 编码器层
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        else:
            raise ValueError("未知提取器类型")

    def forward(self, x):
        # x 形状: (N, T, D_input)
        if self.extractor_type == 'gru':
            if isinstance(self.gru, nn.GRU):
                outputs, _ = self.gru(x) # outputs 形状: (N, T, D_model)
                # 使用最后一个时间步的输出
                final_features = outputs[:, -1, :] # 形状: (N, D_model)
            else: # 线性层情况
                final_features = self.gru(x[:,-1,:]) # 对最后一个时间步应用线性

        elif self.extractor_type == 'attention':
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            # Transformer 期望 (N, T, D_model)
            # TODO: 如果需要，实施掩码（例如，用于填充）
            memory = self.transformer_encoder(x) # 形状: (N, T, D_model)

            # --- 时间序列聚合（模拟硕士论文的层 3） ---
            # 使用最后一个隐藏状态作为查询，关注所有之前的状态（键/值）
            query = memory[:, -1:, :] # 形状: (N, 1, D_model)
            keys_values = memory # 形状: (N, T, D_model)
            # 基于注意力分数的简单加权平均（或使用另一个 MHA 层）
            # 这里使用点积注意力以简化：
            attn_weights = torch.bmm(query, keys_values.transpose(1, 2)) # (N, 1, T)
            attn_weights = torch.softmax(attn_weights / (self.d_model**0.5), dim=-1)
            final_features = torch.bmm(attn_weights, keys_values).squeeze(1) # (N, D_model)
            # 替代方法: 仅取最后一个输出状态
            # final_features = memory[:, -1, :]

        return final_features # 形状: (N, D_model)

class PositionalEncoding(nn.Module):
    # 标准位置编码
    # ...（实现省略以简洁，标准 PE 公式）...
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """参数: x: Tensor, 形状 [batch_size, seq_len, embedding_dim]"""
        x = x + self.pe[:x.size(1)].transpose(0,1) # 匹配 batch_first=True 格式
        return self.dropout(x)


class StockPredictor(nn.Module):
    """结合门控和特征提取的主模型。"""
    def __init__(self, d_stock_features_in, d_market_features, config):
        super().__init__()
        self.use_market_gating = config["use_market_gating"]
        self.d_model = config["d_model"]

        # 1. 可选的市场门控（在主要特征提取之前应用）
        if self.use_market_gating:
            self.market_gating = MarketGatingUnit(
                d_market_features, d_stock_features_in, # 门控应用于输入特征
                gating_type=config["gating_type"],
                beta=config["gating_beta"],
                n_heads=config["n_heads"]
            )
            feature_extractor_input_dim = d_stock_features_in
        else:
            self.market_gating = None
            feature_extractor_input_dim = d_stock_features_in

        # 2. 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=feature_extractor_input_dim,
            d_model=self.d_model,
            extractor_type=config["feature_extractor_type"],
            num_gru_layers=config["num_gru_layers"],
            num_attention_layers=config["num_attention_layers"],
            n_heads=config["n_heads"],
            dropout=config["dropout_rate"]
        )

        # 3. 最终预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(self.d_model // 2, 1) # 预测单值（收益）
        )

    def forward(self, stock_features_seq, market_features_seq):
        # stock_features_seq: (N, T, D_stock_in)
        # market_features_seq: (N, T, D_market)

        # 首先应用门控（如果启用）
        if self.market_gating:
            gated_stock_features = self.market_gating(stock_features_seq, market_features_seq)
        else:
            gated_stock_features = stock_features_seq

        # 从（可能门控的）序列中提取特征
        final_stock_representation = self.feature_extractor(gated_stock_features) # (N, D_model)

        # 预测
        prediction = self.prediction_head(final_stock_representation) # (N, 1)

        return prediction.squeeze(-1) # (N,)


# ----------------------------------------------------------------------------
# --- 6. 训练循环 ---
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, config):
    """训练 PyTorch 模型。"""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # 损失函数: MSE 常见，但基于 IC 的损失更适合因子排名
    # criterion = nn.MSELoss()
    criterion = RankICLoss() # 因子 IC 损失的占位实现（负 IC）

    best_val_loss = float('inf')
    patience_counter = 0

    print("开始训练...")
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for stock_seq, market_seq, target in train_pbar:
            stock_seq, market_seq, target = stock_seq.to(device), market_seq.to(device), target.to(device)

            optimizer.zero_grad()
            predictions = model(stock_seq, market_seq)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": loss.item()})

        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
             val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
             for stock_seq, market_seq, target in val_pbar:
                stock_seq, market_seq, target = stock_seq.to(device), market_seq.to(device), target.to(device)
                predictions = model(stock_seq, market_seq)
                loss = criterion(predictions, target)
                val_loss += loss.item()
                val_preds.append(predictions.cpu().numpy())
                val_targets.append(target.cpu().numpy())
                val_pbar.set_postfix({"loss": loss.item()})

        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        # 计算验证 Rank IC
        val_rank_ic = calculate_rank_ic(val_preds, val_targets)


        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Rank IC: {val_rank_ic:.4f}")

        # 提前停止
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型检查点
            torch.save(model.state_dict(), "best_model.pth")
            print("已保存最佳模型检查点。")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print("触发提前停止。")
                break

    print("训练完成。")
    # 加载最佳模型权重
    model.load_state_dict(torch.load("best_model.pth"))
    return model

# --- 辅助函数: IC 损失与计算 ---
def calculate_rank_ic(y_pred, y_true):
    """计算 Spearman 排名相关系数（Rank IC）。"""
    if len(y_pred) < 2: return 0.0
    ranked_pred = pd.Series(y_pred).rank()
    ranked_true = pd.Series(y_true).rank()
    return np.corrcoef(ranked_pred, ranked_true)[0, 1]

class RankICLoss(nn.Module):
    """Rank IC 损失的近似实现（负 IC）。"""
    def forward(self, y_pred, y_true):
        # 中心化预测和目标
        pred_centered = y_pred - y_pred.mean()
        target_centered = y_true - y_true.mean()
        # 计算协方差和标准差
        cov = (pred_centered * target_centered).mean()
        pred_std = torch.sqrt((pred_centered**2).mean() + 1e-6)
        target_std = torch.sqrt((target_centered**2).mean() + 1e-6)
        # 计算 Pearson 相关系数（对正态分布的排名近似 Rank IC）
        corr = cov / (pred_std * target_std)
        # 返回负相关系数（因为我们希望最大化 IC）
        return -corr


# ----------------------------------------------------------------------------
# --- 7. 预测 / 因子生成 ---
# ----------------------------------------------------------------------------
def generate_predictions(model, data_loader):
    """使用训练好的模型生成预测。"""
    model.eval()
    model.to(device)
    predictions = []
    print("生成预测...")
    with torch.no_grad():
        pred_pbar = tqdm(data_loader, desc="Predicting")
        for stock_seq, market_seq, _ in pred_pbar:
             stock_seq, market_seq = stock_seq.to(device), market_seq.to(device)
             preds = model(stock_seq, market_seq)
             predictions.append(preds.cpu().numpy())
    return np.concatenate(predictions)

# ----------------------------------------------------------------------------
# --- 8. 因子综合 ---
# ----------------------------------------------------------------------------
def synthesize_factors(factors_dict, weights="equal"):
    """将多个因子合成为一个综合因子。"""
    print("综合因子...")
    factor_df = pd.DataFrame(factors_dict)

    if weights == "equal":
        # 简单的等权重作为基线/报告方法
        composite_factor = factor_df.mean(axis=1)
    elif weights == "ml_based":
        # TODO: 实现基于机器学习的因子权重（例如，训练一个模型以从因子预测收益）
        print("警告: 基于机器学习的综合未实现。")
        composite_factor = factor_df.mean(axis=1) # 回退
    else:
        raise ValueError("未知综合权重类型")

    print("因子综合完成。")
    return composite_factor.rename("composite_factor")

# ----------------------------------------------------------------------------
# --- 9. 投资组合构建与回测 ---
# ----------------------------------------------------------------------------
def run_backtest(factor_series, stock_data, config):
    """运行指数增强回测。"""
    print("运行回测...")
    # TODO: 实现投资组合优化与回测逻辑
    # 1. 将因子与股票数据对齐（日期，资产）
    # 2. 在每个再平衡日期：
    #    a. 获取当前股票池的因子值
    #    b. 获取基准权重（例如，CSI 1000 权重）
    #    c. 获取约束所需的数据（行业、市值/风格因子）
    #    d. 运行优化器（例如，二次规划）以找到目标投资组合权重
    #       - 目标: 最大化预期 alpha（基于因子）或最小化跟踪误差
    #       - 约束: 换手率限制、行业限制、风格限制、股票限制等。
    # 3. 根据目标权重模拟交易（计算 PnL、换手率、成本）
    # 4. 计算绩效指标（夏普比率、信息比率、最大回撤、年化收益、Beta）

    # 占位实现：
    print("回测逻辑需要使用专用库或自定义代码进行完整实现。")
    performance_metrics = {
        "年化收益": 0.15,
        "年化波动率": 0.20,
        "夏普比率": 0.75,
        "最大回撤": -0.25,
        "信息比率": 2.0, # 占位 - 应与基准计算
        "年化换手率": 0.50 # 占位
    }
    print("回测完成（占位实现）。")
    return performance_metrics

# ----------------------------------------------------------------------------
# --- 10. 主执行逻辑 ---
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 加载基础数据 ---
    stock_data_full = load_stock_data(CONFIG["start_date"], CONFIG["test_end_date"], CONFIG["stock_universe_source"])
    market_features_full = load_market_features(CONFIG["market_index_codes"], CONFIG["start_date"], CONFIG["test_end_date"], CONFIG["market_feature_lookback"])

    all_predictions = {} # 存储不同模型/特征的预测

    # --- 遍历选定的特征集 ---
    for feature_set in CONFIG["feature_sets"]:
        print(f"\n--- 处理特征集: {feature_set} ---")

        # 1. 计算特征与目标
        features = calculate_features(stock_data_full, feature_set)
        target_returns = calculate_forward_returns(stock_data_full, CONFIG["prediction_horizon"])

        # 2. 数据预处理
        data = preprocess_data(features, market_features_full, target_returns)
        del features, target_returns; gc.collect() # 释放内存

        # --- 滚动窗口训练与预测 ---
        # 确定滚动窗口的起始/结束日期
        all_dates = data.index.get_level_values('date').unique().sort_values()
        test_dates = all_dates[all_dates >= pd.to_datetime(CONFIG["test_start_date"])]

        # 存储此特征集的预测
        feature_set_predictions = pd.Series(index=data.loc[test_dates].index, dtype=float)

        # 示例: 简化的滚动循环（需要正确的日期处理）
        # 实际中，您会循环测试期（例如，每月，每季度）
        # 并在该点之前的数据上重新训练模型。
        print("开始滚动窗口模拟（简化）...")

        # 定义第一个窗口的训练/验证/测试分割
        train_data = data[ (data.index.get_level_values('date') >= pd.to_datetime(CONFIG["train_start_date"])) &
                           (data.index.get_level_values('date') <= pd.to_datetime(CONFIG["train_end_date"])) ] # 根据滚动调整日期
        val_data = train_data # 简化: 使用训练的一部分或单独的时期作为验证
        test_data_current_window = data[ (data.index.get_level_values('date') >= pd.to_datetime(CONFIG["test_start_date"])) &
                                         (data.index.get_level_values('date') <= pd.to_datetime(CONFIG["test_end_date"])) ]


        # 为第一个窗口创建 DataLoader
        stock_cols = [c for c in data.columns if c.startswith(feature_set)]
        market_cols = [c for c in data.columns if c.startswith('mkt_feat_')]
        target_col = 'target_return'

        train_dataset = StockTimeSeriesDataset(train_data, stock_cols, market_cols, target_col, CONFIG["time_series_length"])
        val_dataset = StockTimeSeriesDataset(val_data, stock_cols, market_cols, target_col, CONFIG["time_series_length"]) # 使用验证分割
        test_dataset = StockTimeSeriesDataset(test_data_current_window, stock_cols, market_cols, target_col, CONFIG["time_series_length"])


        # 确保数据集不为空
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
             print(f"警告: 数据集为空，特征集 {feature_set}，跳过。")
             continue

        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=4, pin_memory=True)


        # 3. 初始化模型
        model = StockPredictor(
            d_stock_features_in=len(stock_cols),
            d_market_features=len(market_cols),
            config=CONFIG
        )

        # 4. 训练模型
        trained_model = train_model(model, train_loader, val_loader, CONFIG)

        # 5. 为此窗口的测试期生成预测
        # 需要将预测映射回原始多索引（日期，资产）
        # 数据集输出顺序需要保留或映射回。
        # 这部分较复杂，强烈依赖于数据集的实现。
        # 假设 test_loader 生成的数据与 test_data_current_window.index 对应：
        raw_predictions = generate_predictions(trained_model, test_loader)

        # 映射预测回去 - 需要从数据集中仔细处理索引
        # 假设数据集保留与 test_data_current_window 匹配的顺序
        # 并且长度匹配。这需要稳健的实现。
        if len(raw_predictions) == len(test_dataset.sequences):
             pred_index = [seq['original_index'] for seq in test_dataset.sequences] # 假设数据集存储原始索引
             feature_set_predictions.loc[pred_index] = raw_predictions # 分配预测
        else:
             print(f"警告: 预测长度与 {feature_set} 不匹配。检查数据集/加载器。")


        all_predictions[f"{CONFIG['feature_extractor_type']}_{feature_set}_{CONFIG['gating_type'] if CONFIG['use_market_gating'] else 'nogate'}"] = feature_set_predictions.dropna()
        print(f"已生成 {feature_set} 的预测。")

        del model, trained_model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, data
        gc.collect() # 在特征集之间释放内存

        # --- 简化滚动循环结束 ---
        # 正确的实现会为每个测试月/季度重新定义训练/验证/测试数据

    # --- 添加基础 AI 因子（占位实现） ---
    print("加载/生成基础 AI 因子（pv1, pv5, pv20）...")
    # TODO: 加载预计算因子或使用其他模型/方法生成
    for factor_name in CONFIG["base_ai_factors"]:
        # 与测试期索引对齐的虚拟因子
        test_period_index = pd.MultiIndex.from_tuples([]) # 获取实际测试索引
        if all_predictions:
             test_period_index = list(all_predictions.values())[0].index
        if not test_period_index.empty:
             all_predictions[factor_name] = pd.Series(np.random.randn(len(test_period_index)), index=test_period_index)
             print(f"已添加虚拟因子: {factor_name}")


    # --- 综合因子 ---
    # 选择最终策略对应的因子（例如，所有门控注意力因子 + 基础 AI 因子）
    factors_to_synthesize = {k: v for k, v in all_predictions.items()} # 示例中使用所有生成的因子
    if not factors_to_synthesize:
        print("错误: 未生成因子用于综合。退出。")
        exit()

    composite_factor = synthesize_factors(factors_to_synthesize, weights=CONFIG["synthesis_weights"])

    # --- 运行回测 ---
    # 确保因子索引与回测期所需的股票数据对齐
    backtest_stock_data = stock_data_full.loc[composite_factor.index.get_level_values('date').min():composite_factor.index.get_level_values('date').max()]
    # 将因子日期/资产与回测期的股票数据对齐
    aligned_factor, aligned_stock_data = composite_factor.align(backtest_stock_data['close'], join='inner', level=['date', 'asset'])


    if aligned_factor.empty:
         print("错误: 因子对齐导致数据为空。无法回测。")
    else:
         performance = run_backtest(aligned_factor, backtest_stock_data, CONFIG) # 传递完整股票数据以满足约束

         print("\n--- 回测绩效 ---")
         for metric, value in performance.items():
             print(f"{metric}: {value:.4f}")

    print("\n--- 脚本结束 ---")