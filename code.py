# ----------------------------------------------------------------------------
# --- 0. Setup & Imports ---
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split # Or time-series split
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc # Garbage collection

# Placeholder for feature calculation library (like qlib or custom)
# from feature_calculator import calculate_alpha158, calculate_moneyflow93, calculate_fundamental64

# Placeholder for backtesting library
# from backtester import Backtester, PortfolioOptimizer

print(f"PyTorch version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------------
# --- 1. Configuration ---
# ----------------------------------------------------------------------------
CONFIG = {
    # --- Data & Time ---
    "start_date": "2015-01-01", # Data start date (need history for features)
    "train_start_date": "2016-12-30", # As per report backtest start
    "train_end_date": "2023-12-31", # Example train end
    "test_start_date": "2024-01-01",
    "test_end_date": "2025-01-27", # As per report backtest end
    "rolling_window_years": 8, # Example: Use 8 years data for training each period
    "prediction_horizon": 10, # Predict T+10 return
    "market_index_codes": ["000300.SH", "000905.SH", "000852.SH", "000985.CSI"], # HS300, CSI500, CSI1000, All-Share
    "target_index_code": "000852.SH", # CSI 1000 for enhancement
    "stock_universe_source": "000985.CSI", # Use CSI All-Share constituents as pool

    # --- Features ---
    "feature_sets": ["alpha158", "moneyflow93", "fundamental64"], # Choose which base features to use
    "market_feature_lookback": [5, 10, 20, 30, 60], # Lookback days for market features
    "use_market_gating": True,
    "gating_type": "attention", # 'linear' or 'attention'
    "gating_beta": 3.0, # Softmax temperature for linear gating (report found 1-3 optimal)

    # --- Model ---
    "feature_extractor_type": "attention", # 'gru' or 'attention' (Master's 3-layer attention)
    "d_model": 64, # Model dimension
    "n_heads": 4, # Number of attention heads
    "num_gru_layers": 2, # For GRU extractor
    "num_attention_layers": 3, # For Master-like attention extractor
    "dropout_rate": 0.1,
    "time_series_length": 10, # Input sequence length (T) for stock features

    # --- Training ---
    "batch_size": 1024, # Adjust based on GPU memory
    "learning_rate": 1e-3,
    "weight_decay": 1e-4, # AdamW weight decay
    "epochs": 50, # Max epochs
    "early_stopping_patience": 10, # Stop if validation loss doesn't improve

    # --- Factor Synthesis ---
    "synthesis_weights": "equal", # 'equal' or potentially 'ml_based' in future
    "base_ai_factors": ["pv1", "pv5", "pv20"], # From report's reference model

    # --- Backtesting ---
    "rebalance_freq": "W", # Weekly rebalancing
    "commission_rate": 0.0003, # Transaction cost (3bps)
    "vwap_execution": True, # Assume VWAP execution (as per report)
    "turnover_limit": 0.20, # Max one-way turnover per rebalance (20%)
    "optimizer_constraints": { # As per report
        "min_in_universe_weight": 0.80, # Min 80% weight in target index constituents
        "industry_deviation": 0.02, # Max +/- 2% industry deviation vs benchmark
        "style_deviation": { # Max +/- 0.3 style factor exposure deviation (Size, NLS)
            "size": 0.3,
            "nls": 0.3, # Non-linear size placeholder
        },
        "max_stock_weight": 0.015, # Example max single stock weight (not explicitly mentioned but common)
        "max_stock_deviation": 0.01 # Example max deviation from benchmark weight
    }
}

# ----------------------------------------------------------------------------
# --- 2. Data Loading ---
# ----------------------------------------------------------------------------
def load_stock_data(start_date, end_date, universe_code):
    """Loads daily stock OHLCV, index constituents, etc."""
    print(f"Loading stock data from {start_date} to {end_date} for {universe_code}...")
    # TODO: Implement actual data loading from Wind, Tushare, JoinQuant, local DB, etc.
    # Needs: OHLCV, market cap, industry classification, index constituents, adj factor
    # Returns a multi-index DataFrame (date, asset)
    dummy_dates = pd.date_range(start_date, end_date)
    dummy_assets = [f"stock_{i:03d}.SZ" for i in range(5)]
    dummy_index = pd.MultiIndex.from_product([dummy_dates, dummy_assets], names=['date', 'asset'])
    dummy_data = pd.DataFrame(index=dummy_index)
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'market_cap', 'industry']:
        dummy_data[col] = np.random.rand(len(dummy_index))
    dummy_data['adj_factor'] = 1.0
    dummy_data['is_constituent'] = True # Assume all are constituents for placeholder
    print("Stock data loaded (placeholder).")
    return dummy_data

def load_market_features(index_codes, start_date, end_date, lookbacks):
    """Calculates and loads market features based on index OHLCV."""
    print("Loading market features...")
    market_features = {}
    # TODO: Load actual index OHLCV data
    for index_code in index_codes:
         # Calculate features like MR, MRAVG(d), MRSTD(d), MAAVG(d), MASTD(d) for d in lookbacks
         # This should return a DataFrame indexed by date
        dummy_dates = pd.date_range(start_date, end_date)
        num_features = 1 + len(lookbacks) * 4 # MR + 4 types * num_lookbacks
        market_features[index_code] = pd.DataFrame(
            np.random.randn(len(dummy_dates), num_features),
            index=dummy_dates,
            columns=[f"mkt_feat_{i}" for i in range(num_features)]
        )
    # Combine features from all indices (e.g., concatenate columns)
    all_market_features = pd.concat(market_features.values(), axis=1)
    print(f"Market features loaded (placeholder), shape: {all_market_features.shape}")
    return all_market_features

# ----------------------------------------------------------------------------
# --- 3. Feature Engineering ---
# ----------------------------------------------------------------------------
def calculate_features(stock_data, feature_set_name):
    """Calculates the specified feature set (Alpha158, MF93, Fund64)."""
    print(f"Calculating feature set: {feature_set_name}...")
    # TODO: Implement actual feature calculation using stock_data
    # This is complex and often relies on libraries like qlib or custom implementations.
    if feature_set_name == "alpha158":
        num_features = 158
    elif feature_set_name == "moneyflow93":
        num_features = 93
    elif feature_set_name == "fundamental64":
        num_features = 64
    else:
        raise ValueError("Unknown feature set")

    features = pd.DataFrame(
        np.random.randn(len(stock_data), num_features),
        index=stock_data.index,
        columns=[f"{feature_set_name}_{i}" for i in range(num_features)]
    )
    print(f"{feature_set_name} features calculated (placeholder), shape: {features.shape}")
    return features

def calculate_forward_returns(stock_data, horizon):
    """Calculates forward adjusted returns."""
    print(f"Calculating T+{horizon} forward returns...")
    # TODO: Implement actual forward return calculation using adjusted close prices
    forward_returns = stock_data['close'].groupby(level='asset').pct_change(periods=-horizon).shift(horizon)
    # Handle potential issues with adjusted prices if needed
    print("Forward returns calculated (placeholder).")
    return forward_returns.rename('target_return')

# ----------------------------------------------------------------------------
# --- 4. Data Preprocessing ---
# ----------------------------------------------------------------------------
def preprocess_data(features, market_features, target_returns):
    """Handles NaNs, normalization, standardization, etc."""
    print("Preprocessing data...")
    data = features.join(market_features, on='date').join(target_returns)

    # --- Handle NaNs ---
    # Option 1: Fill (e.g., with median, mean, zero) - careful about lookahead bias
    # data = data.groupby(level='date').transform(lambda x: x.fillna(x.median())) # Cross-sectional median fill
    # data = data.fillna(0) # Simple zero fill

    # Option 2: Drop stocks/dates with too many NaNs (or drop NaNs directly)
    data = data.dropna() # Simplest approach, might lose data

    # --- Feature Scaling / Normalization ---
    # Report mentions: Rank normalization, standardization, industry/market_cap neutralization
    # Applying cross-sectional rank normalization is common for factors
    # Applying time-series normalization (like RevIN mentioned in report survey) can help with drift

    feature_cols = [col for col in data.columns if col not in ['target_return'] and not col.startswith('mkt_feat_')]
    market_feature_cols = [col for col in data.columns if col.startswith('mkt_feat_')]
    target_col = 'target_return'

    # Example: Cross-sectional Rank Normalization (QuantileTransformer mimics ranking)
    def rank_normalize_group(group):
        # Use Gaussian output for better behavior with linear models/losses
        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        group[feature_cols] = qt.fit_transform(group[feature_cols])
        # Also rank normalize target for some loss functions (like Rank IC loss)
        # group[target_col] = qt.fit_transform(group[[target_col]])[:, 0]
        return group

    # data = data.groupby(level='date', group_keys=False).apply(rank_normalize_group)
    print("Rank normalization applied (placeholder action).")


    # Example: Standardization (Subtract mean, divide by std) - Applied cross-sectionally
    def standardize_group(group):
        scaler_feat = StandardScaler()
        scaler_mkt = StandardScaler()
        scaler_target = StandardScaler()
        group[feature_cols] = scaler_feat.fit_transform(group[feature_cols])
        group[market_feature_cols] = scaler_mkt.fit_transform(group[market_feature_cols])
        # Standardize target as well for stable training
        group[target_col] = scaler_target.fit_transform(group[[target_col]])[:, 0]
        return group

    data = data.groupby(level='date', group_keys=False).apply(standardize_group)
    print("Standardization applied.")

    # --- Neutralization (Optional but often helpful) ---
    # TODO: Implement industry/market_cap neutralization if needed
    # Requires regressing features on industry dummies/market cap factors and taking residuals

    print(f"Preprocessing complete. Data shape: {data.shape}")
    return data.dropna() # Drop NaNs again after potential transform issues


# --- Custom Dataset for Time Series ---
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

        # Precompute sequences for faster access (might use significant memory)
        self.sequences = self._precompute_sequences()

    def _precompute_sequences(self):
        print("Precomputing sequences...")
        sequences = []
        all_assets = self.data.index.get_level_values('asset').unique()
        # Group data by asset for efficient time series slicing
        grouped_by_asset = self.data.reset_index().set_index('date').groupby('asset')

        for date_idx, current_date in enumerate(tqdm(self.unique_dates)):
            if date_idx < self.seq_len - 1:
                continue # Not enough history

            start_date_idx = date_idx - self.seq_len + 1
            start_date = self.unique_dates[start_date_idx]

            # Get data for the current date
            current_data_slice = self.data.loc[current_date]
            if isinstance(current_data_slice, pd.Series): # Handle single stock case
                 current_data_slice = current_data_slice.to_frame().T
            current_assets = current_data_slice.index

            # Get market features for the sequence
            mkt_seq = self.data.loc[start_date:current_date, self.market_feature_cols].iloc[0:self.seq_len].values # Assumes market features are duplicated across stocks

            # Get stock features and target for each asset present on the current date
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
                    # Asset might not exist for the full sequence
                    continue
        print(f"Precomputed {len(sequences)} sequences.")
        return sequences


    def __len__(self):
         return len(self.sequences)

    def __getitem__(self, idx):
         return self.sequences[idx]['stock_features'], self.sequences[idx]['market_features'], self.sequences[idx]['target']


# ----------------------------------------------------------------------------
# --- 5. Model Definition (Key Components) ---
# ----------------------------------------------------------------------------

class MarketGatingUnit(nn.Module):
    """Applies market context gating to stock features."""
    def __init__(self, d_market_features, d_stock_features, gating_type='attention', beta=1.0, n_heads=4):
        super().__init__()
        self.gating_type = gating_type
        self.beta = beta
        self.d_stock_features = d_stock_features

        if gating_type == 'linear':
            # Linear projection from market features to stock feature weights
            self.market_to_weight = nn.Linear(d_market_features, d_stock_features)
            self.softmax = nn.Softmax(dim=-1)
        elif gating_type == 'attention':
            # Use market features (averaged over time?) to query stock features
            # Q: Market features, K: Market features, V: Stock features? Or Q: Stock, K: Market, V: Market?
            # Let's follow report: Market features --> weights for stock features
            # Q: Market, K: Dummy, V: Dummy -> gets weights; OR Market -> Linear -> weights
            # A simpler attention: Market features generate Q, K; Stock features are V.
            # Here, we interpret report's Fig 24: Market -> Linear -> Softmax -> Weights
            # Let's implement *that* attention idea slightly differently:
            # Use market context to modulate attention *within* the stock feature extractor later,
            # OR implement the explicit gating weight generation via attention:
            self.market_q = nn.Linear(d_market_features, d_stock_features) # Market features form query
            self.stock_k = nn.Linear(d_stock_features, d_stock_features) # Stock features form key
            self.attention = nn.MultiheadAttention(d_stock_features, n_heads, batch_first=True, dropout=0.1)
            # Output of attention could be weights, or used differently.
            # Sticking closer to Fig 24's *linear* path first for clarity in the gating unit:
            print("WARN: MarketGatingUnit attention type implementation is complex, using Linear for now.")
            self.gating_type = 'linear' # Revert to linear for simpler example skeleton
            self.market_to_weight = nn.Linear(d_market_features, d_stock_features)
            self.softmax = nn.Softmax(dim=-1) # Report uses Softmax(H_market / beta)
        else:
            raise ValueError("Unknown gating type")

    def forward(self, stock_features, market_features_seq):
        # stock_features: (N, T, D_stock) or (N*T, D_stock) if flattened
        # market_features_seq: (N, T, D_market) - market features over the sequence
        # N: batch size (number of stocks on a given day), T: sequence length

        # Use market features from the *last* time step as the current market context
        market_context = market_features_seq[:, -1, :] # Shape: (N, D_market)

        if self.gating_type == 'linear':
            # Project market context to weights: (N, D_market) -> (N, D_stock)
            market_weights = self.market_to_weight(market_context)

            # Apply temperature beta and softmax
            # Reshape weights to broadcast: (N, 1, D_stock) to multiply with (N, T, D_stock)
            # Apply weights to the *last* stock feature representation? Or the whole sequence?
            # Fig 24 implies weights (T x D_feature) applied to stock features (N x T x D_feature)
            # Let's assume weights are applied to the *final* feature representation (after extractor)
            # If applied *before* extractor:
            # Need weights shape (N, 1, D_stock) or (N, T, D_stock)
            # Let's try (N, 1, D_stock) applied to each step T.
            market_weights = self.softmax(market_weights / self.beta) # Shape: (N, D_stock)
            gating_weights = market_weights.unsqueeze(1) # Shape: (N, 1, D_stock)

            # Apply gating: Element-wise multiplication
            # Ensure stock_features has shape (N, T, D_stock)
            gated_stock_features = stock_features * gating_weights

        elif self.gating_type == 'attention':
            # TODO: Implement attention-based gating
            # Query (from market), Key (from stock?), Value (stock)
            # This needs careful design based on report's intent.
            print("WARN: Attention gating not fully implemented in skeleton.")
            gated_stock_features = stock_features # Pass through for now

        else: # No gating
             gated_stock_features = stock_features

        return gated_stock_features


class FeatureExtractor(nn.Module):
    """Extracts features using GRU or Multi-Level Attention."""
    def __init__(self, input_dim, d_model, extractor_type='attention', num_gru_layers=2, num_attention_layers=3, n_heads=4, dropout=0.1):
        super().__init__()
        self.extractor_type = extractor_type
        self.d_model = d_model

        if extractor_type == 'gru':
            # Simple BiGRU baseline
            self.gru = nn.GRU(input_dim, d_model // 2 if num_gru_layers > 0 else d_model,
                              num_layers=num_gru_layers, batch_first=True,
                              bidirectional=(num_gru_layers > 0), dropout=dropout if num_gru_layers > 1 else 0)
            if num_gru_layers == 0: # Use a linear layer if no GRU layers
                 self.gru = nn.Linear(input_dim, d_model)

        elif extractor_type == 'attention':
            # Simulate Master's 3-layer attention (conceptual)
            # Layer 1: Intra-Stock Temporal Attention (Self-attention over time T)
            # Layer 2: Inter-Stock Cross-Sectional Attention (Self-attention over stocks N) - Hard in batch! Needs careful padding/masking or alternative.
            # Layer 3: Temporal Aggregation Attention (Attend to past steps for final output) - As per Fig 26
            print("WARN: Multi-Level Attention is complex to implement correctly, using simplified Self-Attention.")
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_encoder = PositionalEncoding(d_model, dropout) # Add positional encoding

            # Simplified: Use standard Transformer Encoder layers
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)

        else:
            raise ValueError("Unknown extractor type")

    def forward(self, x):
        # x shape: (N, T, D_input)
        if self.extractor_type == 'gru':
            if isinstance(self.gru, nn.GRU):
                outputs, _ = self.gru(x) # outputs shape: (N, T, D_model)
                # Use the output of the last time step
                final_features = outputs[:, -1, :] # Shape: (N, D_model)
            else: # Linear layer case
                final_features = self.gru(x[:,-1,:]) # Apply linear to last time step

        elif self.extractor_type == 'attention':
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            # Transformer expects (N, T, D_model)
            # TODO: Implement masking if needed (e.g., for padding)
            memory = self.transformer_encoder(x) # Shape: (N, T, D_model)

            # --- Temporal Aggregation (Simulating Layer 3 of Master) ---
            # Use the last hidden state as Query, attend to all previous states (Keys/Values)
            query = memory[:, -1:, :] # Shape: (N, 1, D_model)
            keys_values = memory # Shape: (N, T, D_model)
            # Simple weighted average based on attention scores (or use another MHA layer)
            # Using dot product attention for simplicity here:
            attn_weights = torch.bmm(query, keys_values.transpose(1, 2)) # (N, 1, T)
            attn_weights = torch.softmax(attn_weights / (self.d_model**0.5), dim=-1)
            final_features = torch.bmm(attn_weights, keys_values).squeeze(1) # (N, D_model)
            # Alternative: Just take the last output state
            # final_features = memory[:, -1, :]

        return final_features # Shape: (N, D_model)

class PositionalEncoding(nn.Module):
    # Standard Positional Encoding
    # ... (Implementation omitted for brevity, standard PE formula) ...
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
        """Args: x: Tensor, shape [batch_size, seq_len, embedding_dim]"""
        x = x + self.pe[:x.size(1)].transpose(0,1) # Match batch_first=True format
        return self.dropout(x)


class StockPredictor(nn.Module):
    """Main model combining gating and feature extraction."""
    def __init__(self, d_stock_features_in, d_market_features, config):
        super().__init__()
        self.use_market_gating = config["use_market_gating"]
        self.d_model = config["d_model"]

        # 1. Optional Market Gating (applied *before* main feature extraction)
        if self.use_market_gating:
            self.market_gating = MarketGatingUnit(
                d_market_features, d_stock_features_in, # Gate applied to input features
                gating_type=config["gating_type"],
                beta=config["gating_beta"],
                n_heads=config["n_heads"]
            )
            feature_extractor_input_dim = d_stock_features_in
        else:
            self.market_gating = None
            feature_extractor_input_dim = d_stock_features_in

        # 2. Feature Extractor
        self.feature_extractor = FeatureExtractor(
            input_dim=feature_extractor_input_dim,
            d_model=self.d_model,
            extractor_type=config["feature_extractor_type"],
            num_gru_layers=config["num_gru_layers"],
            num_attention_layers=config["num_attention_layers"],
            n_heads=config["n_heads"],
            dropout=config["dropout_rate"]
        )

        # 3. Final Prediction Head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout_rate"]),
            nn.Linear(self.d_model // 2, 1) # Predict single value (return)
        )

    def forward(self, stock_features_seq, market_features_seq):
        # stock_features_seq: (N, T, D_stock_in)
        # market_features_seq: (N, T, D_market)

        # Apply gating first if enabled
        if self.market_gating:
            gated_stock_features = self.market_gating(stock_features_seq, market_features_seq)
        else:
            gated_stock_features = stock_features_seq

        # Extract features from (potentially gated) sequence
        final_stock_representation = self.feature_extractor(gated_stock_features) # (N, D_model)

        # Predict
        prediction = self.prediction_head(final_stock_representation) # (N, 1)

        return prediction.squeeze(-1) # (N,)


# ----------------------------------------------------------------------------
# --- 6. Training Loop ---
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, config):
    """Trains the PyTorch model."""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Loss function: MSE is common, but IC-based losses are better for ranking factors
    # criterion = nn.MSELoss()
    criterion = RankICLoss() # Placeholder for IC Loss (negative IC)

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
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

        # Validation
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
        # Calculate validation Rank IC
        val_rank_ic = calculate_rank_ic(val_preds, val_targets)


        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Rank IC: {val_rank_ic:.4f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model checkpoint
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    print("Training finished.")
    # Load best model weights
    model.load_state_dict(torch.load("best_model.pth"))
    return model

# --- Helper: IC Loss & Calculation ---
def calculate_rank_ic(y_pred, y_true):
    """Calculates Spearman Rank Correlation Coefficient (Rank IC)."""
    if len(y_pred) < 2: return 0.0
    ranked_pred = pd.Series(y_pred).rank()
    ranked_true = pd.Series(y_true).rank()
    return np.corrcoef(ranked_pred, ranked_true)[0, 1]

class RankICLoss(nn.Module):
    """Approximation of Rank IC Loss (negative IC)."""
    def forward(self, y_pred, y_true):
        # Center predictions and targets
        pred_centered = y_pred - y_pred.mean()
        target_centered = y_true - y_true.mean()
        # Calculate covariance and standard deviations
        cov = (pred_centered * target_centered).mean()
        pred_std = torch.sqrt((pred_centered**2).mean() + 1e-6)
        target_std = torch.sqrt((target_centered**2).mean() + 1e-6)
        # Calculate Pearson correlation (approximates Rank IC for normally distributed ranks)
        corr = cov / (pred_std * target_std)
        # Return negative correlation (since we want to maximize IC)
        return -corr


# ----------------------------------------------------------------------------
# --- 7. Prediction / Factor Generation ---
# ----------------------------------------------------------------------------
def generate_predictions(model, data_loader):
    """Generates predictions using the trained model."""
    model.eval()
    model.to(device)
    predictions = []
    print("Generating predictions...")
    with torch.no_grad():
        pred_pbar = tqdm(data_loader, desc="Predicting")
        for stock_seq, market_seq, _ in pred_pbar:
             stock_seq, market_seq = stock_seq.to(device), market_seq.to(device)
             preds = model(stock_seq, market_seq)
             predictions.append(preds.cpu().numpy())
    return np.concatenate(predictions)

# ----------------------------------------------------------------------------
# --- 8. Factor Synthesis ---
# ----------------------------------------------------------------------------
def synthesize_factors(factors_dict, weights="equal"):
    """Combines multiple factors into a composite factor."""
    print("Synthesizing factors...")
    factor_df = pd.DataFrame(factors_dict)

    if weights == "equal":
        # Simple equal weighting as baseline/report method
        composite_factor = factor_df.mean(axis=1)
    elif weights == "ml_based":
        # TODO: Implement ML-based factor weighting (e.g., train a model to predict returns from factors)
        print("WARN: ML-based synthesis not implemented.")
        composite_factor = factor_df.mean(axis=1) # Fallback
    else:
        raise ValueError("Unknown synthesis weights type")

    print("Factor synthesis complete.")
    return composite_factor.rename("composite_factor")

# ----------------------------------------------------------------------------
# --- 9. Portfolio Construction & Backtesting ---
# ----------------------------------------------------------------------------
def run_backtest(factor_series, stock_data, config):
    """Runs the index enhancement backtest."""
    print("Running backtest...")
    # TODO: Implement portfolio optimization and backtesting logic
    # 1. Align factor with stock data (dates, assets)
    # 2. On each rebalance date:
    #    a. Get current factor values for the stock universe
    #    b. Get benchmark weights (e.g., CSI 1000 weights)
    #    c. Get necessary data for constraints (industry, market cap/style factors)
    #    d. Run optimizer (e.g., quadratic programming) to find target portfolio weights
    #       - Objective: Maximize expected alpha (based on factor) or minimize tracking error
    #       - Constraints: Turnover limit, industry limits, style limits, stock limits, etc.
    # 3. Simulate trading based on target weights (calculate PnL, turnover, costs)
    # 4. Calculate performance metrics (Sharpe, Info Ratio, Max Drawdown, Annual Return, Beta)

    # Placeholder implementation:
    print("Backtesting logic needs full implementation using a dedicated library or custom code.")
    performance_metrics = {
        "Annualized Return": 0.15,
        "Annualized Volatility": 0.20,
        "Sharpe Ratio": 0.75,
        "Max Drawdown": -0.25,
        "Information Ratio": 2.0, # Placeholder - should be calculated vs benchmark
        "Annualized Turnover": 0.50 # Placeholder
    }
    print("Backtest complete (placeholder).")
    return performance_metrics

# ----------------------------------------------------------------------------
# --- 10. Main Execution Logic ---
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Load Base Data ---
    stock_data_full = load_stock_data(CONFIG["start_date"], CONFIG["test_end_date"], CONFIG["stock_universe_source"])
    market_features_full = load_market_features(CONFIG["market_index_codes"], CONFIG["start_date"], CONFIG["test_end_date"], CONFIG["market_feature_lookback"])

    all_predictions = {} # Store predictions from different models/features

    # --- Iterate through chosen feature sets ---
    for feature_set in CONFIG["feature_sets"]:
        print(f"\n--- Processing Feature Set: {feature_set} ---")

        # 1. Calculate Features & Target
        features = calculate_features(stock_data_full, feature_set)
        target_returns = calculate_forward_returns(stock_data_full, CONFIG["prediction_horizon"])

        # 2. Preprocess Data
        data = preprocess_data(features, market_features_full, target_returns)
        del features, target_returns; gc.collect() # Free memory

        # --- Rolling Window Training & Prediction ---
        # Determine rolling window start/end dates
        all_dates = data.index.get_level_values('date').unique().sort_values()
        test_dates = all_dates[all_dates >= pd.to_datetime(CONFIG["test_start_date"])]

        # Store predictions for this feature set
        feature_set_predictions = pd.Series(index=data.loc[test_dates].index, dtype=float)

        # Example: Simplified rolling loop (needs proper date handling)
        # In practice, you'd loop through test periods (e.g., monthly, quarterly)
        # and retrain the model on data up to that point.
        print("Starting Rolling Window Simulation (Simplified)...")

        # Define train/val/test split for the first window
        train_data = data[ (data.index.get_level_values('date') >= pd.to_datetime(CONFIG["train_start_date"])) &
                           (data.index.get_level_values('date') <= pd.to_datetime(CONFIG["train_end_date"])) ] # Adjust dates for rolling
        val_data = train_data # Simplification: Use part of train or a separate period for validation
        test_data_current_window = data[ (data.index.get_level_values('date') >= pd.to_datetime(CONFIG["test_start_date"])) &
                                         (data.index.get_level_values('date') <= pd.to_datetime(CONFIG["test_end_date"])) ]


        # Create DataLoaders for the first window
        stock_cols = [c for c in data.columns if c.startswith(feature_set)]
        market_cols = [c for c in data.columns if c.startswith('mkt_feat_')]
        target_col = 'target_return'

        train_dataset = StockTimeSeriesDataset(train_data, stock_cols, market_cols, target_col, CONFIG["time_series_length"])
        val_dataset = StockTimeSeriesDataset(val_data, stock_cols, market_cols, target_col, CONFIG["time_series_length"]) # Use validation split
        test_dataset = StockTimeSeriesDataset(test_data_current_window, stock_cols, market_cols, target_col, CONFIG["time_series_length"])


        # Ensure datasets are not empty
        if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
             print(f"WARN: Dataset empty for feature set {feature_set}, skipping.")
             continue

        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"] * 2, shuffle=False, num_workers=4, pin_memory=True)


        # 3. Initialize Model
        model = StockPredictor(
            d_stock_features_in=len(stock_cols),
            d_market_features=len(market_cols),
            config=CONFIG
        )

        # 4. Train Model
        trained_model = train_model(model, train_loader, val_loader, CONFIG)

        # 5. Generate Predictions for the test period of this window
        # Need to map predictions back to the original multi-index (date, asset)
        # The dataset output order needs to be preserved or mapped back.
        # This part is tricky and depends heavily on the Dataset implementation.
        # Assuming test_loader yields data corresponding to test_data_current_window.index:
        raw_predictions = generate_predictions(trained_model, test_loader)

        # Map predictions back - Requires careful index handling from dataset
        # This assumes the dataset preserves order matching test_data_current_window
        # And that the length matches. This needs robust implementation.
        if len(raw_predictions) == len(test_dataset.sequences):
             pred_index = [seq['original_index'] for seq in test_dataset.sequences] # Assumes dataset stores original index
             feature_set_predictions.loc[pred_index] = raw_predictions # Assign predictions
        else:
             print(f"WARN: Prediction length mismatch for {feature_set}. Check Dataset/Loader.")


        all_predictions[f"{CONFIG['feature_extractor_type']}_{feature_set}_{CONFIG['gating_type'] if CONFIG['use_market_gating'] else 'nogate'}"] = feature_set_predictions.dropna()
        print(f"Generated predictions for {feature_set}.")

        del model, trained_model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, data
        gc.collect() # Free memory between feature sets

        # --- End of simplified rolling loop ---
        # A proper implementation would re-define train/val/test data for each test month/quarter

    # --- Add Base AI Factors (Placeholder) ---
    print("Loading/Generating Base AI Factors (pv1, pv5, pv20)...")
    # TODO: Load pre-computed factors or generate them using another model/method
    for factor_name in CONFIG["base_ai_factors"]:
        # Dummy factors aligned with the test period index
        test_period_index = pd.MultiIndex.from_tuples([]) # Get the actual test index
        if all_predictions:
             test_period_index = list(all_predictions.values())[0].index
        if not test_period_index.empty:
             all_predictions[factor_name] = pd.Series(np.random.randn(len(test_period_index)), index=test_period_index)
             print(f"Added dummy factor: {factor_name}")


    # --- Synthesize Factors ---
    # Select the factors corresponding to the final strategy (e.g., all gated attention factors + base AI factors)
    factors_to_synthesize = {k: v for k, v in all_predictions.items()} # Use all generated for example
    if not factors_to_synthesize:
        print("ERROR: No factors generated for synthesis. Exiting.")
        exit()

    composite_factor = synthesize_factors(factors_to_synthesize, weights=CONFIG["synthesis_weights"])

    # --- Run Backtest ---
    # Ensure factor index aligns with required stock data for backtesting period
    backtest_stock_data = stock_data_full.loc[composite_factor.index.get_level_values('date').min():composite_factor.index.get_level_values('date').max()]
    # Align factor dates/assets with available stock data for backtesting
    aligned_factor, aligned_stock_data = composite_factor.align(backtest_stock_data['close'], join='inner', level=['date', 'asset'])


    if aligned_factor.empty:
         print("ERROR: Factor alignment resulted in empty data. Cannot backtest.")
    else:
         performance = run_backtest(aligned_factor, backtest_stock_data, CONFIG) # Pass full stock data for constraints

         print("\n--- Backtest Performance ---")
         for metric, value in performance.items():
             print(f"{metric}: {value:.4f}")

    print("\n--- End of Script ---")