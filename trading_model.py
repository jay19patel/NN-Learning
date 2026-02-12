# -*- coding: utf-8 -*-
"""
⭐⭐⭐ PROFESSIONAL MULTI-HEAD TRADING MODEL ⭐⭐⭐
PyTorch implementation with Attention, Risk Management, and Actionable Signals

Features:
- Multi-head architecture (Direction, Confidence, Risk)
- Attention mechanism for pattern learning
- Risk-aware loss function
- Automatic Stop-Loss and Take-Profit calculation
- Position sizing based on confidence
- Proper train/validation/test split
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ⭐⭐⭐ MULTI-HEAD TRADING NEURAL NETWORK
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Self-attention mechanism to learn temporal patterns"""
    
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Attention scores
        energy = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(energy, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        
        return self.fc_out(out)


class TradingTransformerBlock(nn.Module):
    """Transformer block with attention and feedforward"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Attention with residual
        attended = self.attention(x)
        x = self.norm1(x + attended)
        
        # FFN with residual
        forwarded = self.ffn(x)
        x = self.norm2(x + forwarded)
        
        return x


class MultiHeadTradingModel(nn.Module):
    """
    ⭐⭐⭐ PROFESSIONAL MULTI-HEAD TRADING MODEL ⭐⭐⭐
    
    Outputs:
    1. Direction: Buy(1), Sell(-1), or Hold(0) - 3 classes
    2. Confidence: 0 to 1 - How confident is the model
    3. Upside Prediction: Expected % upside
    4. Downside Prediction: Expected % downside
    5. Risk Score: 0 to 1 - How risky is this trade
    """
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.2):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers for pattern learning
        self.transformer_blocks = nn.ModuleList([
            TradingTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # ⭐ HEAD 1: Direction Classification (Buy/Sell/Hold)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # 3 classes: Buy(0), Hold(1), Sell(2)
        )
        
        # ⭐ HEAD 2: Confidence Score (0-1)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 to 1
        )
        
        # ⭐ HEAD 3: Upside Prediction (%)
        self.upside_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # ⭐ HEAD 4: Downside Prediction (%)
        self.downside_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # ⭐ HEAD 5: Risk Score (0-1)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0 to 1
        )
        
    def forward(self, x):
        # Input embedding
        x = self.input_projection(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer layers
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Shared representation
        shared = self.shared_layer(x)
        
        # Multi-head outputs
        direction = self.direction_head(shared)  # 3 classes
        confidence = self.confidence_head(shared)  # 0-1
        upside = self.upside_head(shared)  # %
        downside = self.downside_head(shared)  # %
        risk = self.risk_head(shared)  # 0-1
        
        return {
            'direction': direction,
            'confidence': confidence,
            'upside': upside,
            'downside': downside,
            'risk': risk
        }


# ============================================================================
# ⭐⭐⭐ RISK-AWARE LOSS FUNCTION
# ============================================================================

class RiskAwareLoss(nn.Module):
    """
    Custom loss function that:
    1. Penalizes wrong high-confidence predictions MORE
    2. Rewards correct high-confidence predictions MORE
    3. Considers risk-reward ratio
    4. Balances all prediction tasks
    """
    
    def __init__(self, 
                 direction_weight=2.0,
                 confidence_weight=1.0,
                 upside_weight=1.5,
                 downside_weight=1.5,
                 risk_weight=1.0):
        super().__init__()
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.upside_weight = upside_weight
        self.downside_weight = downside_weight
        self.risk_weight = risk_weight
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, predictions, targets):
        """
        predictions: dict with model outputs
        targets: dict with ground truth
        """
        batch_size = predictions['direction'].shape[0]
        
        # 1. Direction Loss (Confidence-weighted)
        direction_loss = self.ce_loss(
            predictions['direction'], 
            targets['direction']
        )
        # ⭐ HIGH CONFIDENCE wrong prediction = HIGH penalty
        confidence_weight = 1 + predictions['confidence'].squeeze() * 2  # 1 to 3x
        direction_loss = (direction_loss * confidence_weight).mean()
        
        # 2. Confidence Loss (Should be high when prediction is correct)
        direction_correct = (predictions['direction'].argmax(dim=1) == targets['direction']).float()
        confidence_target = direction_correct  # 1 if correct, 0 if wrong
        confidence_loss = self.mse_loss(
            predictions['confidence'].squeeze(),
            confidence_target
        ).mean()
        
        # 3. Upside Prediction Loss
        upside_loss = self.mse_loss(
            predictions['upside'].squeeze(),
            targets['upside']
        ).mean()
        
        # 4. Downside Prediction Loss
        downside_loss = self.mse_loss(
            predictions['downside'].squeeze(),
            targets['downside']
        ).mean()
        
        # 5. Risk Score Loss (High risk when future drawdown is high)
        risk_target = torch.clamp(
            torch.abs(targets['future_drawdown']) / 20.0,  # Normalize to 0-1
            0, 1
        )
        risk_loss = self.mse_loss(
            predictions['risk'].squeeze(),
            risk_target
        ).mean()
        
        # Combined loss
        total_loss = (
            self.direction_weight * direction_loss +
            self.confidence_weight * confidence_loss +
            self.upside_weight * upside_loss +
            self.downside_weight * downside_loss +
            self.risk_weight * risk_loss
        )
        
        return {
            'total': total_loss,
            'direction': direction_loss,
            'confidence': confidence_loss,
            'upside': upside_loss,
            'downside': downside_loss,
            'risk': risk_loss
        }


# ============================================================================
# DATASET CLASS
# ============================================================================

class TradingDataset(Dataset):
    """PyTorch Dataset for trading data"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': {
                'direction': torch.LongTensor([self.targets['direction'][idx]])[0],
                'upside': torch.FloatTensor([self.targets['upside'][idx]])[0],
                'downside': torch.FloatTensor([self.targets['downside'][idx]])[0],
                'future_drawdown': torch.FloatTensor([self.targets['future_drawdown'][idx]])[0]
            }
        }


# ============================================================================
# ⭐⭐⭐ TRAINING PIPELINE
# ============================================================================

class TradingModelTrainer:
    """Complete training pipeline with validation and model saving"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = RiskAwareLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            # Calculate loss
            losses = self.criterion(predictions, targets)
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                
                predictions = self.model(features)
                losses = self.criterion(predictions, targets)
                total_loss += losses['total'].item()
        
        return total_loss / len(val_loader)
    
    def fit(self, train_loader, val_loader, epochs=100, early_stop_patience=10):
        """Complete training loop"""
        print("🚀 Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 80)
        
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_trading_model.pth')
                patience_counter = 0
                best_marker = "⭐ NEW BEST"
            else:
                patience_counter += 1
                best_marker = ""
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} {best_marker}")
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        print("\n✅ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_trading_model.pth'))


# ============================================================================
# ⭐⭐⭐ INFERENCE & TRADING SIGNALS
# ============================================================================

class TradingSignalGenerator:
    """
    Generates actionable trading signals with:
    - Direction (Buy/Sell/Hold)
    - Confidence (0-100%)
    - Entry Price
    - Stop Loss
    - Take Profit
    - Position Size
    - Risk-Reward Ratio
    """
    
    def __init__(self, model, scaler, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.device = device
        
    def generate_signal(self, features, current_price, account_size=10000, max_risk_per_trade=0.02):
        """
        Generate trading signal for a single data point
        
        Args:
            features: DataFrame row or numpy array of features
            current_price: Current market price
            account_size: Total account size
            max_risk_per_trade: Maximum risk per trade (e.g., 2% = 0.02)
        
        Returns:
            dict with complete trading signal
        """
        # Prepare features
        if isinstance(features, pd.Series):
            features = features.values.reshape(1, -1)
        elif isinstance(features, np.ndarray) and len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.model(features_tensor)
        
        # Extract predictions
        direction_probs = F.softmax(predictions['direction'], dim=1)[0]
        direction_class = direction_probs.argmax().item()  # 0=Buy, 1=Hold, 2=Sell
        confidence = predictions['confidence'][0].item()
        upside = predictions['upside'][0].item()
        downside = predictions['downside'][0].item()
        risk_score = predictions['risk'][0].item()
        
        # Map direction
        direction_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
        direction = direction_map[direction_class]
        
        # Calculate Stop Loss and Take Profit
        if direction == 'BUY':
            stop_loss = current_price * (1 + downside / 100)  # Downside is negative
            take_profit = current_price * (1 + upside / 100)
        elif direction == 'SELL':
            stop_loss = current_price * (1 - downside / 100)
            take_profit = current_price * (1 - upside / 100)
        else:  # HOLD
            stop_loss = current_price
            take_profit = current_price
        
        # Calculate position size based on confidence and risk
        risk_amount = account_size * max_risk_per_trade
        stop_loss_distance = abs(current_price - stop_loss)
        
        if stop_loss_distance > 0:
            position_size = risk_amount / stop_loss_distance
            
            # ⭐ Adjust position size based on confidence
            # High confidence = larger position
            # Low confidence = smaller position
            confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5x to 2x
            position_size = position_size * confidence_multiplier
            
            # Cap position size at 20% of account
            max_position = account_size * 0.2 / current_price
            position_size = min(position_size, max_position)
        else:
            position_size = 0
        
        # Risk-Reward Ratio
        if abs(downside) > 0.01:
            risk_reward = abs(upside / downside)
        else:
            risk_reward = 0
        
        # Trading signal
        signal = {
            'timestamp': pd.Timestamp.now(),
            'direction': direction,
            'confidence': confidence * 100,  # Convert to percentage
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_score': risk_score * 100,  # Convert to percentage
            'expected_upside': upside,
            'expected_downside': downside,
            'risk_reward_ratio': risk_reward,
            'max_loss': abs(current_price - stop_loss) * position_size,
            'max_profit': abs(take_profit - current_price) * position_size,
            
            # Raw probabilities
            'buy_probability': direction_probs[0].item() * 100,
            'hold_probability': direction_probs[1].item() * 100,
            'sell_probability': direction_probs[2].item() * 100
        }
        
        return signal
    
    def print_signal(self, signal):
        """Pretty print trading signal"""
        print("\n" + "="*80)
        print("⭐⭐⭐ TRADING SIGNAL ⭐⭐⭐")
        print("="*80)
        print(f"📅 Time: {signal['timestamp']}")
        print(f"🎯 Direction: {signal['direction']} (Confidence: {signal['confidence']:.1f}%)")
        print(f"💰 Entry Price: ${signal['entry_price']:.2f}")
        print(f"🛑 Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"🎁 Take Profit: ${signal['take_profit']:.2f}")
        print(f"📊 Position Size: {signal['position_size']:.2f} units")
        print(f"⚠️  Risk Score: {signal['risk_score']:.1f}%")
        print(f"📈 Expected Upside: {signal['expected_upside']:.2f}%")
        print(f"📉 Expected Downside: {signal['expected_downside']:.2f}%")
        print(f"⚖️  Risk-Reward: {signal['risk_reward_ratio']:.2f}")
        print(f"💵 Max Loss: ${signal['max_loss']:.2f}")
        print(f"💰 Max Profit: ${signal['max_profit']:.2f}")
        print("-"*80)
        print(f"Buy: {signal['buy_probability']:.1f}% | "
              f"Hold: {signal['hold_probability']:.1f}% | "
              f"Sell: {signal['sell_probability']:.1f}%")
        print("="*80)


# ============================================================================
# ⭐ EXAMPLE USAGE
# ============================================================================

def prepare_data_for_training(df):
    """
    Prepare DataFrame for model training
    
    Args:
        df: DataFrame with all features from data_prepare_organized.py
    
    Returns:
        train_loader, val_loader, test_loader, scaler, feature_columns
    """
    print("📊 Preparing data for training...")
    
    # Remove NaN values and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_clean = df.dropna()
    
    # Define target columns
    target_cols = ['upside_pct', 'downside_pct', 'future_drawdown_pct']
    
    # Feature columns (exclude OHLCV and targets)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + target_cols
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    print(f"✅ Features: {len(feature_cols)}")
    print(f"✅ Samples: {len(df_clean)}")
    
    # Create direction labels
    # Buy(0) if upside > downside * 2
    # Sell(2) if downside < upside * 2
    # Hold(1) otherwise
    df_clean['direction_label'] = 1  # Default Hold
    buy_condition = df_clean['upside_pct'] > abs(df_clean['downside_pct']) * 2
    sell_condition = abs(df_clean['downside_pct']) > df_clean['upside_pct'] * 2
    df_clean.loc[buy_condition, 'direction_label'] = 0  # Buy
    df_clean.loc[sell_condition, 'direction_label'] = 2  # Sell
    
    # Features and targets
    X = df_clean[feature_cols].values
    y = {
        'direction': df_clean['direction_label'].values,
        'upside': df_clean['upside_pct'].values,
        'downside': df_clean['downside_pct'].values,
        'future_drawdown': df_clean['future_drawdown_pct'].values
    }
    
    # Train/Val/Test split (70/15/15)
    # ⭐ IMPORTANT: Use chronological split for time series
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train = {k: v[:train_size] for k, v in y.items()}
    y_val = {k: v[train_size:train_size+val_size] for k, v in y.items()}
    y_test = {k: v[train_size+val_size:] for k, v in y.items()}
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = TradingDataset(X_train_scaled, y_train)
    val_dataset = TradingDataset(X_val_scaled, y_val)
    test_dataset = TradingDataset(X_test_scaled, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"✅ Train samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    print(f"✅ Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, scaler, feature_cols


if __name__ == "__main__":
    print("⭐⭐⭐ Multi-Head Trading Model - Ready to Use! ⭐⭐⭐")
    print("\nThis module provides:")
    print("1. MultiHeadTradingModel - Advanced PyTorch model")
    print("2. TradingModelTrainer - Complete training pipeline")
    print("3. TradingSignalGenerator - Actionable trading signals")
    print("\nSee train_model.py for complete training example")
