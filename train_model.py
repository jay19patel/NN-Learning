# -*- coding: utf-8 -*-
"""
⭐⭐⭐ COMPLETE TRAINING PIPELINE ⭐⭐⭐
Train the multi-head trading model and generate signals
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from trading_model import (
    MultiHeadTradingModel,
    TradingModelTrainer,
    TradingSignalGenerator,
    prepare_data_for_training
)
from data_prepare_organized import fetch_data, create_full_feature_set
import warnings
warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = 'model_outputs'

# ============================================================================
# STEP 1: LOAD OR GENERATE DATA
# ============================================================================

def load_and_prepare_data(csv_path=None, generate_new=True, days=500):
    """
    Load existing data or generate new data
    
    Args:
        csv_path: Path to existing CSV file (optional)
        generate_new: If True, generate synthetic data
        days: Number of days to generate
    
    Returns:
        DataFrame with all features
    """
    print("="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    if csv_path and not generate_new and os.path.exists(csv_path):
        print(f"📂 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"🔧 Fetching {days} days of OHLC data...")
        # Use fetch_data instead of generate_ohlc_data
        df_ohlc = fetch_data(total_days=days)
        
        print("🔧 Creating feature set...")
        df = create_full_feature_set(df_ohlc, lookahead=20)
    
    print(f"✅ Data shape: {df.shape}")
    print(f"✅ Features: {df.shape[1]}")
    print(f"✅ Samples: {df.shape[0]}")
    
    return df


# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================

from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# STEP 2: TRAIN MODEL
# ============================================================================

def train_trading_model(df, epochs=50):
    """
    Train the multi-head trading model
    
    Args:
        df: DataFrame with features
        epochs: Number of training epochs
    
    Returns:
        model, trainer, scaler, feature_cols, test_loader
    """
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_cols = prepare_data_for_training(df)
    
    # Calculate class weights to handle imbalance
    train_labels = train_loader.dataset.targets['direction']
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.FloatTensor(class_weights)
    print(f"⚖️ Class Weights: {class_weights}")
    
    # Initialize model
    input_dim = len(feature_cols)
    print(f"\n🔧 Initializing model with {input_dim} input features...")
    
    model = MultiHeadTradingModel(
        input_dim=input_dim,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        dropout=0.2
    )
    
    # Initialize trainer with output directory and class weights
    trainer = TradingModelTrainer(model, model_dir=OUTPUT_DIR, class_weights=class_weights)
    
    # Train model
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stop_patience=15
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("STEP 3: TEST SET EVALUATION")
    print("="*80)
    test_loss = trainer.validate(test_loader)
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    return model, trainer, scaler, feature_cols, test_loader


# ============================================================================
# STEP 3: SEQUENTIAL BACKTEST
# ============================================================================

def run_sequential_backtest(df, model, scaler, feature_cols, initial_capital=10000):
    """
    Run a realistic sequential backtest on the test set.
    Rules:
    - Iterate row by row through the test set.
    - If no position is open: check for entry signals.
    - If position is open: check for exit (TP/SL) against High/Low of current bar.
    - Only one trade active at a time.
    """
    print("\n" + "="*80)
    print("STEP 4: SEQUENTIAL BACKTEST (REALISTIC)")
    print("="*80)
    
    # Get test data (last 15% roughly, aligning with prepare_data_for_training split)
    # Re-creating the split logic to get exact indices
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    train_size = int(len(df_clean) * 0.7)
    val_size = int(len(df_clean) * 0.15)
    test_start_idx = train_size + val_size
    
    test_df = df_clean.iloc[test_start_idx:].copy()
    print(f"📊 Backtesting on {len(test_df)} recent candles...")
    print(f"🛡️ Confidence Threshold: 0.6")
    
    signal_gen = TradingSignalGenerator(model, scaler)
    
    capital = initial_capital
    position = None # {direction, entry_price, size, sl, tp, entry_time}
    trades = []
    
    for idx, row in test_df.iterrows():
        current_price = row['Close']
        current_high = row['High']
        current_low = row['Low']
        current_time = idx
        
        # Check active position
        if position:
            exit_reason = None
            exit_price = None
            
            # ⭐ TRAILING STOP LOGIC
            if position['direction'] == 'BUY':
                # Calculate current max profit potential
                profit_pct = (current_high - position['entry_price']) / position['entry_price']
                
                # Move SL to Breakeven if > 0.2% profit
                if profit_pct > 0.002:
                    new_sl = position['entry_price'] * 1.0005 # BE + small buffer
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                        
                # Trailing step if > 0.4% profit: lock 0.25%
                if profit_pct > 0.004:
                    new_sl = position['entry_price'] * 1.0025
                    if new_sl > position['sl']:
                        position['sl'] = new_sl
                
                # Check SL (Hit Low?)
                if current_low <= position['sl']:
                    exit_price = position['sl'] 
                    exit_reason = 'STOP_LOSS'
                # Check TP (Hit High?)
                elif current_high >= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TAKE_PROFIT'
                
                if exit_price:
                    pnl = (exit_price - position['entry_price']) * position['size']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': 'BUY',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'capital': capital,
                        'reason': exit_reason
                    })
                    position = None
                    continue
            
            elif position['direction'] == 'SELL':
                # Calculate current max profit potential (price goes down)
                profit_pct = (position['entry_price'] - current_low) / position['entry_price']
                
                # Move SL to Breakeven if > 0.2% profit
                if profit_pct > 0.002:
                    new_sl = position['entry_price'] * 0.9995 # BE + small buffer
                    if new_sl < position['sl']:
                        position['sl'] = new_sl
                        
                # Trailing step if > 0.4% profit: lock 0.25%
                if profit_pct > 0.004:
                    new_sl = position['entry_price'] * 0.9975
                    if new_sl < position['sl']:
                        position['sl'] = new_sl
                
                # Check SL (Hit High?)
                if current_high >= position['sl']:
                    exit_price = position['sl']
                    exit_reason = 'STOP_LOSS'
                # Check TP (Hit Low?)
                elif current_low <= position['tp']:
                    exit_price = position['tp']
                    exit_reason = 'TAKE_PROFIT'
                
                if exit_price:
                    pnl = (position['entry_price'] - exit_price) * position['size']
                    capital += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': 'SELL',
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'capital': capital,
                        'reason': exit_reason
                    })
                    position = None
                    continue

        # No active position (or just closed), check for entry
        if position is None:
            features = row[feature_cols].values
            signal = signal_gen.generate_signal(
                features=features,
                current_price=current_price,
                account_size=capital,
                max_risk_per_trade=0.02,
                threshold=0.6
            )
            
            if signal['direction'] != 'HOLD':
                # Open Position
                position = {
                    'direction': signal['direction'],
                    'entry_price': current_price,
                    'size': signal['position_size'],
                    'sl': signal['stop_loss'],
                    'tp': signal['take_profit'],
                    'entry_time': current_time
                }
                # print(f"OPEN {signal['direction']} at {current_time} @ {current_price}")

    # Analyze results
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        total_trades = len(trades)
        profitable = trades_df[trades_df['pnl'] > 0]
        win_rate = len(profitable) / total_trades * 100
        total_pnl = capital - initial_capital
        roi = (total_pnl / initial_capital) * 100
        
        print(f"📊 Total Trades: {total_trades}")
        print(f"✅ Win Rate: {win_rate:.2f}%")
        print(f"💰 Total PnL: ${total_pnl:.2f}")
        print(f"📈 ROI: {roi:.2f}%")
        print(f"💵 Final Capital: ${capital:.2f}")
        
        # Save trades
        trades_file = os.path.join(OUTPUT_DIR, 'backtest_trades.csv')
        trades_df.to_csv(trades_file)
        print(f"✅ Trade history saved to: {trades_file}")
        
        # Plot historical trades with entries, exits, and PnL
        import matplotlib.dates as mdates
        
        print("\n📊 Plotting historical trades...")
        plt.figure(figsize=(16, 12))
        
        # Price and Trade Markers
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(test_df.index, test_df['Close'], label='Price', color='black', alpha=0.5, linewidth=1)
        
        for t in trades:
            color = 'green' if t['pnl'] > 0 else 'red'
            marker = '^' if t['direction'] == 'BUY' else 'v'
            
            # Entry point
            ax1.scatter(t['entry_time'], t['entry_price'], color='blue', marker=marker, s=100, zorder=5)
            # Exit point
            ax1.scatter(t['exit_time'], t['exit_price'], color=color, marker='X' if t['pnl'] > 0 else 'x', s=100, zorder=5)
            # Connect them
            ax1.plot([t['entry_time'], t['exit_time']], [t['entry_price'], t['exit_price']], color=color, linestyle='--', alpha=0.7)
            
            # Label
            ax1.text(t['exit_time'], t['exit_price'], f" ${t['pnl']:.2f}", color=color, fontsize=10, fontweight='bold')
            
        ax1.set_title('Historical Trades: Entries, Exits, and PnL', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Custom legend elements
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='blue', marker='^', linestyle='None', markersize=10, label='BUY Entry'),
            Line2D([0], [0], color='blue', marker='v', linestyle='None', markersize=10, label='SELL Entry'),
            Line2D([0], [0], color='green', marker='X', linestyle='None', markersize=10, label='Exit (Profit)'),
            Line2D([0], [0], color='red', marker='x', linestyle='None', markersize=10, label='Exit (Loss)')
        ]
        ax1.legend(handles=custom_lines, loc='upper left')
        
        # Cumulative PnL
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        ax2.plot(trades_df['exit_time'], trades_df['cumulative_pnl'], color='green', drawstyle='steps-post', linewidth=2)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Cumulative PnL Over Time', fontsize=14)
        ax2.set_ylabel('Cumulative PnL ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, 'historical_trades.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved historical trades plot to: {plot_path}")
    else:
        print("⚠️ No trades executed in the test period.")
        
    return trades

# ============================================================================
# STEP 4: PLOT TRAINING HISTORY
# ============================================================================

def plot_training_history(trainer):
    """Plot training and validation loss"""
    print("\n📊 Plotting training history...")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer.val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_losses, label='Train Loss', linewidth=2)
    plt.plot(trainer.val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete end-to-end pipeline"""
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "⭐"*40)
    print("⭐⭐⭐ PROFESSIONAL TRADING MODEL PIPELINE ⭐⭐⭐")
    print("⭐"*40 + "\n")
    
    # Step 1: Load/Generate Data
    # For training, we need features. 
    # data_prepare_organized.py handles fetching/calculation but doesn't auto-save
    # features_output.csv anymore.
    # So we call load_and_prepare_data which calls fetch_data + create_full_feature_set
    
    df = load_and_prepare_data(
        csv_path=os.path.join(OUTPUT_DIR, 'features_output.csv'), # Try to load from here if exists?
        generate_new=True, # Always regenerate for now to ensure freshness or fetch from API cache
        days=500
    )
    
    # Step 2: Train Model
    model, trainer, scaler, feature_cols, test_loader = train_trading_model(
        df=df,
        epochs=50 
    )
    
    # Save scaler and feature columns for inference (Disabled per user request)
    # joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.sav'))
    # joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, 'feature_cols.sav'))
    # print(f"✅ Saved scaler and feature columns to {OUTPUT_DIR}")
    
    # Step 3: Plot Training History (Disabled per user request)
    # plot_training_history(trainer)
    
    # Step 4: Sequential Backtest
    run_sequential_backtest(
        df=df,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        initial_capital=10000
    )
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print(f"All outputs saved in: {OUTPUT_DIR}/")
    print("="*80)


# ============================================================================
# INFERENCE MODE (For live trading)
# ============================================================================

def live_trading_mode(df_latest, model_path=None):
    """
    Use trained model for live trading
    """
    if model_path is None:
        model_path = os.path.join(OUTPUT_DIR, 'best_trading_model.pth')
        
    print("\n" + "="*80)
    print("🔴 LIVE TRADING MODE")
    print("="*80)
    
    # Load model, scaler, feature_cols
    # ... implementation details ...
    
    print("⚠️  Implement detailed live trading logic using loaded artifacts")


if __name__ == "__main__":
    main()
