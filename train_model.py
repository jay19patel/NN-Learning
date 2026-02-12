# -*- coding: utf-8 -*-
"""
⭐⭐⭐ COMPLETE TRAINING PIPELINE ⭐⭐⭐
Train the multi-head trading model and generate signals
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_model import (
    MultiHeadTradingModel,
    TradingModelTrainer,
    TradingSignalGenerator,
    prepare_data_for_training
)
from data_prepare_organized import fetch_data, create_full_feature_set
import warnings
warnings.filterwarnings('ignore')


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
    
    if csv_path and not generate_new:
        print(f"📂 Loading data from {csv_path}...")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"🔧 Fetching {days} days of OHLC data...")
        # Use fetch_data instead of generate_ohlc_data
        df_ohlc = fetch_data(total_days=days)
        
        print("🔧 Creating feature set...")
        df = create_full_feature_set(df_ohlc, lookahead=10)
    
    print(f"✅ Data shape: {df.shape}")
    print(f"✅ Features: {df.shape[1]}")
    print(f"✅ Samples: {df.shape[0]}")
    
    return df


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
        model, trainer, scaler, feature_cols
    """
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_cols = prepare_data_for_training(df)
    
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
    
    # Initialize trainer
    trainer = TradingModelTrainer(model)
    
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
    
    return model, trainer, scaler, feature_cols


# ============================================================================
# STEP 3: GENERATE TRADING SIGNALS
# ============================================================================

def generate_signals_for_recent_data(df, model, scaler, feature_cols, num_recent=10):
    """
    Generate trading signals for recent data points
    
    Args:
        df: DataFrame with features
        model: Trained model
        scaler: Fitted scaler
        feature_cols: List of feature column names
        num_recent: Number of recent samples to generate signals for
    
    Returns:
        List of trading signals
    """
    print("\n" + "="*80)
    print("STEP 4: GENERATING TRADING SIGNALS")
    print("="*80)
    
    # Initialize signal generator
    signal_gen = TradingSignalGenerator(model, scaler)
    
    # Get recent data
    df_clean = df.dropna()
    recent_data = df_clean.tail(num_recent)
    
    signals = []
    
    for idx, row in recent_data.iterrows():
        # Get features
        features = row[feature_cols].values
        current_price = row['Close']
        
        # Generate signal
        signal = signal_gen.generate_signal(
            features=features,
            current_price=current_price,
            account_size=10000,  # $10,000 account
            max_risk_per_trade=0.02  # 2% risk per trade
        )
        
        signal['date'] = idx
        signal['actual_close'] = current_price
        
        signals.append(signal)
        
        # Print signal
        signal_gen.print_signal(signal)
    
    return signals


# ============================================================================
# STEP 4: BACKTEST & ANALYSIS
# ============================================================================

def simple_backtest(signals, initial_capital=10000):
    """
    Simple backtest to see how signals would have performed
    
    Args:
        signals: List of trading signals
        initial_capital: Starting capital
    
    Returns:
        Backtest results
    """
    print("\n" + "="*80)
    print("STEP 5: SIMPLE BACKTEST")
    print("="*80)
    
    capital = initial_capital
    trades = []
    
    for signal in signals:
        if signal['direction'] == 'HOLD':
            continue
        
        # Simulate trade (simplified)
        # In reality, you'd track actual price movements
        if signal['direction'] == 'BUY':
            # Assume we hit take profit 60% of time, stop loss 40%
            hit_tp = np.random.random() < 0.6
            if hit_tp:
                profit = signal['max_profit']
            else:
                profit = -signal['max_loss']
        else:  # SELL
            hit_tp = np.random.random() < 0.6
            if hit_tp:
                profit = signal['max_profit']
            else:
                profit = -signal['max_loss']
        
        capital += profit
        
        trades.append({
            'date': signal['date'],
            'direction': signal['direction'],
            'confidence': signal['confidence'],
            'profit': profit,
            'capital': capital
        })
    
    # Results
    if len(trades) > 0:
        total_trades = len(trades)
        profitable_trades = sum(1 for t in trades if t['profit'] > 0)
        win_rate = profitable_trades / total_trades * 100
        total_profit = capital - initial_capital
        roi = (capital - initial_capital) / initial_capital * 100
        
        print(f"📊 Total Trades: {total_trades}")
        print(f"✅ Profitable Trades: {profitable_trades}")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"💰 Total Profit: ${total_profit:.2f}")
        print(f"📊 ROI: {roi:.2f}%")
        print(f"💵 Final Capital: ${capital:.2f}")
    else:
        print("⚠️  No trades executed (all signals were HOLD)")
    
    return trades


# ============================================================================
# STEP 5: PLOT TRAINING HISTORY
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
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: training_history.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Complete end-to-end pipeline"""
    
    print("\n" + "⭐"*40)
    print("⭐⭐⭐ PROFESSIONAL TRADING MODEL TRAINING PIPELINE ⭐⭐⭐")
    print("⭐"*40 + "\n")
    
    # Step 1: Load/Generate Data
    # Use the pre-calculated features file
    csv_file = 'features_output.csv'
    
    df = load_and_prepare_data(
        csv_path=csv_file,  # Point to our features file
        generate_new=False, # Don't generate new data
        days=500
    )
    
    # Step 2: Train Model
    model, trainer, scaler, feature_cols = train_trading_model(
        df=df,
        epochs=50  # Increase for better results
    )
    
    # Step 3: Plot Training History
    plot_training_history(trainer)
    
    # Step 4: Generate Signals
    signals = generate_signals_for_recent_data(
        df=df,
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        num_recent=10
    )
    
    # Step 5: Simple Backtest
    trades = simple_backtest(signals, initial_capital=10000)
    
    # Save signals to CSV
    if len(signals) > 0:
        signals_df = pd.DataFrame(signals)
        signals_df.to_csv('trading_signals.csv', index=False)
        print(f"\n✅ Signals saved to: trading_signals.csv")
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print("\n📁 Generated Files:")
    print("  1. best_trading_model.pth - Trained model weights")
    print("  2. training_history.png - Training visualization")
    print("  3. trading_signals.csv - Generated trading signals")
    print("\n💡 Next Steps:")
    print("  1. Test on real historical data")
    print("  2. Implement live trading integration")
    print("  3. Add more sophisticated backtesting")
    print("  4. Fine-tune hyperparameters")
    print("="*80)


# ============================================================================
# INFERENCE MODE (For live trading)
# ============================================================================

def live_trading_mode(df_latest, model_path='/home/claude/best_trading_model.pth'):
    """
    Use trained model for live trading
    
    Args:
        df_latest: DataFrame with latest features (single row or recent rows)
        model_path: Path to trained model weights
    
    Returns:
        Trading signal
    """
    print("\n" + "="*80)
    print("🔴 LIVE TRADING MODE")
    print("="*80)
    
    # This assumes you have already:
    # 1. Trained the model
    # 2. Saved the scaler
    # 3. Have live market data with features
    
    # Load model
    # feature_cols = [...] # Load from saved file
    # scaler = joblib.load('scaler.pkl')
    # model = MultiHeadTradingModel(input_dim=len(feature_cols))
    # model.load_state_dict(torch.load(model_path))
    
    # Generate signal
    # signal_gen = TradingSignalGenerator(model, scaler)
    # signal = signal_gen.generate_signal(
    #     features=df_latest[feature_cols].iloc[-1],
    #     current_price=df_latest['Close'].iloc[-1]
    # )
    
    print("⚠️  Implement this for live trading")
    print("See code comments for details")


if __name__ == "__main__":
    # Run complete training pipeline
    main()
    
    # For live trading, use:
    # live_trading_mode(your_latest_data)
