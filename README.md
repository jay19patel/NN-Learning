# 🚀 QUICK START GUIDE - 5 Minutes to Your First Model

## ⚡ Option 1: Train Immediately (Synthetic Data)

```bash
# Step 1: Install dependencies
pip install torch pandas numpy scikit-learn matplotlib pandas_ta scipy --break-system-packages

# Step 2: Train the model (this will take 5-10 minutes)
python train_model.py
```

**That's it!** 🎉

This will:
1. ✅ Generate 500 days of synthetic OHLC data
2. ✅ Create 150+ technical features
3. ✅ Train the neural network
4. ✅ Generate 10 sample trading signals
5. ✅ Save the trained model

**Output files:**
- `best_trading_model.pth` - Your trained model
- `training_history.png` - Training visualization
- `trading_signals.csv` - Sample signals

---

## ⚡ Option 2: Use Your Own Data

### Step 1: Prepare your data

Your CSV should have these columns:
```
Date,Open,High,Low,Close,Volume
2024-01-01,150.23,152.45,149.80,151.20,1234567
2024-01-02,151.30,153.10,150.90,152.50,1345678
...
```

### Step 2: Modify train_model.py

```python
# In train_model.py, change this line:
df = load_and_prepare_data(
    csv_path='/path/to/your/data.csv',  # ← Add your CSV path here
    generate_new=False,  # ← Change to False
    days=500
)
```

### Step 3: Run training

```bash
python train_model.py
```

---

## 📊 Understanding the Output

### **Training Progress:**

```
Epoch [5/50] Train Loss: 0.2345 | Val Loss: 0.2567
Epoch [10/50] Train Loss: 0.1876 | Val Loss: 0.2123 ⭐ NEW BEST
Epoch [15/50] Train Loss: 0.1654 | Val Loss: 0.2098 ⭐ NEW BEST
...
```

**Good signs:**
- ✅ Both losses decreasing
- ✅ Val loss close to train loss (±20%)
- ✅ NEW BEST appearing regularly

**Bad signs:**
- ❌ Train loss decreasing but val loss increasing (overfitting)
- ❌ Both losses stuck (model not learning)

### **Trading Signal Example:**

```
================================================================================
⭐⭐⭐ TRADING SIGNAL ⭐⭐⭐
================================================================================
📅 Time: 2026-02-12 10:30:00
🎯 Direction: BUY (Confidence: 87.3%)
💰 Entry Price: $152.45
🛑 Stop Loss: $149.80
🎁 Take Profit: $158.20
📊 Position Size: 134.50 units
⚠️  Risk Score: 23.4%
📈 Expected Upside: 3.77%
📉 Expected Downside: -1.74%
⚖️  Risk-Reward: 2.17
💵 Max Loss: $356.52
💰 Max Profit: $773.07
--------------------------------------------------------------------------------
Buy: 87.3% | Hold: 8.2% | Sell: 4.5%
================================================================================
```

**What this means:**
- **BUY** with **87.3% confidence** → High conviction trade
- Risk **$356** to make **$773** → 2.17:1 risk-reward
- Position size: **134 shares** (larger because high confidence)
- Stop Loss: **$149.80** (automatic risk management)

---

## 🎯 What to Do with Signals

### **High Confidence (>70%) + Good Risk-Reward (>2)**
```python
if signal['confidence'] > 70 and signal['risk_reward_ratio'] > 2:
    print("✅ STRONG SIGNAL - Execute trade")
    print(f"Entry: {signal['entry_price']}")
    print(f"Stop: {signal['stop_loss']}")
    print(f"Target: {signal['take_profit']}")
    print(f"Size: {signal['position_size']} shares")
```

### **Medium Confidence (50-70%)**
```python
if 50 < signal['confidence'] <= 70:
    print("⚠️ MODERATE SIGNAL - Reduce position by 50%")
    reduced_size = signal['position_size'] * 0.5
```

### **Low Confidence (<50%)**
```python
if signal['confidence'] < 50:
    print("❌ SKIP - Confidence too low")
```

---

## 🔧 Common Issues & Solutions

### **Issue 1: Import Error**
```
ModuleNotFoundError: No module named 'pandas_ta'
```

**Solution:**
```bash
pip install pandas_ta --break-system-packages
```

### **Issue 2: CUDA Not Available**
```
Device: cpu
```

**This is OK!** Model will train on CPU (just slower).

To use GPU:
```bash
# Check if CUDA available
python -c "import torch; print(torch.cuda.is_available())"
```

### **Issue 3: Model Not Learning (Loss Stuck)**

**Solution 1:** Decrease learning rate
```python
# In TradingModelTrainer.__init__()
self.optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,  # ← Decrease from 0.001
    weight_decay=0.01
)
```

**Solution 2:** Increase epochs
```python
# In train_model.py
model, trainer, scaler, feature_cols = train_trading_model(
    df=df,
    epochs=100  # ← Increase from 50
)
```

### **Issue 4: All Predictions are HOLD**

**Solution:** Adjust direction labeling
```python
# In prepare_data_for_training() function
# Change threshold from 2 to 1.5
buy_condition = df_clean['upside_pct'] > abs(df_clean['downside_pct']) * 1.5  # ← Changed
sell_condition = abs(df_clean['downside_pct']) > df_clean['upside_pct'] * 1.5  # ← Changed
```

---

## 📈 Improving Your Model

### **1. More Data = Better Model**
- Minimum: 500 days
- Recommended: 2-3 years
- Optimal: 5+ years with different market conditions

### **2. Better Features**
```python
# Add your own custom features
def add_custom_features(df):
    df['my_indicator'] = ...
    return df

# In data_prepare_organized.py
df = add_custom_features(df)
```

### **3. Hyperparameter Tuning**

Try different configurations:

**Configuration 1: Bigger Model**
```python
model = MultiHeadTradingModel(
    input_dim=input_dim,
    hidden_dim=512,      # ← Increased
    num_heads=16,        # ← Increased
    num_layers=4,        # ← Increased
    dropout=0.2
)
```

**Configuration 2: Prevent Overfitting**
```python
model = MultiHeadTradingModel(
    input_dim=input_dim,
    hidden_dim=128,      # ← Decreased
    num_heads=4,         # ← Decreased
    num_layers=2,        # ← Decreased
    dropout=0.4          # ← Increased
)
```

### **4. Ensemble Methods**

Train 3-5 models and average predictions:

```python
# Train multiple models
models = []
for seed in [42, 123, 456, 789, 1011]:
    torch.manual_seed(seed)
    model = train_model(...)
    models.append(model)

# Average predictions
signals = [gen.generate_signal(...) for gen in signal_generators]
avg_confidence = np.mean([s['confidence'] for s in signals])
```

---

## 📊 Monitoring Live Performance

### **Track These Metrics:**

1. **Win Rate vs Confidence**
```python
# If model says 80% confident, win rate should be ~80%
actual_win_rate = wins / total_trades
expected_confidence = average_confidence

if abs(actual_win_rate - expected_confidence) > 0.1:
    print("⚠️ Model needs recalibration")
```

2. **Average Risk-Reward**
```python
avg_rr = sum([signal['risk_reward_ratio'] for signal in signals]) / len(signals)
print(f"Average R:R: {avg_rr:.2f}")

# Should be > 1.5 for profitable trading
```

3. **Sharpe Ratio**
```python
returns = [trade['profit'] for trade in trades]
sharpe = np.mean(returns) / (np.std(returns) + 1e-6)

# > 1.0 is good
# > 2.0 is excellent
```

---

## 🎯 Next Steps After Training

### **Week 1: Paper Trading**
- Generate signals daily
- Don't execute, just track
- Compare predictions vs actual outcomes

### **Week 2-4: Small Capital**
- Start with $500-1000
- Execute only high confidence (>80%) signals
- Maximum 1-2% risk per trade

### **Month 2-3: Scale Up**
- If profitable, increase capital
- Track all metrics
- Retrain model monthly

### **Month 4+: Optimize**
- Analyze which signals work best
- Remove bad features
- Add new features
- Try different model architectures

---

## ⚡ ONE-LINE COMMANDS

**Complete training:**
```bash
python train_model.py
```

**Generate signal for latest data:**
```python
python -c "from train_model import generate_signals_for_recent_data; generate_signals_for_recent_data(df, model, scaler, feature_cols, num_recent=1)"
```

**Backtest:**
```python
python -c "from train_model import simple_backtest; simple_backtest(signals)"
```

---

## 🎓 Learning Path

**Beginner:**
1. ✅ Run train_model.py
2. ✅ Understand signals
3. ✅ Paper trade 1 month

**Intermediate:**
1. ✅ Train on real data
2. ✅ Adjust hyperparameters
3. ✅ Add custom features
4. ✅ Live trade small capital

**Advanced:**
1. ✅ Ensemble models
2. ✅ Multi-timeframe analysis
3. ✅ Walk-forward optimization
4. ✅ Integration with broker API

---

## 📞 Help & Support

**Common Questions:**

**Q: How long to train?**
A: 5-10 minutes on CPU, 1-2 minutes on GPU (for 50 epochs)

**Q: How much data needed?**
A: Minimum 500 days, recommended 1000+ days

**Q: Can I use for stocks/crypto/forex?**
A: Yes! Just provide OHLCV data in same format

**Q: How often to retrain?**
A: Every 1-3 months, or when performance degrades

**Q: What win rate to expect?**
A: 55-65% is realistic, >70% is excellent

---

## 🚀 READY TO START?

```bash
# Just run this:
python train_model.py
```

**Good luck! 🎯📈💰**
