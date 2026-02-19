# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering for Trading NN Model
Organized by feature categories with detailed comments
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import random
from datetime import datetime, timedelta
import scipy.stats as stats


# ============================================================================
# DATA GENERATION
# ============================================================================

import requests
import time
import os

def fetch_data(symbol="ADAUSD", total_days=100, interval="15m"):
    """
    Fetch OHLC data from Delta Exchange API with CSV caching.
    If a local CSV exists for the symbol/interval, load it.
    Otherwise, fetch from API, save to CSV, and return.
    """
    
    # Define cache filename
    filename = f"data_{symbol}_{interval}.csv"
    
    # Check if CSV exists
    if os.path.exists(filename):
        print(f"Loading data from local CSV: {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        # Ensure DateTime index is tz-aware if not already (read_csv might lose tz info or keep it as string)
        if df.index.tz is None:
             # Assuming the CSV was saved with UTC, we localize to UTC then convert to Kolkata
             # However, if we saved it after converting to Kolkata, it might be naive but in Kolkata time.
             # Let's standardize: The inputs are UTC. The user code converts to Asia/Kolkata.
             # When reading back, we should check.
             # Simplest approach for consistency: adhere to what was saved.
             # If the saved CSV has a named index "DateTime", read_csv(..., index_col=0) works.
             pass
        return df

    print(f"Local CSV not found. Fetching data from API for {symbol}...")
    
    api_url = "https://api.india.delta.exchange/v2/history/candles"
    headers = {'Accept': 'application/json'}
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=total_days)
    
    date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")
    all_dfs = []

    for i in range(len(date_ranges)):
        chunk_start = date_ranges[i]
        chunk_end = date_ranges[i + 1] if i + 1 < len(date_ranges) else end_date

        start_ts = int(chunk_start.timestamp())
        end_ts = int(chunk_end.timestamp())

        params = {
            "resolution": interval,
            "symbol": symbol,
            "start": str(start_ts),
            "end": str(end_ts)
        }

        print(f"Fetching: {chunk_start.date()} → {chunk_end.date()}")

        for attempt in range(3):
            try:
                response = requests.get(api_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("result"):
                        rows = []
                        for c in data["result"]:
                            rows.append({
                                "time": c["time"],
                                "Open": float(c["open"]),
                                "High": float(c["high"]),
                                "Low": float(c["low"]),
                                "Close": float(c["close"]),
                                "Volume": float(c["volume"] or 0)
                            })
                        df_chunk = pd.DataFrame(rows)
                        # Convert time to datetime objects
                        df_chunk["DateTime"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
                        # Convert to target timezone
                        df_chunk["DateTime"] = df_chunk["DateTime"].dt.tz_convert("Asia/Kolkata")
                        df_chunk.set_index("DateTime", inplace=True)
                        
                        all_dfs.append(df_chunk)
                        break
                time.sleep(1)
            except Exception as e:
                print(f"Retry error on attempt {attempt+1}: {e}")
                time.sleep(1)

    if not all_dfs:
        print("No data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    
    # Save to CSV for future use
    df.to_csv(filename)
    print(f"Data saved to {filename}")
    
    return df


# ============================================================================
# ⭐ BASIC PRICE & VOLUME FEATURES
# ============================================================================

def add_basic_features(df):
    """Add fundamental price and volume based features"""
    
    # Price returns - Measures price changes in logarithmic scale for better normalization
    df['close_return'] = np.log(df['Close'] / df['Close'].shift(1))  # Daily return
    df['return_1'] = df['Close'].pct_change()  # Simple percentage return for 1 day
    df['return_5'] = df['Close'].pct_change(5)  # 5-day return
    df['return_10'] = df['Close'].pct_change(10)  # 10-day return
    df['return_20'] = df['Close'].pct_change(20)  # 20-day return
    
    # Intraday price relationships - Captures intraday price movements relative to open
    df['open_close_return'] = np.log(df['Close'] / df['Open'])  # Overall daily movement
    df['high_open_return'] = np.log(df['High'] / df['Open'])  # Maximum upside during day
    df['low_open_return'] = np.log(df['Low'] / df['Open'])  # Maximum downside during day
    
    # Volume features - Volume patterns indicate buying/selling pressure
    df['log_volume'] = np.log(df['Volume'] + 1)  # Log-normalized volume
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()  # Volume relative to average
    df['volume_std'] = df['Volume'].rolling(20).std()  # Volume volatility
    
    # Price range features - Measures daily price spread
    df['daily_range'] = (df['High'] - df['Low']) / df['Close']  # Daily range as % of close
    df['range_pct'] = ((df['High'] - df['Low']) / df['Open']) * 100  # Range as % of open
    
    return df


# ============================================================================
# ⭐ MOVING AVERAGES & TREND INDICATORS (pandas_ta)
# ============================================================================

def add_moving_averages(df):
    """Add various moving averages to identify trends"""
    
    # EMAs - Exponential Moving Average gives more weight to recent prices
    for period in [5, 10, 20, 50, 100]:
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period).bfill()
    
    # SMAs - Simple Moving Average for longer-term trends
    for period in [20, 50, 100, 200]:
        df[f'SMA_{period}'] = ta.sma(df['Close'], length=period).bfill()
    
    # Price position relative to EMAs - Shows if price is above/below key levels
    df['price_to_ema_5'] = (df['Close'] - df['EMA_5']) / df['EMA_5']
    df['price_to_ema_20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['price_to_sma_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    
    # EMA crossovers - Classic trend signals
    df['ema_5_10_cross'] = df['EMA_5'] - df['EMA_10']  # Short-term trend
    df['ema_10_20_cross'] = df['EMA_10'] - df['EMA_20']  # Medium-term trend
    
    # VWAP - Volume Weighted Average Price shows institutional entry/exit zones
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume']).bfill()
    df['price_to_vwap'] = (df['Close'] - df['VWAP']) / df['VWAP']  # Distance from VWAP
    
    return df


# ============================================================================
# ⭐ MOMENTUM INDICATORS (pandas_ta)
# ============================================================================

def add_momentum_indicators(df):
    """Add momentum oscillators to identify overbought/oversold conditions"""
    
    # RSI - Relative Strength Index identifies overbought (>70) and oversold (<30) conditions
    for period in [7, 14, 21]:
        df[f'RSI_{period}'] = ta.rsi(df['Close'], length=period).bfill()
    
    # MACD - Moving Average Convergence Divergence shows trend strength and reversals
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9'].bfill()  # MACD line
    df['MACD_signal'] = macd['MACDs_12_26_9'].bfill()  # Signal line
    df['MACD_hist'] = macd['MACDh_12_26_9'].bfill()  # Histogram shows momentum
    
    # Stochastic Oscillator - Compares close price to price range over period
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    df['Stoch_K'] = stoch['STOCHk_14_3_3'].bfill()  # Fast line
    df['Stoch_D'] = stoch['STOCHd_14_3_3'].bfill()  # Slow line (signal)
    
    # CCI - Commodity Channel Index measures variation from statistical mean
    df['CCI_20'] = ta.cci(df['High'], df['Low'], df['Close'], length=20).bfill()
    
    # Williams %R - Shows overbought/oversold with inverse scale
    df['WilliamsR_14'] = ta.willr(df['High'], df['Low'], df['Close'], length=14).bfill()
    
    # ROC - Rate of Change measures momentum
    df['ROC_10'] = ta.roc(df['Close'], length=10).bfill()
    df['ROC_20'] = ta.roc(df['Close'], length=20).bfill()
    
    return df


# ============================================================================
# ⭐ VOLATILITY INDICATORS (pandas_ta)
# ============================================================================

def add_volatility_indicators(df):
    """Add volatility measures to gauge market risk and movement potential"""
    
    # ATR - Average True Range measures market volatility (higher = more volatile)
    for period in [7, 14, 21]:
        df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period).bfill()
    
    # ATR as percentage of price - Normalized volatility measure
    df['ATR_pct'] = (df['ATR_14'] / df['Close']) * 100
    
    # Bollinger Bands - Shows price deviation from moving average
    bbands = ta.bbands(df['Close'], length=20, std=2)
    # print(f"DEBUG: bbands columns: {bbands.columns}") # Debugging
    # Common pandas-ta column names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
    # In some versions it might be different. Let's inspect or use iloc if needed, 
    # but inspecting is safer. 
    if bbands is not None:
         df['BB_upper'] = bbands.iloc[:, 2].bfill()  # Upper band (usually 3rd col)
         df['BB_middle'] = bbands.iloc[:, 1].bfill()  # Middle band (usually 2nd col)
         df['BB_lower'] = bbands.iloc[:, 0].bfill()  # Lower band (usually 1st col)
    # df['BB_upper'] = bbands['BBU_20_2.0'].bfill()  # Upper band
    # df['BB_middle'] = bbands['BBM_20_2.0'].bfill()  # Middle band (SMA)
    # df['BB_lower'] = bbands['BBL_20_2.0'].bfill()  # Lower band
    
    df['BB_width'] = ((df['BB_upper'] - df['BB_lower']) / df['BB_middle']) * 100  # Band width
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])  # Price position in bands
    
    # Keltner Channels - Similar to Bollinger but uses ATR
    kc = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2)
    # df['KC_upper'] = kc['KCUe_20_2'].bfill()
    # df['KC_lower'] = kc['KCLe_20_2'].bfill()
    if kc is not None:
        df['KC_upper'] = kc.iloc[:, 2].bfill()
        df['KC_lower'] = kc.iloc[:, 0].bfill()
    
    # Historical Volatility - Standard deviation of returns
    df['volatility_10'] = df['Close'].pct_change().rolling(10).std()  # 10-day volatility
    df['volatility_20'] = df['Close'].pct_change().rolling(20).std()  # 20-day volatility
    df['volatility_50'] = df['Close'].pct_change().rolling(50).std()  # 50-day volatility
    
    return df


# ============================================================================
# ⭐ TREND STRENGTH INDICATORS (pandas_ta)
# ============================================================================

def add_trend_indicators(df):
    """Add indicators that measure trend strength and direction"""
    
    # ADX - Average Directional Index measures trend strength (>25 = strong trend)
    for period in [14, 20]:
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=period)
        df[f'ADX_{period}'] = adx_df[f'ADX_{period}'].bfill()  # Trend strength
        df[f'DMP_{period}'] = adx_df[f'DMP_{period}'].bfill()  # Positive directional movement
        df[f'DMN_{period}'] = adx_df[f'DMN_{period}'].bfill()  # Negative directional movement
    
    # Directional bias - Difference between positive and negative movement
    df['directional_bias'] = df['DMP_14'] - df['DMN_14']
    
    # Supertrend - Trend following indicator (above = downtrend, below = uptrend)
    supertrend = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    # Check columns to be safe, likely SUPERT_10_3 without .0
    # Dictionary lookup or checking first column might be safer
    df['supertrend'] = supertrend["SUPERT_10_3"].bfill()
    df['supertrend_direction'] = supertrend["SUPERTd_10_3"].bfill()  # 1 = uptrend, -1 = downtrend
    
    # Supertrend flip detection - Identifies trend reversals
    df['st_flip'] = df['supertrend_direction'].diff().abs()  # 2 when trend flips
    df['bars_since_flip'] = df.groupby((df['st_flip'] == 2).cumsum()).cumcount()  # Bars since last flip
    
    # Aroon - Measures time since highest high and lowest low
    aroon = ta.aroon(df['High'], df['Low'], length=25)
    df['aroon_up'] = aroon['AROONU_25'].bfill()  # Uptrend strength
    df['aroon_down'] = aroon['AROOND_25'].bfill()  # Downtrend strength
    df['aroon_oscillator'] = df['aroon_up'] - df['aroon_down']  # Net trend direction
    
    return df


# ============================================================================
# ⭐ VOLUME INDICATORS (pandas_ta)
# ============================================================================

def add_volume_indicators(df):
    """Add volume-based indicators to confirm price movements"""
    
    # OBV - On Balance Volume accumulates volume based on price direction
    df['OBV'] = ta.obv(df['Close'], df['Volume']).bfill()
    df['OBV_ema'] = ta.ema(df['OBV'], length=20).bfill()  # Smoothed OBV
    
    # AD - Accumulation/Distribution measures money flow
    df['AD'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume']).bfill()
    
    # CMF - Chaikin Money Flow measures buying/selling pressure
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20).bfill()
    
    # MFI - Money Flow Index is RSI weighted by volume
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14).bfill()
    
    # Volume-Price Trend
    df['VPT'] = ta.pvt(df['Close'], df['Volume']).bfill()
    
    return df


# ============================================================================
# ⭐⭐ ADVANCED CANDLE PATTERN FEATURES
# ============================================================================

def add_candle_features(df):
    """Add candlestick pattern recognition features"""
    
    # Basic candle anatomy
    df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']  # Body size shows conviction
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)  # Upper shadow
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']  # Lower shadow
    df['total_wick'] = df['upper_wick'] + df['lower_wick']  # Total shadow
    
    # Wick analysis - Rejection zones and smart money activity
    df['wick_imbalance'] = (df['upper_wick'] - df['lower_wick']) / df['Close']  # Which side rejected
    df['wick_to_body'] = (df['total_wick'] / (abs(df['Close'] - df['Open']) + 0.0001))  # Indecision ratio
    
    # Candle position within range
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)  # 0=low, 1=high
    
    # Candle color and strength
    df['is_bullish'] = (df['Close'] > df['Open']).astype(int)  # 1 if green candle
    df['candle_strength'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.0001)  # Range coverage
    
    # Gap detection
    df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)  # Gap up from previous close
    df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)  # Gap down from previous close
    df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)  # Gap magnitude
    
    return df


# ============================================================================
# ⭐⭐ ADVANCED STATISTICAL FEATURES
# ============================================================================

def add_statistical_features(df):
    """Add statistical measures for distribution and anomaly detection"""
    
    # Z-Score - Measures standard deviations from mean (identifies extremes)
    for period in [10, 20, 50]:
        df[f'zscore_{period}'] = ((df['Close'] - df['Close'].rolling(period).mean()) / 
                                   (df['Close'].rolling(period).std() + 0.0001))
    
    # Skewness - Measures asymmetry of return distribution
    df['skew_20'] = df['return_1'].rolling(20).skew()
    
    # Kurtosis - Measures tail risk (fat tails = higher kurtosis)
    df['kurt_20'] = df['return_1'].rolling(20).kurt()
    
    # Percentile rank - Where current price stands in recent range
    df['percentile_rank_20'] = df['Close'].rolling(20).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100 if len(x) > 0 else 0.5
    )
    
    return df


# ============================================================================
# ⭐⭐⭐ ADVANCED VOLATILITY MICROSTRUCTURE
# ============================================================================

def add_advanced_volatility_features(df):
    """Add sophisticated volatility and market microstructure features"""
    
    # Create a DataFrame to hold new features
    # (Note: Using a dict then DataFrame is often faster, but direct assignment to new DF is fine too)
    new_features = pd.DataFrame(index=df.index)
    
    # Realized Variance - Squared returns capture large moves
    new_features['realized_var_20'] = (df['return_1'] ** 2).rolling(20).sum()
    
    # Bipower Variation - Separates jumps from continuous volatility
    new_features['bipower_var'] = (abs(df['return_1']) * abs(df['return_1'].shift())).rolling(20).sum()
    
    # Jump Detection - Identifies sudden institutional activity or news
    new_features['jump_strength'] = new_features['realized_var_20'] - new_features['bipower_var']
    
    # Volatility clustering - Volatility of volatility
    new_features['vol_cluster'] = df['volatility_10'].rolling(20).std()
    
    # Volatility regime - High vs low volatility environment
    new_features['vol_regime'] = (df['volatility_10'] > df['volatility_10'].rolling(50).mean()).astype(int)
    
    # Range metrics
    new_features['range_compression'] = ((df['High'] - df['Low']).rolling(10).mean() / 
                                (df['High'] - df['Low']).rolling(50).mean())  # Detects consolidation before breakout
    
    new_features['range_velocity'] = (df['High'] - df['Low']).pct_change()  # Speed of range expansion
    
    # Fractal dimension proxy - Structured vs chaotic movement
    new_features['fractal_proxy'] = df['ATR_pct'] / (df['volatility_10'] + 0.0001)
    
    # Volatility mean reversion
    new_features['vol_reversion_speed'] = (df['volatility_10'] - df['volatility_10'].shift(10)) / 10
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# ⭐⭐⭐ ADVANCED TREND & MOMENTUM MICROSTRUCTURE
# ============================================================================

def add_advanced_trend_features(df):
    """Add sophisticated trend quality and momentum features"""
    new_features = pd.DataFrame(index=df.index)

    # Trend efficiency - Clean trend vs noisy chop
    new_features['efficiency_ratio'] = (abs(df['Close'] - df['Close'].shift(10)) / 
                               (df['High'].rolling(10).max() - df['Low'].rolling(10).min() + 0.0001))
    
    # Trend persistence - Consecutive moves in same direction
    new_features['trend_persistence'] = np.sign(df['return_1']).rolling(10).sum()
    
    # Trend smoothness - Institutional vs retail movement
    new_features['trend_smoothness'] = (abs(df['Close'] - df['Close'].shift(20)) / 
                               (df['return_1'].rolling(20).std() + 0.0001))
    
    # Path curvature - Straight vs zigzag movement
    new_features['path_curvature'] = df['return_1'].diff().abs().rolling(10).mean()
    
    # Trend strength derived from Supertrend
    new_features['trend_strength'] = abs(df['Close'] - df['supertrend']) / df['Close']  # Distance from trend line
    new_features['trend_acceleration'] = new_features['trend_strength'].diff()  # Is trend strengthening?
    
    # Directional entropy - Random vs structured movement
    new_features['dir_entropy'] = df['return_1'].rolling(20).apply(
        lambda x: -np.mean(np.sign(x) * np.log(np.abs(np.sign(x)) + 1e-6))
    )
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# ⭐⭐⭐ INFORMATION THEORY FEATURES (Very Advanced)
# ============================================================================

def add_information_theory_features(df):
    """Add entropy and information-based features for pattern recognition"""
    new_features = pd.DataFrame(index=df.index)

    # Price entropy - Low = structured move, High = noise
    new_features['price_entropy'] = df['return_1'].rolling(20).apply(
        lambda x: stats.entropy(np.histogram(x, bins=5)[0] + 1) if len(x) > 0 else 0,
        raw=False
    )
    
    # Surprise measure - Unexpected moves
    new_features['surprise'] = ((df['return_1'] - df['return_1'].rolling(20).mean()) / 
                      (df['return_1'].rolling(20).std() + 1e-6))
    
    # Shock elasticity - How market absorbs/amplifies shocks
    new_features['shock_elasticity'] = df['return_1'].abs() / (df['volatility_10'] + 1e-6)
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# ⭐⭐⭐ MARKET MICROSTRUCTURE & LIQUIDITY
# ============================================================================

def add_microstructure_features(df):
    """Add features related to market structure and liquidity"""
    new_features = pd.DataFrame(index=df.index)

    # Buying pressure - Where candle closes in its range
    new_features['buy_pressure'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.0001)
    
    # Slippage proxy - Thin vs deep liquidity
    new_features['slippage_proxy'] = (df['High'] - df['Low']) / df['Close'].rolling(10).mean()
    
    # Stop hunt detection - Wide range relative to normal
    new_features['stop_hunt_proxy'] = (df['High'] - df['Low']) / (df['ATR_14'] + 0.0001)
    
    # Amihud Illiquidity - Price impact per unit volume
    new_features['amihud_illiquidity'] = abs(df['return_1']) / (df['Volume'] + 1)
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# ⭐⭐ RISK-REWARD FEATURES
# ============================================================================

def add_risk_reward_features(df, lookahead=10):
    """Add forward-looking risk/reward metrics for target creation"""
    new_features = pd.DataFrame(index=df.index)

    # Future price movements (for training targets)
    future_max_high = df['High'].shift(-1).rolling(lookahead).max()
    future_min_low = df['Low'].shift(-1).rolling(lookahead).min()
    future_close = df['Close'].shift(-lookahead)
    
    # Potential upside/downside
    new_features['upside_pct'] = ((future_max_high - df['Close']) / df['Close']) * 100
    new_features['downside_pct'] = ((future_min_low - df['Close']) / df['Close']) * 100
    
    # Worst drawdown during holding period
    new_features['future_drawdown_pct'] = ((future_min_low - future_max_high) / future_max_high) * 100
    
    # Risk-reward ratios
    new_features['reward_risk_ratio'] = new_features['upside_pct'] / (abs(new_features['downside_pct']) + 0.0001)
    new_features['edge_ratio'] = new_features['upside_pct'] / (abs(new_features['downside_pct']) + 1e-6)  # Similar but with different epsilon
    
    # Return vs pain ratio
    new_features['pain_ratio'] = df['return_10'] / (abs(new_features['future_drawdown_pct']) + 0.0001)
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# ⭐ INTERACTION FEATURES
# ============================================================================

def add_interaction_features(df):
    """Add interaction features combining multiple indicators"""
    new_features = pd.DataFrame(index=df.index)

    # Momentum + Volatility interactions
    new_features['rsi_vol'] = df['RSI_14'] * df['volatility_10']  # Momentum in volatile conditions
    new_features['rsi_atr'] = df['RSI_14'] * df['ATR_pct']  # RSI weighted by volatility
    
    # Trend + Volume interactions
    new_features['trend_volume'] = df['trend_strength'] * df['volume_ratio']  # Strong trends with volume
    new_features['adx_volume'] = df['ADX_14'] * df['volume_ratio']  # Trend strength with volume
    
    # Price position interactions
    new_features['bb_rsi'] = df['BB_position'] * df['RSI_14']  # Combined positioning
    
    # Volume-volatility relationship
    new_features['vol_atr_ratio'] = df['volume_ratio'] / (df['ATR_pct'] + 0.0001)  # Volume vs volatility
    
    return pd.concat([df, new_features], axis=1)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# ============================================================================
# ⭐ TARGET LABELS
# ============================================================================

def add_target_labels(df):
    """
    Add classification labels based on risk/reward and minimum return.
    
    Labels:
    - 0: BUY (High reward/risk ratio + sufficient upside)
    - 1: HOLD (Low conviction or chop)
    - 2: SELL (High reward/risk ratio + sufficient downside)
    """
    
    # Default to HOLD
    df['direction_label'] = 1
    
    # Parameters
    min_return = 0.8  # Minimum 0.8% move required (filters out noise)
    risk_reward_ratio = 1.0  # Reward must be 1.0x risk (Balanced)
    
    # Ensure upside/downside exist
    if 'upside_pct' in df.columns and 'downside_pct' in df.columns:
        # BUY Condition:
        # 1. Upside > Downside * Ratio
        # 2. Upside >= Minimum Return
        buy_condition = (
            (df['upside_pct'] > abs(df['downside_pct']) * risk_reward_ratio) & 
            (df['upside_pct'] >= min_return)
        )
        
        # SELL Condition:
        # 1. Downside > Upside * Ratio (absolute values)
        # 2. Downside >= Minimum Return (absolute value)
        sell_condition = (
            (abs(df['downside_pct']) > df['upside_pct'] * risk_reward_ratio) & 
            (abs(df['downside_pct']) >= min_return)
        )
        
        df.loc[buy_condition, 'direction_label'] = 0  # Buy
        df.loc[sell_condition, 'direction_label'] = 2  # Sell
        
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def create_full_feature_set(df, lookahead=10):
    """
    Complete feature engineering pipeline
    
    Args:
        df: DataFrame with OHLC data
        lookahead: Days to look ahead for target variables
    
    Returns:
        DataFrame with all features
    """
    print("Adding basic features...")
    df = add_basic_features(df)
    
    print("Adding moving averages...")
    df = add_moving_averages(df)
    
    print("Adding momentum indicators...")
    df = add_momentum_indicators(df)
    
    print("Adding volatility indicators...")
    df = add_volatility_indicators(df)
    
    print("Adding trend indicators...")
    df = add_trend_indicators(df)
    
    print("Adding volume indicators...")
    df = add_volume_indicators(df)
    
    print("Adding candle features...")
    df = add_candle_features(df)
    
    print("Adding statistical features...")
    df = add_statistical_features(df)
    
    print("⭐ Adding advanced volatility features...")
    df = add_advanced_volatility_features(df)
    
    print("⭐ Adding advanced trend features...")
    df = add_advanced_trend_features(df)
    
    print("⭐ Adding information theory features...")
    df = add_information_theory_features(df)
    
    print("⭐ Adding microstructure features...")
    df = add_microstructure_features(df)
    
    print("⭐ Adding risk-reward features...")
    df = add_risk_reward_features(df, lookahead)
    
    print("Adding interaction features...")
    df = add_interaction_features(df)
    
    print("Adding target labels...")
    df = add_target_labels(df)
    
    # Fill any remaining NaN values
    df = df.bfill().ffill()
    
    print(f"\n✅ Feature engineering complete!")
    print(f"Total features: {len(df.columns)}")
    print(f"Shape: {df.shape}")
    
    return df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Generate sample data
    print("Fetching OHLC data...")
    # Using fetch_data instead of generate_ohlc_data
    df = fetch_data(symbol="ADAUSD", total_days=100, interval="15m")
    
    # Create all features
    df_with_features = create_full_feature_set(df, lookahead=10)
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE OF FEATURES:")
    print("="*80)
    print(df_with_features.tail(10))
    
    # Show feature categories
    print("\n" + "="*80)
    print("FEATURE SUMMARY BY CATEGORY:")
    print("="*80)
    
    feature_categories = {
        'Basic': ['return', 'volume', 'range', 'log'],
        'Moving Averages': ['EMA', 'SMA', 'VWAP'],
        'Momentum': ['RSI', 'MACD', 'Stoch', 'CCI', 'Williams', 'ROC'],
        'Volatility': ['ATR', 'BB_', 'KC_', 'volatility'],
        'Trend': ['ADX', 'DMP', 'DMN', 'supertrend', 'aroon'],
        'Volume': ['OBV', 'AD', 'CMF', 'MFI', 'VPT'],
        'Candle': ['body', 'wick', 'close_position', 'candle', 'gap'],
        'Statistical': ['zscore', 'skew', 'kurt', 'percentile'],
        '⭐ Advanced Vol': ['realized_var', 'bipower', 'jump', 'vol_cluster', 'fractal'],
        '⭐ Advanced Trend': ['efficiency', 'persistence', 'smoothness', 'curvature'],
        '⭐ Information': ['entropy', 'surprise', 'shock'],
        '⭐ Microstructure': ['pressure', 'slippage', 'stop_hunt', 'amihud'],
        '⭐ Risk/Reward': ['upside', 'downside', 'drawdown', 'ratio', 'pain'],
        'Interactions': ['_vol', '_atr', '_volume', '_rsi']
    }
    
    for category, keywords in feature_categories.items():
        matching_cols = [col for col in df_with_features.columns 
                        if any(kw in col for kw in keywords)]
        if matching_cols:
            print(f"\n{category} ({len(matching_cols)} features):")
            print(f"  {', '.join(matching_cols[:10])}")
            if len(matching_cols) > 10:
                print(f"  ... and {len(matching_cols) - 10} more")
    
    # Save to CSV
    # output_file = 'features_output.csv'
    # df_with_features.to_csv(output_file)
    # print(f"\n✅ Features saved to: {output_file}")
    print("\n✅ Features generation complete (not saved to CSV)")
