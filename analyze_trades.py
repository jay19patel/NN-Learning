import pandas as pd

df = pd.read_csv('model_outputs/backtest_trades.csv')
print("Total Trades:", len(df))
print("Win Rate:", (df['pnl'] > 0).mean())
print("Total PnL:", df['pnl'].sum())
print("Average Profit on Win:", df[df['pnl'] > 0]['pnl'].mean())
print("Average Loss on Loss:", df[df['pnl'] <= 0]['pnl'].mean())
print("\nExit Reasons:")
print(df['reason'].value_counts())
