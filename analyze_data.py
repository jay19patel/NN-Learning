import pandas as pd
from data_prepare_organized import fetch_data, create_full_feature_set, add_target_labels

def analyze():
    print("Fetching data...")
    df = fetch_data(total_days=500)
    print("Creating features...")
    df = create_full_feature_set(df, lookahead=10)
    
    print("\nClass Distribution:")
    dist = df['direction_label'].value_counts(normalize=True) * 100
    counts = df['direction_label'].value_counts()
    
    label_map = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
    
    for label, name in label_map.items():
        if label in dist:
            print(f"{name} ({label}): {dist[label]:.2f}%  (Count: {counts[label]})")
            
if __name__ == "__main__":
    analyze()
