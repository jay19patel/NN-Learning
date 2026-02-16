
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import copy

from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import fetch_data, prepare_features, StockDataset # Modified import
from model import StockTCN

# ================= CONFIGURATION =================
# Training Hyperparameters
TOTAL_DAYS = 100      # Increased to ensure enough data
SEQ_LENGTH = 30      # Increased sequence length for TCN (better context)
NN_LAYERS = 3        # Increased layers for deeper abstraction
EPOCHS = 100         # Max epochs (will stop early)
LEARNING_RATE = 0.0005 # Slower learning rate for stability
WEIGHT_DECAY = 1e-4  # Regularization
PATIENCE = 15        # Early stopping patience
BATCH_SIZE = 32      # Batch size (implicit in full batch for now, but good to note)

# Output Configuration
RESULTS_FILE = "results.csv"
GRAPHS_DIR = "graphs"
MODEL_SAVE_PATH = "best_model.pth"

# ===============================================

def prepare_global_data(total_days):
    print(f"\n" + "="*40)
    print(f"FETCHING DATA (Days={total_days})")
    print("="*40)
    
    # 1. Fetch and process data
    df = fetch_data(total_days=total_days)
    if df.empty:
        print("Error: No data fetched.")
        return None, None, None
        
    raw_shape = df.shape
    df, feature_cols = prepare_features(df)
    features_shape = df.shape
    
    # --- DATA DIAGNOSTICS ---
    print(f"\n--- DATA DIAGNOSTICS ---")
    print(f"1. Raw Data Shape (from API): {raw_shape}")
    print(f"2. Shape after Features:      {features_shape}")
    
    # ActionArea Distribution
    counts = df['ActionArea'].value_counts().sort_index()
    print(f"3. Final Class Distribution:")
    print(f"   - Hold (0): {counts.get(0, 0)}")
    print(f"   - Buy (1):  {counts.get(1, 0)}")
    print(f"   - Sell (2): {counts.get(2, 0)}")
    print(f"------------------------\n")

    if df.empty:
        print("Error: DataFrame is empty after cleaning.")
        return None, None, None
        
    X_raw = df[feature_cols].values
    y_raw = df["ActionArea"].values.astype(int)
    
    return X_raw, y_raw, feature_cols

def plot_training_metrics(train_loss, val_loss, train_acc, val_acc, save_dir):
    """
    Plots Training and Validation Loss & Accuracy on a single chart with dual y-axes.
    """
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss (Left Axis)
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    l1 = ax1.plot(epochs, train_loss, label='Train Loss', color=color, linestyle='--')
    l2 = ax1.plot(epochs, val_loss, label='Val Loss', color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for Accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    l3 = ax2.plot(epochs, train_acc, label='Train Acc', color=color, linestyle='--')
    l4 = ax2.plot(epochs, val_acc, label='Val Acc', color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    # Unified Legend
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    
    plt.title('Training & Validation Metrics')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved unified metrics chart to {save_path}")

def plot_confusion_matrix(y_true, y_pred, title, filename, save_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Hold', 'Buy', 'Sell'], 
                yticklabels=['Hold', 'Buy', 'Sell'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"Saved {filename}")

def train_and_evaluate(X_raw, y_raw, feature_cols):
    if not os.path.exists(GRAPHS_DIR):
        os.makedirs(GRAPHS_DIR)

    start_time = time.time()
    print(f"\n" + "="*40)
    print(f"STARTING TRAINING PIPELINE")
    print(f"Config: Seq={SEQ_LENGTH}, Layers={NN_LAYERS}, Epochs={EPOCHS}")
    print("="*40)
    
    # --- 1. Data Splitting (60% Train, 20% Val, 20% Test) ---
    total_samples = len(X_raw)
    train_split = int(total_samples * 0.60)
    val_split = int(total_samples * 0.80)
    
    X_train_raw = X_raw[:train_split]
    y_train_raw = y_raw[:train_split]
    
    X_val_raw = X_raw[train_split:val_split]
    y_val_raw = y_raw[train_split:val_split]
    
    X_test_raw = X_raw[val_split:]
    y_test_raw = y_raw[val_split:]
    
    print(f"Data Split:")
    print(f"   Train: {len(X_train_raw)} samples")
    print(f"   Val:   {len(X_val_raw)} samples")
    print(f"   Test:  {len(X_test_raw)} samples")
    
    # --- 2. Scaling (Fit on Train, Apply to All) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # --- 3. Create Datasets & DataLoaders ---
    # Instantiate custom Datasets
    train_dataset = StockDataset(X_train_scaled, y_train_raw, SEQ_LENGTH)
    val_dataset = StockDataset(X_val_scaled, y_val_raw, SEQ_LENGTH)
    test_dataset = StockDataset(X_test_scaled, y_test_raw, SEQ_LENGTH)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Not enough data for sequences. Increase total_days or decrease seq_length.")
        return

    # Create DataLoaders
    # Pin memory for faster transfer to GPU (MPS/CUDA)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for safety on Mac sometimes
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 4. Setup Model & Device ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Class Weights for Imbalance (Calculate on full train set)
    # Note: StockDataset labels are at [seq_len:], so we need to approximate or calculate correctly.
    # We can access labels directly from the dataset logic or just use y_train_raw[SEQ_LENGTH:]
    y_train_actual = y_train_raw[SEQ_LENGTH:]
    classes = np.unique(y_train_actual)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_actual)
    
    # Map back to [0, 1, 2]
    full_weights = np.ones(3, dtype=np.float32)
    for i, cls in enumerate(classes):
        full_weights[cls] = weights[i]
        
    class_weights = torch.tensor(full_weights, dtype=torch.float32).to(device)
    print(f"Class Weights: {class_weights.cpu().numpy()}")
    
    model = StockTCN(input_size=len(feature_cols), num_classes=3, num_layers=NN_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # --- 5. Training Loop ---
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []
    
    best_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Use progress bar
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for X_batch, y_batch in loop:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)
            
            loop.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                
                running_val_loss += loss.item() * X_val.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == y_val).sum().item()
                total_val += y_val.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100
            
        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)
        train_acc_hist.append(epoch_train_acc)
        val_acc_hist.append(epoch_val_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}%")
        
        # Early Stopping
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
                
    # --- 6. Final Evaluation on TEST Set ---
    # Load best model
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # Final Test Loop
    correct_test = 0
    total_test = 0
    test_loss_sum = 0
    all_test_preds = []
    all_test_targets = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
             X_test, y_test = X_test.to(device), y_test.to(device)
             outputs = model(X_test)
             loss = criterion(outputs, y_test)
             
             test_loss_sum += loss.item() * X_test.size(0)
             preds = torch.argmax(outputs, dim=1)
             correct_test += (preds == y_test).sum().item()
             total_test += y_test.size(0)
             
             all_test_preds.extend(preds.cpu().numpy())
             all_test_targets.extend(y_test.cpu().numpy())
    
    test_acc = (correct_test / total_test) * 100
    test_loss = test_loss_sum / total_test
    
    duration = time.time() - start_time
    print(f"\n" + "="*40)
    print(f"FINAL TEST RESULTS (on unseen data)")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Duration:      {duration:.2f} seconds")
    print("="*40)
    
    # Detailed Class Performance
    all_test_preds_np = np.array(all_test_preds)
    all_test_targets_np = np.array(all_test_targets)
    
    classes = [0, 1, 2]
    class_names = ['Hold', 'Buy', 'Sell']
    
    print("\n" + "-"*40)
    print("DETAILED CLASS PERFORMANCE")
    print("-"*40)
    
    for cls, name in zip(classes, class_names):
        total_cls = (all_test_targets_np == cls).sum()
        correct_cls = ((all_test_preds_np == cls) & (all_test_targets_np == cls)).sum()
        acc_cls = (correct_cls / total_cls * 100) if total_cls > 0 else 0.0
        
        print(f"{name:<5} (Class {cls}): Total={total_cls:<4} | Correct={correct_cls:<4} | Acc={acc_cls:.1f}%")
    print("-"*40)
    
    print("="*40)
    
    # --- 7. Visualization & Logging ---
    print("\nGenerating Graphs...")
    plot_training_metrics(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist, GRAPHS_DIR)
    # plot_confusion_matrix(y_test_true, test_pred, "Test Confusion Matrix", "test_cm.png", GRAPHS_DIR)
    plot_confusion_matrix(all_test_targets, all_test_preds, "Test Confusion Matrix", "test_cm.png", GRAPHS_DIR)
    
    # Feature Importance (Proxy via correlation or basic logic? TCN is hard to interpret directly without SHAP)
    # Skipping detailed feature importance for now, focusing on performance.
    
    # Log to CSV
    file_exists = os.path.isfile(RESULTS_FILE)
    is_empty = file_exists and os.path.getsize(RESULTS_FILE) == 0
    
    header = [
        "timestamp", "total_days", "seq_length", "epochs", 
        "train_samples", "test_samples", 
        "best_val_loss", "test_acc", "test_loss", "duration_sec"
    ]
    
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        TOTAL_DAYS, SEQ_LENGTH, EPOCHS,
        len(train_dataset), len(test_dataset),
        f"{best_loss:.4f}", f"{test_acc:.2f}%", f"{test_loss:.4f}",
        f"{duration:.2f}"
    ]
    
    with open(RESULTS_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or is_empty:
            writer.writerow(header)
        writer.writerow(row)
        
    print(f"Results logged to {RESULTS_FILE}")

if __name__ == "__main__":
    X_raw, y_raw, feature_cols = prepare_global_data(total_days=TOTAL_DAYS)

    if X_raw is not None:
        try:
            train_and_evaluate(X_raw, y_raw, feature_cols)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
