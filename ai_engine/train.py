"""
Training Pipeline for AI-Based Driver Behaviour, Fatigue and Risk Detection System.
Trains:
1. Multi-Modal PyTorch CNN-LSTM Architecture (Dual-Head: Classification + Risk Score)
2. Classical Scikit-Learn ML Baseline (Random Forest & Gradient Boosting)
Generates evaluation metrics, confusion matrices, and model checkpoints.
"""

import sys
import os
import json
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.config import (
    CLASSES, NUM_CLASSES, SAVED_MODELS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, EPOCHS, RANDOM_SEED
)
from ai_engine.dataset import build_or_load_dataset
from ai_engine.models import DriverRiskCNNLSTM, ClassicalMLClassifier


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_deep_model(
    train_dataset,
    val_dataset,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE
):
    print("\n" + "=" * 60)
    print("  TRAINING MULTI-MODAL CNN-LSTM DEEP NEURAL NETWORK")
    print("=" * 60)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = DriverRiskCNNLSTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    cls_criterion = nn.CrossEntropyLoss()
    risk_criterion = nn.SmoothL1Loss()
    
    best_val_f1 = 0.0
    best_checkpoint_path = SAVED_MODELS_DIR / "driver_cnn_lstm.pth"
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_risk_mae": [], "val_f1": []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0
        
        for batch in train_loader:
            tel = batch["telemetry"].to(device)
            cam = batch["camera"].to(device)
            labels = batch["label"].to(device)
            risk = batch["risk_score"].to(device)
            
            optimizer.zero_grad()
            outputs = model(telemetry_seq=tel, camera_seq=cam)
            
            logits = outputs["logits"]
            pred_risk = outputs["risk_score"]
            
            loss_cls = cls_criterion(logits, labels)
            loss_risk = risk_criterion(pred_risk, risk)
            # Weighted multi-task loss
            loss = loss_cls + 0.04 * loss_risk
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            total_train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train_samples += len(labels)
            
        scheduler.step()
        
        train_loss = total_train_loss / total_train_samples
        train_acc = correct_train / total_train_samples
        
        # Validation pass
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        all_val_preds = []
        all_val_labels = []
        val_risk_diffs = []
        
        with torch.no_grad():
            for batch in val_loader:
                tel = batch["telemetry"].to(device)
                cam = batch["camera"].to(device)
                labels = batch["label"].to(device)
                risk = batch["risk_score"].to(device)
                
                outputs = model(telemetry_seq=tel, camera_seq=cam)
                logits = outputs["logits"]
                pred_risk = outputs["risk_score"]
                
                loss_cls = cls_criterion(logits, labels)
                loss_risk = risk_criterion(pred_risk, risk)
                val_loss = loss_cls + 0.04 * loss_risk
                
                total_val_loss += val_loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val_samples += len(labels)
                
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                val_risk_diffs.extend(torch.abs(pred_risk - risk).cpu().numpy())
                
        val_loss = total_val_loss / total_val_samples
        val_acc = correct_val / total_val_samples
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")
        val_mae = float(np.mean(val_risk_diffs))
        
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_risk_mae"].append(round(val_mae, 4))
        history["val_f1"].append(round(val_f1, 4))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | "
              f"Val Acc: {val_acc*100:.1f}% | Val F1: {val_f1:.4f} | Risk MAE: {val_mae:.2f}")
        
        # Save best model checkpoint
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_risk_mae": val_mae,
                "classes": CLASSES
            }, best_checkpoint_path)
            
    total_time = time.time() - start_time
    print(f"\nCNN-LSTM Training completed in {total_time:.2f}s! Best Val F1: {best_val_f1:.4f}")
    
    # Final detailed evaluation of best model
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            tel = batch["telemetry"].to(device)
            cam = batch["camera"].to(device)
            labels = batch["label"].to(device)
            outputs = model(telemetry_seq=tel, camera_seq=cam)
            all_preds.extend(torch.argmax(outputs["logits"], dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds).tolist()
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    
    deep_metrics = {
        "best_epoch": checkpoint["epoch"],
        "val_accuracy": float(checkpoint["val_acc"]),
        "val_f1": float(checkpoint["val_f1"]),
        "risk_mae": float(checkpoint["val_risk_mae"]),
        "confusion_matrix": cm,
        "classification_report": report,
        "training_history": history
    }
    
    return model, deep_metrics


def train_ml_baseline(train_X: np.ndarray, val_X: np.ndarray, train_y: np.ndarray, val_y: np.ndarray):
    print("\n" + "=" * 60)
    print("  TRAINING CLASSICAL MACHINE LEARNING BASELINE (Random Forest)")
    print("=" * 60)
    
    ml_model = ClassicalMLClassifier()
    ml_model.fit(train_X, train_y)
    
    metrics = ml_model.evaluate(val_X, val_y)
    print(f"ML Baseline Test Accuracy: {metrics['accuracy'] * 100:.2f}% | F1 Score: {metrics['f1_score']:.4f}")
    
    # Save scikit-learn model
    ml_checkpoint_path = SAVED_MODELS_DIR / "ml_baseline.joblib"
    joblib.dump(ml_model, ml_checkpoint_path)
    print(f"ML baseline model saved to {ml_checkpoint_path}")
    
    return ml_model, metrics


def run_full_training():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected computation device: {device}")
    
    # 1. Prepare Datasets (120 sequences per class = 720 total sequences)
    train_ds, val_ds, train_ml_X, val_ml_X, train_y, val_y = build_or_load_dataset(
        num_samples_per_class=120,
        seed=RANDOM_SEED,
        force_rebuild=False
    )
    
    # 2. Train Classical ML Baseline
    _, ml_metrics = train_ml_baseline(train_ml_X, val_ml_X, train_y, val_y)
    
    # 3. Train Multi-Modal CNN-LSTM Deep Neural Network
    _, deep_metrics = train_deep_model(
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
    
    # 4. Save Combined Evaluation Metrics Summary
    combined_metrics = {
        "classes": CLASSES,
        "cnn_lstm_model": deep_metrics,
        "ml_baseline_model": ml_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metrics_path = SAVED_MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"\nAll training metrics saved to {metrics_path}")
    print("=" * 60)
    print("SYSTEM TRAINING COMPLETE - READY FOR INFERENCE")
    print("=" * 60)


if __name__ == "__main__":
    run_full_training()
