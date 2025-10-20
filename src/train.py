# src/train.py
import os
import train

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data_preprocessing import get_dataloaders
from src.model import get_model
from src.utils import evaluate, save_model, plot_metrics
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)

def train(data_dir="data/chest_xray", epochs=8, batch_size=32, lr=1e-4, device=None):
    # Fix device assignment
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(42)
    
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(data_dir, batch_size=batch_size)

    model = get_model(num_classes=2, pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"loss_train": [], "loss_val": [], "acc_train": [], "acc_val": []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.detach().cpu().numpy().tolist())

        epoch_loss = running_loss / len(train_loader.dataset)
        acc_train = float((np.array(all_preds) == np.array(all_labels)).mean())
        val_metrics = evaluate(model, val_loader, device=device)

        history["loss_train"].append(epoch_loss)
        history["acc_train"].append(acc_train)
        history["loss_val"].append(0)  # optional: compute val loss if needed
        history["acc_val"].append(val_metrics["accuracy"])

        print(f"Epoch {epoch}: train_loss={epoch_loss:.4f} train_acc={acc_train:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_auc={val_metrics['auc']:.4f}")

    # save model
    save_path = os.path.join("results", "models", "resnet18_pneumonia.pt")
    save_model(model, save_path)
    print(f"Model saved to {save_path}")

    # evaluate on test set
    test_metrics = evaluate(model, test_loader, device=device)
    print("Test metrics:", test_metrics)

    plot_metrics(history)

if __name__ == "__main__":
    # run as: python -m src.train
    train(data_dir="data/chest_xray", epochs=8, batch_size=32, lr=1e-4)
