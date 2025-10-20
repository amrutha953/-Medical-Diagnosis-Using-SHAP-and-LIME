import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, dataloader, device="cpu"):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            out = model(xb)
            p = F.softmax(out, dim=1)[:,1].detach().cpu().numpy()
            pred = (p >= 0.5).astype(int)
            preds.extend(pred.tolist())
            probs.extend(p.tolist())
            ys.extend(yb.numpy().tolist())
    acc = accuracy_score(ys, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(ys, preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(ys, probs)
    except:
        auc = float('nan')
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1, "auc": auc}

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def plot_metrics(history, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["loss_train"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss_train"], label="train_loss")
    plt.plot(epochs, history["loss_val"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, history["acc_train"], label="train_acc")
    plt.plot(epochs, history["acc_val"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    plt.close()
