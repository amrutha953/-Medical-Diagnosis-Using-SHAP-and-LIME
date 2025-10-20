# src/explain.py
import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from src.model import get_model
from src.data_preprocessing import get_dataloaders
from src.utils import save_model
import matplotlib.pyplot as plt
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

# helper to convert tensor image to numpy (un-normalized)
def tensor_to_np(img_tensor):
    # img_tensor: C,H,W in normalized range
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std  = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = img_tensor.cpu().numpy()
    img = (img * std + mean)  # unnormalize
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1,2,0))  # H,W,C
    return (img * 255).astype(np.uint8)

def predict_fn(images_np, model, device):
    """images_np: list or array of images in HWC, range 0-255"""
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    batch = []
    for img in images_np:
        x = transform(Image.fromarray(img)).unsqueeze(0)
        batch.append(x)
    batch = torch.cat(batch, dim=0).to(device)
    with torch.no_grad():
        out = model(batch)
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    return probs

def shap_explain(model, test_loader, device, nsamples=50):
    # prepare background (a small set of images)
    bg_images = []
    for i, (xb, yb) in enumerate(test_loader):
        for j in range(min(4, xb.size(0))):
            bg_images.append(tensor_to_np(xb[j]))
        if len(bg_images) >= 16:
            break

    # pick a few target images to explain
    targets = []
    target_tensors = []
    for i, (xb, yb) in enumerate(test_loader):
        for j in range(min(2, xb.size(0))):
            target_tensors.append(xb[j])
            targets.append((tensor_to_np(xb[j]), int(yb[j].item())))
        if len(targets) >= 4:
            break

    # shap expects a function mapping (numpy images) -> model output probs
    f = lambda imgs: predict_fn(imgs, model, device)

    explainer = shap.KernelExplainer(f, bg_images[:50])  # KernelExplainer works for black-box but is slower
    os.makedirs("results/explanations/shap", exist_ok=True)

    for idx, (img_np, label) in enumerate(targets):
        # KernelExplainer wants 2D vectors, but for images this is heavy.
        # We'll ask for prediction for original image.
        shap_values = explainer.shap_values(img_np, nsamples=nsamples)
        # shap_values is list of arrays for each class: [class0_vals, class1_vals]
        # Aggregate across channels to make a heatmap
        shap_sum = np.sum(np.abs(shap_values[1]), axis=2)  # use class 1 (pneumonia) contribution
        plt.figure(figsize=(6,6))
        plt.imshow(img_np)
        plt.imshow(shap_sum, cmap='jet', alpha=0.5)
        plt.title(f"SHAP (class=1) label={label}")
        plt.axis('off')
        plt.savefig(f"results/explanations/shap/shap_{idx}.png", bbox_inches='tight')
        plt.close()
    print("Saved SHAP images to results/explanations/shap/")

def lime_explain(model, test_loader, device, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])

    os.makedirs("results/explanations/lime", exist_ok=True)

    # pick few images
    targets = []
    for i, (xb, yb) in enumerate(test_loader):
        for j in range(min(2, xb.size(0))):
            targets.append((xb[j], int(yb[j].item())))
        if len(targets) >= 4:
            break

    for idx, (timg, label) in enumerate(targets):
        img_np = tensor_to_np(timg)
        # LIME returns explanation object
        explanation = explainer.explain_instance(img_np, 
                                                 classifier_fn=lambda imgs: predict_fn(imgs, model, device),
                                                 top_labels=2,
                                                 hide_color=0,
                                                 num_samples=num_samples)
        # get image and mask for top class (predicted)
        top_label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)
        plt.figure(figsize=(6,6))
        plt.imshow(mark_boundaries(temp / 255.0, mask))
        plt.title(f"LIME top_label={top_label} true={label}")
        plt.axis("off")
        plt.savefig(f"results/explanations/lime/lime_{idx}.png", bbox_inches='tight')
        plt.close()
    print("Saved LIME images to results/explanations/lime/")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load model
    model = get_model(num_classes=2, pretrained=False, device=device)
    model_path = "results/models/resnet18_pneumonia.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first (python -m src.train)")
    model.load_state_dict(torch.load(model_path, map_location=device))

    _, _, test_loader, class_to_idx = get_dataloaders("data", batch_size=8)
    # generate explanations
    shap_explain(model, test_loader, device, nsamples=50)   # KernelExplainer is slow; reduce nsamples if necessary
    lime_explain(model, test_loader, device, num_samples=200)
