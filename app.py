from flask import Flask, request, jsonify
from PIL import Image
import io, base64, torch, os
from torchvision import transforms
import numpy as np

# ✅ Import model and utils
from src.model import get_model
from src.utils import evaluate

app = Flask(__name__)

# -------------------------------
# 🔹 Model Loading
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = os.path.join("results", "models", "resnet18_pneumonia.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

# Load model
model = get_model(num_classes=2, pretrained=False, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------------------------------
# 🔹 Image Preprocessing
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# 🔹 Prediction Function
# -------------------------------
def predict_image(img):
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    labels = ["Normal", "Pneumonia"]
    return labels[pred.item()], conf.item()

# -------------------------------
# 🔹 Dummy SHAP & LIME Heatmaps (for demo)
# -------------------------------
def make_dummy_heatmap(img):
    arr = np.array(img.convert("RGB"))
    red = np.zeros_like(arr)
    red[..., 0] = arr[..., 0]  # Red tint to simulate highlight
    red_img = Image.fromarray(red)
    buffer = io.BytesIO()
    red_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# -------------------------------
# 🔹 Flask API Route
# -------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")

    # Model prediction
    label, conf = predict_image(img)

    # Dummy SHAP & LIME visualization
    shap_heatmap = make_dummy_heatmap(img)
    lime_heatmap = make_dummy_heatmap(img)

    return jsonify({
        "label": label,
        "confidence": round(float(conf), 3),
        "shap_heatmap": f"data:image/png;base64,{shap_heatmap}",
        "lime_heatmap": f"data:image/png;base64,{lime_heatmap}"
    })

# -------------------------------
# 🔹 Run Flask App
# -------------------------------
if __name__ == "__main__":
    print("🚀 Backend running at: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
