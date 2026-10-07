import torch
import torch.nn as nn
import torchvision.models as models
import io
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy as np

# ─────────────────────────────────────────────
#  Model Definition  –  EfficientNet-B0 backbone
#  Pretrained on ImageNet → fine-tuned for deepfake detection
# ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeDetector, self).__init__()

        # EfficientNet-B0: fast, lightweight, yet very accurate
        self.backbone = models.efficientnet_b0(weights=None)

        if pretrained:
            print("[Model] Downloading ImageNet weights manually (ignoring hash)...")
            state_dict = torch.hub.load_state_dict_from_url(
                "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
                check_hash=False
            )
            self.backbone.load_state_dict(state_dict)

        # Freeze early layers (feature extraction) — only fine-tune later layers
        # This speeds up training and prevents overfitting on small datasets
        for name, param in self.backbone.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False

        # Replace classifier head for binary classification (Real=0 vs Fake=1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)  # Single output node → BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.backbone(x)


def get_model(device, pretrained=True):
    model = DeepfakeDetector(pretrained=pretrained)
    model = model.to(device)
    return model


# ─────────────────────────────────────────────
#  Inference Pipeline
# ─────────────────────────────────────────────

# Initialize a global model so we don't reload on every request
global_model = None
global_device = None

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inference transform — matches training normalization
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def extract_face(image_np):
    """Crop the largest detected face from the image (with margin)."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None  # No face found — use full image

    # Pick the largest face
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # Add 20% margin around the face
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_np.shape[1], x + w + margin_x)
    y2 = min(image_np.shape[0], y + h + margin_y)

    return image_np[y1:y2, x1:x2]


def get_initialized_model():
    """Load model once and cache globally."""
    global global_model, global_device

    if global_model is None:
        global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Model] Using device: {global_device}")

        global_model = get_model(global_device, pretrained=False)  # Don't re-download ImageNet weights

        model_path = 'best_model.pth'
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=global_device)
                global_model.load_state_dict(state)
                print("[Model] Loaded best_model.pth ✓")
            except Exception as e:
                print(f"[Model] WARNING: Could not load best_model.pth — {e}")
                print("[Model] Using untrained model. Run train.py first!")
        else:
            print("[Model] WARNING: best_model.pth not found. Run train.py first!")

        global_model.eval()

    return global_model, global_device


def predict_image(image_bytes=None, image_pil=None):
    """
    Predict whether an image is FAKE or REAL.
    Returns probability (0-100%) that the image is FAKE.
    """
    model, device = get_initialized_model()

    if image_bytes is not None:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    elif image_pil is not None:
        image = image_pil.convert('RGB')
    else:
        raise ValueError("Provide either image_bytes or image_pil")

    # Attempt face crop first
    img_np = np.array(image)
    face_np = extract_face(img_np)
    if face_np is not None and face_np.size > 0:
        image = Image.fromarray(face_np)

    try:
        tensor = inference_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(tensor)
            prob = torch.sigmoid(logit).item() * 100.0  # % chance of being FAKE
        return prob
    except Exception as e:
        print(f"[Inference] Error: {e}")
        return 50.0  # Neutral fallback


def predict_video(video_path, max_frames=16):
    """
    Sample frames uniformly from a video and average fake probabilities.
    Returns average probability (0-100%) that the video is FAKE.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return 50.0

    step = max(1, total_frames // max_frames)
    probs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and len(probs) < max_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            prob = predict_image(image_pil=pil_img)
            probs.append(prob)

        frame_idx += 1

    cap.release()

    if not probs:
        return 50.0

    return sum(probs) / len(probs)


# ─────────────────────────────────────────────
#  Quick Sanity Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device, pretrained=True)
    test_tensor = torch.randn(2, 3, 224, 224).to(device)
    output = model(test_tensor)
    print(f"Output shape: {output.shape}  (Expected: [2, 1])")
    print(f"Probs: {torch.sigmoid(output).squeeze().tolist()}")
    print("✓ Model initialized successfully.")
