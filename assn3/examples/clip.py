import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ===== configuration =====
MODEL_NAME = "openai/clip-vit-base-patch32"
METRIC = "COSINE"  # or "L2", "IP"

# ===== load model =====
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()
print("✓ Load Clip Model")

# ===== helpers =====
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

# ===== query encoders =====
@torch.no_grad()
def img2vec(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    feat = clip_model.get_image_features(**inputs)  # [1, 512]
    v = feat.cpu().numpy().astype(np.float32, copy=False)
    if METRIC.upper() in ("COSINE", "IP"):
        v = _l2_normalize(v, axis=1)
    return v

@torch.no_grad()
def text2vec(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt")
    feat = clip_model.get_text_features(**inputs)  # [1, 512]
    v = feat.cpu().numpy().astype(np.float32, copy=False)
    if METRIC.upper() in ("COSINE", "IP"):
        v = _l2_normalize(v, axis=1)
    return v

# ===== test =====
if __name__ == "__main__":
    img_vec = img2vec("data/query/images/animal.jpg")
    txt_vec = text2vec("a photo of a dog")

    print("Image vector shape:", img_vec.shape)
    print("Text vector shape:", txt_vec.shape)
