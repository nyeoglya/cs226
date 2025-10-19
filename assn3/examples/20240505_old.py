# app.py
import os
import io
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# ======== Configuration ========
MODEL_NAME = "openai/clip-vit-base-patch32"
DIM = 512
METRIC = "COSINE"          # "COSINE" | "L2" | "IP"
SEARCH_TOPK = 3
SEARCH_LIST = 64           # DiskANN search parameter
HOST = "localhost"
PORT = "19530"
BATCH_ROWS = 10000

# DB 파일 시스템 경로 (%06d.jpg / %06d.txt)
DB_IMG_DIR = "data/database/images"
DB_TXT_DIR = "data/database/texts"

# 결과 미리보기 제한(텍스트)
TEXT_PREVIEW_CHARS = 400

# ======== Load Model ========
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
clip_model.eval()
torch.set_grad_enabled(False)
print("✓ Load Clip Model")

# ======== Helpers ========
def _to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

# ----- Encoders -----
def img2vec_from_path(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return img2vec_from_image(img)

def img2vec_from_bytes(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return img2vec_from_image(img)

def img2vec_from_image(img: Image.Image) -> np.ndarray:
    inputs = clip_processor(images=img, return_tensors="pt")
    feat = clip_model.get_image_features(**inputs)
    v = feat.detach().cpu().numpy().astype(np.float32, copy=False)
    if METRIC.upper() in ("COSINE", "IP"):
        v = _l2_normalize(v, axis=1)
    return v

def text2vec(text: str) -> np.ndarray:
    inputs = clip_processor(text=[text], return_tensors="pt")
    feat = clip_model.get_text_features(**inputs)
    v = feat.detach().cpu().numpy().astype(np.float32, copy=False)
    if METRIC.upper() in ("COSINE", "IP"):
        v = _l2_normalize(v, axis=1)
    return v

# ======== Load Embeddings ========
img_embeddings = np.load("data/database/img_embeddings.npy")
txt_embeddings = np.load("data/database/txt_embeddings.npy")
print("✓ Load Embeddings")

# ======== Data Validation ========
assert img_embeddings.ndim == 2 and img_embeddings.shape[1] == DIM, f"img_embeddings shape must be (N, {DIM})"
assert txt_embeddings.ndim == 2 and txt_embeddings.shape[1] == DIM, f"txt_embeddings shape must be (M, {DIM})"
assert img_embeddings.dtype in (np.float32, np.float64), "img dtype must be float32/64"
assert txt_embeddings.dtype in (np.float32, np.float64), "txt dtype must be float32/64"
print("✓ Data Validation")

# ======== Milvus Schema / Collections ========
def _build_schema() -> CollectionSchema:
    vec_field = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    id_field  = FieldSchema(name="id",  dtype=DataType.INT64, is_primary=True, auto_id=False)
    return CollectionSchema(fields=[id_field, vec_field])

def _create_or_recreate_collection(name: str, schema: CollectionSchema) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)
    return Collection(name=name, schema=schema)

def _create_diskann_index(col: Collection):
    # DiskANN은 metric_type에 따라 index가 달라짐
    col.create_index(
        field_name="vec",
        index_params={"index_type": "DISKANN", "metric_type": METRIC, "params": {}}
    )

def insert_in_batches(col: Collection, ids: np.ndarray, vecs: np.ndarray, batch: int = BATCH_ROWS):
    assert vecs.dtype == np.float32 and vecs.ndim == 2
    assert ids.dtype == np.int64
    n = vecs.shape[0]
    for start in tqdm(range(0, n, batch), desc=f"Insert {col.name}"):
        end = min(start + batch, n)
        col.insert([ids[start:end].tolist(), vecs[start:end].tolist()])
    col.flush()

# ======== Connect & Initialize Milvus ========
connections.connect("default", host=HOST, port=PORT)
print("✓ Connect")

T2I_COL_NAME = "text_to_img"
I2T_COL_NAME = "img_to_text"
schema = _build_schema()

t2i_col = _create_or_recreate_collection(T2I_COL_NAME, schema)
i2t_col = _create_or_recreate_collection(I2T_COL_NAME, schema)
print("✓ Build Schema")

def init_milvus():
    img_vecs = _to_float32(img_embeddings)[:1000] # TODO: 제거
    txt_vecs = _to_float32(txt_embeddings)[:1000] # TODO: 제거
    if METRIC.upper() in ("COSINE", "IP"):
        img_vecs = _l2_normalize(img_vecs, axis=1)
        txt_vecs = _l2_normalize(txt_vecs, axis=1)
    img_ids = np.arange(img_vecs.shape[0], dtype=np.int64)
    txt_ids = np.arange(txt_vecs.shape[0], dtype=np.int64)

    insert_in_batches(t2i_col, img_ids, img_vecs)
    insert_in_batches(i2t_col, txt_ids, txt_vecs)
    print("✓ Insert Data")

    _create_diskann_index(t2i_col); print("✓ Create Text→Image Index")
    _create_diskann_index(i2t_col); print("✓ Create Image→Text Index")

    t2i_col.load()
    i2t_col.load()
    print("✓ Load Collections")

# 초기화 1회 수행
init_milvus()
print("✓ Initialization Completed")

# ======== Search Helpers (return ids, distances) ========
def i2t_search_vec(query_vec: np.ndarray, topk: int = SEARCH_TOPK) -> Tuple[List[int], List[float]]:
    res = i2t_col.search(
        data=query_vec,
        anns_field="vec",
        param={"search_list": max(SEARCH_LIST, topk)},
        limit=topk,
        output_fields=[]
    )
    hits = res[0]
    return hits.ids, hits.distances

def t2i_search_vec(query_vec: np.ndarray, topk: int = SEARCH_TOPK) -> Tuple[List[int], List[float]]:
    res = t2i_col.search(
        data=query_vec,
        anns_field="vec",
        param={"search_list": max(SEARCH_LIST, topk)},
        limit=topk,
        output_fields=[]
    )
    hits = res[0]
    return hits.ids, hits.distances

# ======== Flask App ========
app = Flask(__name__)

def _read_text_by_id(item_id: int, limit_chars: int = TEXT_PREVIEW_CHARS) -> str:
    path = os.path.join(DB_TXT_DIR, f"{item_id:06d}.txt")
    if not os.path.isfile(path):
        return "[파일 없음]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949", errors="replace") as f:
            txt = f.read()
    if len(txt) > limit_chars:
        txt = txt[:limit_chars] + " …"
    return txt

@app.get("/")
def index():
    return render_template(
        "index.html",
        t2i_results=[],
        i2t_results=[],
        query_text=""
    )

@app.post("/search")
def search():
    action = request.form.get("action")
    query_text = request.form.get("query_text", "").strip()
    file = request.files.get("query_image")

    t2i_results = []
    i2t_results = []

    try:
        if action == "t2i":
            if not query_text:
                return redirect(url_for("index"))
            qv = text2vec(query_text)  # already normalized if COSINE/IP
            ids, dists = t2i_search_vec(qv, topk=SEARCH_TOPK)
            for i, d in zip(ids, dists):
                # 이미지가 실제로 존재하는지 체크
                img_path = os.path.join(DB_IMG_DIR, f"{i:06d}.jpg")
                if not os.path.isfile(img_path):
                    # jpg가 없으면 png도 시도 (선택)
                    alt = os.path.join(DB_IMG_DIR, f"{i:06d}.png")
                    if os.path.isfile(alt):
                        pass
                t2i_results.append({"id": i, "distance": float(d)})

        elif action == "i2t":
            if not file or not file.filename:
                return redirect(url_for("index"))
            file_bytes = file.read()
            qv = img2vec_from_bytes(file_bytes)  # already normalized if COSINE/IP
            ids, dists = i2t_search_vec(qv, topk=SEARCH_TOPK)
            for i, d in zip(ids, dists):
                preview = _read_text_by_id(i)
                i2t_results.append({"id": i, "distance": float(d), "text": preview})

    except Exception as e:
        # 간단한 오류 표출 (실서비스면 로깅/에러페이지 분리)
        return f"Search error: {type(e).__name__}: {e}", 500

    return render_template("index.html",
        t2i_results=t2i_results,
        i2t_results=i2t_results,
        query_text=query_text
    )

# 정적 미디어 라우트
@app.get("/media/image/<int:item_id>")
def media_image(item_id: int):
    # jpg 우선, 없으면 png 시도
    fname_jpg = f"{item_id:06d}.jpg"
    fname_png = f"{item_id:06d}.png"
    if os.path.isfile(os.path.join(DB_IMG_DIR, fname_jpg)):
        return send_from_directory(DB_IMG_DIR, fname_jpg)
    if os.path.isfile(os.path.join(DB_IMG_DIR, fname_png)):
        return send_from_directory(DB_IMG_DIR, fname_png)
    abort(404)

@app.get("/media/text/<int:item_id>")
def media_text(item_id: int):
    fname = f"{item_id:06d}.txt"
    if os.path.isfile(os.path.join(DB_TXT_DIR, fname)):
        return send_from_directory(DB_TXT_DIR, fname, mimetype="text/plain", as_attachment=False)
    abort(404)

if __name__ == "__main__":
    # 개발 서버 실행
    # 환경에 맞게 host/port 조정
    app.run(host="0.0.0.0", port=5000, debug=False)
