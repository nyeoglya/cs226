import os
import io
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, abort
from pymilvus import (
    connections, Collection, utility
)

# ======== Configuration ========
MODEL_NAME = "openai/clip-vit-base-patch32" # 사용할 CLIP 모델 이름
SEARCH_TOPK = 3 # K값
SEARCH_LIST = 64 # DiskANN 검색 파라미터
HOST = "localhost" # Milvus 서버 호스트
PORT = "19530" # Milvus 서버 포트
T2I_COL_NAME = "text_to_img" # Milvus 컬럼 이름
I2T_COL_NAME = "img_to_text"
DB_IMG_DIR = "data/database/images" # DB 파일 경로
DB_TXT_DIR = "data/database/texts"

# ======== Load Model ========
clip_model = CLIPModel.from_pretrained(MODEL_NAME)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
clip_model.eval()
torch.set_grad_enabled(False)
print("✓ Load Clip Model")

# ======== Helpers ========
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

# ======== Encoders ========
def img2vec(file_bytes: bytes) -> np.ndarray:
    '''
    목적: 이미지를 벡터로 임베딩
    입력: file_bytes(이미지 데이터)
    출력: 정규화된 numpy 벡터
    '''
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    feat = clip_model.get_image_features(**inputs)
    v = feat.detach().cpu().numpy().astype(np.float32, copy=False)
    v = _l2_normalize(v, axis=1)
    return v

def text2vec(text: str) -> np.ndarray:
    '''
    목적: 텍스트를 벡터로 임베딩
    입력: text(텍스트 데이터)
    출력: 정규화된 numpy 벡터
    '''
    inputs = clip_processor(text=[text], return_tensors="pt")
    feat = clip_model.get_text_features(**inputs)
    v = feat.detach().cpu().numpy().astype(np.float32, copy=False)
    v = _l2_normalize(v, axis=1)
    return v

# ======== Milvus Collections ========
t2i_col: Collection = None
i2t_col: Collection = None

# ======== Search Helpers ========
def i2t_search_vec(query_vec: np.ndarray, topk: int = SEARCH_TOPK) -> Tuple[List[int], List[float]]:
    '''
    목적: DiskANN으로 주어진 이미지와 유사한 텍스트 찾기
    입력:
        query_vec(이미지 쿼리)
        topk(반환할 id 개수)
    출력: 텍스트 id, 계산된 거리
    '''
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
    '''
    목적: DiskANN으로 주어진 텍스트와 유사한 이미지 찾기
    입력:
        query_vec(텍스트 쿼리)
        topk(반환할 id 개수)
    출력: 이미지 id, 계산된 거리
    '''
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

def _read_text_by_id(item_id: int) -> str:
    path = os.path.join(DB_TXT_DIR, f"{item_id:06d}.txt")
    if not os.path.isfile(path):
        return "[파일 없음]"
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp949", errors="replace") as f:
            txt = f.read()
    return txt

# 기본 경로
@app.get("/")
def index():
    return render_template(
        "index.html",
        t2i_results=[],
        i2t_results=[],
        query_text=""
    )

# 검색 수행시 경로
@app.post("/search")
def search():
    action = request.form.get("action")
    query_text = request.form.get("query_text", "").strip()
    file = request.files.get("query_image")

    t2i_results = []
    i2t_results = []

    try:
        if action == "t2i": # txt2img 검색
            if not query_text:
                return redirect(url_for("index"))
            qv = text2vec(query_text) # 텍스트 쿼리를 벡터로 변환
            ids, dists = t2i_search_vec(qv, topk=SEARCH_TOPK) # 유사한 이미지 찾기
            for i, d in zip(ids, dists): # 각각을 불러와서 결과 객체에 저장
                img_path = os.path.join(DB_IMG_DIR, f"{i:06d}.jpg")
                if os.path.isfile(img_path):
                    t2i_results.append({"id": i, "distance": float(d)})
                else:
                    raise FileNotFoundError(f"Image file not found: {img_path}") # 파일 없으면 오류 출력
        elif action == "i2t": # img2txt 검색
            if not file or not file.filename:
                return redirect(url_for("index"))
            file_bytes = file.read() # 이미지 쿼리를 벡터로 변환
            qv = img2vec(file_bytes) # 유사한 텍스트 찾기
            ids, dists = i2t_search_vec(qv, topk=SEARCH_TOPK)
            for i, d in zip(ids, dists): # 각각을 불러와서 결과 객체에 저장
                preview = _read_text_by_id(i)
                i2t_results.append({"id": i, "distance": float(d), "text": preview})
    except Exception as e:
        print(f"Search error: {type(e).__name__}: {e}")
        return f"Search error: {type(e).__name__}: {e}", 500

    # index.html을 불러온 결과와 함께 로딩
    return render_template("index.html",
        t2i_results=t2i_results,
        i2t_results=i2t_results,
        query_text=query_text
    )

# 정적 미디어 라우트
@app.get("/media/image/<int:item_id>")
def media_image(item_id: int):
    fname_jpg = f"{item_id:06d}.jpg"
    if os.path.isfile(os.path.join(DB_IMG_DIR, fname_jpg)):
        return send_from_directory(DB_IMG_DIR, fname_jpg)
    abort(404)

@app.get("/media/text/<int:item_id>")
def media_text(item_id: int):
    fname = f"{item_id:06d}.txt"
    if os.path.isfile(os.path.join(DB_TXT_DIR, fname)):
        return send_from_directory(DB_TXT_DIR, fname, mimetype="text/plain", as_attachment=False)
    abort(404)

# 메인 함수
if __name__ == "__main__":
    # ======== Connect Milvus Server ========
    try:
        connections.connect("default", host=HOST, port=PORT)
        print("✓ Connect Milvus Server")
    except Exception as e:
        print(f"Failed to connect to Milvus at {HOST}:{PORT}. Is it running?")
        print(f"Error: {e}")
        exit()

    # ======== Get & Load Collections ========
    if not utility.has_collection(T2I_COL_NAME) or not utility.has_collection(I2T_COL_NAME):
        print(f"Error: Collections '{T2I_COL_NAME}' or '{I2T_COL_NAME}' not found.")
        print("Please run the init_milvus.py script first.")
        exit()

    # 컬렉션 할당
    t2i_col = Collection(T2I_COL_NAME)
    i2t_col = Collection(I2T_COL_NAME)

    # 검색을 위해 컬렉션을 메모리로 로드
    print("Loading collections into memory...")
    t2i_col.load()
    i2t_col.load()
    print("✓ Load Collections")

    # ======== Run Flask Server ========
    print("Starting Flask server at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
