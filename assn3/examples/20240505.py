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
MODEL_NAME = "openai/clip-vit-base-patch32" # 사용할 CLIP 모델 이름
DIM = 512 # 임베딩 벡터 차원
METRIC = "COSINE" # 코사인 유사도
SEARCH_TOPK = 3 # K값
SEARCH_LIST = 64 # DiskANN 검색 파라미터
HOST = "localhost" # Milvus 서버 호스트
PORT = "19530" # Milvus 서버 포트
BATCH_ROWS = 10000 # 데이터베이스 파일들 쪼개기 (성능 이슈)
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
def _to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)

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

# ======== Milvus Schema / Collections ========
def _build_schema() -> CollectionSchema:
    '''
    목적: DiskANN 컬렉션 생성 위한 스키마 객체 정의
    입력: [없음]
    출력: 컬렉션에 포함할 데이터의 속성(ID, 벡터)이 정의된 스키마 객체
    '''
    vec_field = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    id_field  = FieldSchema(name="id",  dtype=DataType.INT64, is_primary=True, auto_id=False)
    return CollectionSchema(fields=[id_field, vec_field])

def _create_or_recreate_collection(name: str, schema: CollectionSchema) -> Collection:
    '''
    목적: 주어진 이름으로 컬렉션 생성. 동일한 이름의 컬렉션이 이미 있으면, 삭제 후 재생성
    입력:
        name(생성할 컬렉션 이름)
        schema(컬렉션 구조 정의한 스키마 객체)
    출력: 생성한 DiskANN 컬렉션
    '''

    if utility.has_collection(name):
        utility.drop_collection(name)
    return Collection(name=name, schema=schema)

def _create_diskann_index(col: Collection):
    '''
    목적: 벡터 검색을 위한 vec 필드에 DiskANN 인덱스 생성
    입력: col(인덱스 생성할 DiskANN 컬렉션 객체)
    출력: [없음]
    '''
    # DiskANN은 metric_type에 따라 index가 달라짐
    col.create_index(
        field_name="vec",
        index_params={"index_type": "DISKANN", "metric_type": METRIC, "params": {}}
    )

def insert_in_batches(col: Collection, ids: np.ndarray, vecs: np.ndarray, batch: int = BATCH_ROWS):
    '''
    목적: 컴퓨터 성능상 문제로 DiskANN에 데이터를 한번에 넣을 수 없으면 배치 크기로 나눠서 넣기
    입력:
        col(데이터를 삽입할 대상 DiskANN 컬렉션 객체)
        ids(삽입할 데이터 ID 배열)
        vecs(삽입할 임베딩 벡터 배열)
        batch(한 번에 삽입할 데이터의 최대 행 수)
    출력: [없음]
    '''
    assert vecs.dtype == np.float32 and vecs.ndim == 2
    assert ids.dtype == np.int64
    n = vecs.shape[0]
    for start in tqdm(range(0, n, batch), desc=f"Insert {col.name}"):
        end = min(start + batch, n)
        col.insert([ids[start:end].tolist(), vecs[start:end].tolist()])
    col.flush()

# ======== Connect & Initialize Milvus ========
def init_milvus(img_embeddings, txt_embeddings, t2i_col, i2t_col):
    '''
    목적: 이미지 및 텍스트 임베딩 준비. DiskANN 컬렉션에 데이터 삽입. DiskANN 인덱스 생성 및 불러오기. 최종적인 검색 준비.
    입력:
        img_embeddings(이미지 데이터 임베딩 벡터 배열)
        txt_embeddings(텍스트 데이터 임베딩 벡터 배열)
        t2i_col(txt2img 검색을 위한 DiskANN 컬렉션 객체. 이미지 벡터 저장)
        i2t_col(img2txt 검색을 위한 DiskANN 컬렉션 객체. 텍스트 벡터 저장)
    출력: [없음]
    '''
    img_vecs = _to_float32(img_embeddings)
    txt_vecs = _to_float32(txt_embeddings)
    img_vecs = _l2_normalize(img_vecs, axis=1)
    txt_vecs = _l2_normalize(txt_vecs, axis=1)
    img_ids = np.arange(img_vecs.shape[0], dtype=np.int64)
    txt_ids = np.arange(txt_vecs.shape[0], dtype=np.int64)

    insert_in_batches(t2i_col, img_ids, img_vecs)
    insert_in_batches(i2t_col, txt_ids, txt_vecs)
    print("✓ Insert Data")

    _create_diskann_index(t2i_col); print("✓ Create txt2img Index")
    _create_diskann_index(i2t_col); print("✓ Create img2txt Index")

    t2i_col.load()
    i2t_col.load()
    print("✓ Load Collections")

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
                    raise FileNotFoundError
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

    # ======== Connect Milvus Server ========
    connections.connect("default", host=HOST, port=PORT)
    print("✓ Connect Milvus Server")

    # ======== Build Schema ========
    schema = _build_schema()
    t2i_col = _create_or_recreate_collection(T2I_COL_NAME, schema)
    i2t_col = _create_or_recreate_collection(I2T_COL_NAME, schema)
    print("✓ Build Schema")

    # ======== Milvus Initialization ========
    init_milvus(img_embeddings, txt_embeddings, t2i_col, i2t_col)
    print("✓ Initialization Completed")

    # ======== Run Flask Server ========
    app.run(host="0.0.0.0", port=5000, debug=False)
