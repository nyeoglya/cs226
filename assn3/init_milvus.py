import os
import io
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# ======== Configuration ========
DIM = 512 # 임베딩 벡터 차원
METRIC = "COSINE" # 코사인 유사도
HOST = "localhost" # Milvus 서버 호스트
PORT = "19530" # Milvus 서버 포트
BATCH_ROWS = 10000 # 데이터베이스 파일들 쪼개기 (성능 이슈)
T2I_COL_NAME = "text_to_img" # Milvus 컬럼 이름
I2T_COL_NAME = "img_to_text"
DB_EMBED_DIR = "data/database" # DB 파일 경로

# ======== Helpers ========
def _to_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False)

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

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
        print(f"Dropping existing collection: {name}")
        utility.drop_collection(name)
    print(f"Creating collection: {name}")
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
    print(f"Flushing collection: {col.name}")
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

    print("✓ Milvus initialization finished. Collections are ready.")


# 메인 함수
if __name__ == "__main__":
    # ======== Load Embeddings ========
    img_embed_path = os.path.join(DB_EMBED_DIR, "img_embeddings.npy")
    txt_embed_path = os.path.join(DB_EMBED_DIR, "txt_embeddings.npy")
    
    if not os.path.exists(img_embed_path) or not os.path.exists(txt_embed_path):
        print(f"Error: Embedding files not found in {DB_EMBED_DIR}")
        print("Please make sure 'img_embeddings.npy' and 'txt_embeddings.npy' exist.")
        exit()
        
    img_embeddings = np.load(img_embed_path)
    txt_embeddings = np.load(txt_embed_path)
    print("✓ Load Embeddings")

    # ======== Data Validation ========
    assert img_embeddings.ndim == 2 and img_embeddings.shape[1] == DIM, f"img_embeddings shape must be (N, {DIM})"
    assert txt_embeddings.ndim == 2 and txt_embeddings.shape[1] == DIM, f"txt_embeddings shape must be (M, {DIM})"
    assert img_embeddings.dtype in (np.float32, np.float64), "img dtype must be float32/64"
    assert txt_embeddings.dtype in (np.float32, np.float64), "txt dtype must be float32/64"
    print("✓ Data Validation")

    # ======== Connect Milvus Server ========
    try:
        connections.connect("default", host=HOST, port=PORT)
        print("✓ Connect Milvus Server")
    except Exception as e:
        print(f"Failed to connect to Milvus at {HOST}:{PORT}. Is it running?")
        print(f"Error: {e}")
        exit()

    # ======== Build Schema ========
    schema = _build_schema()
    t2i_col = _create_or_recreate_collection(T2I_COL_NAME, schema)
    i2t_col = _create_or_recreate_collection(I2T_COL_NAME, schema)
    print("✓ Build Schema")

    # ======== Milvus Initialization ========
    init_milvus(img_embeddings, txt_embeddings, t2i_col, i2t_col)
    print("✓ Initialization Completed")
