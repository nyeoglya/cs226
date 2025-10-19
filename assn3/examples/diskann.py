import numpy as np
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, utility
)

# ===== configuration =====
T2I_COL_NAME = "text_to_img"
I2T_COL_NAME = "img_to_text"
DIM = 512
METRIC = "COSINE"          # "COSINE" | "L2" | "IP"
SEARCH_TOPK = 3
SEARCH_LIST = 64           # DiskANN search parameter
HOST = "localhost"
PORT = "19530"
BATCH_ROWS = 10_000        # <- 청크 크기: RAM에 맞춰 조절

# ===== load embeddings =====
img_embeddings = np.load("data/database/img_embeddings.npy")
txt_embeddings = np.load("data/database/txt_embeddings.npy")
print("✓ Load Embeddings")

# ===== data validation =====
assert img_embeddings.ndim == 2 and img_embeddings.shape[1] == DIM, f"img_embeddings shape must be (N, {DIM})"
assert txt_embeddings.ndim == 2 and txt_embeddings.shape[1] == DIM, f"txt_embeddings shape must be (M, {DIM})"
assert img_embeddings.dtype in (np.float32, np.float64), "img dtype must be float32/64"
assert txt_embeddings.dtype in (np.float32, np.float64), "txt dtype must be float32/64"
print("✓ Data Validation")

# ===== helpers =====
def _to_float32(x: np.ndarray) -> np.ndarray:
    """Milvus FLOAT_VECTOR는 FP32."""
    return x.astype(np.float32, copy=False)

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """COSINE/IP 검색 시 권장: 벡터 정규화"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm

def _build_schema() -> CollectionSchema:
    vec_field = FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=DIM)
    id_field  = FieldSchema(name="id",  dtype=DataType.INT64, is_primary=True, auto_id=False)
    return CollectionSchema(fields=[id_field, vec_field])

def _create_or_recreate_collection(name: str, schema: CollectionSchema) -> Collection:
    if utility.has_collection(name):
        utility.drop_collection(name)
    return Collection(name=name, schema=schema)

def _create_diskann_index(col: Collection):
    col.create_index(
        field_name="vec",
        index_params={"index_type": "DISKANN", "metric_type": METRIC, "params": {}}
    )

def insert_in_chunks(col: Collection, ids: np.ndarray, vecs: np.ndarray, batch_rows: int = BATCH_ROWS):
    """클라이언트 OOM 방지용 청크 삽입."""
    assert vecs.ndim == 2 and vecs.shape[0] == ids.shape[0], "ids/vecs length mismatch"
    assert vecs.dtype == np.float32, f"vecs must be float32, got {vecs.dtype}"
    assert ids.dtype == np.int64, f"ids must be int64, got {ids.dtype}"

    n = vecs.shape[0]
    for start in range(0, n, batch_rows):
        end = min(start + batch_rows, n)
        col.insert([
            ids[start:end].tolist(),
            vecs[start:end].tolist()
        ])
    col.flush()

# ===== connect =====
connections.connect("default", host=HOST, port=PORT)
print("✓ Connect")

# ===== schema/collections =====
schema = _build_schema()
t2i_col = _create_or_recreate_collection(T2I_COL_NAME, schema)
i2t_col = _create_or_recreate_collection(I2T_COL_NAME, schema)
print("✓ Build Schema")

# ===== init milvus with chunked insert =====
def init_milvus():
    img_vecs = _to_float32(img_embeddings)
    txt_vecs = _to_float32(txt_embeddings)
    if METRIC.upper() in ("COSINE", "IP"):
        img_vecs = _l2_normalize(img_vecs, axis=1)
        txt_vecs = _l2_normalize(txt_vecs, axis=1)

    img_ids = np.arange(img_vecs.shape[0], dtype=np.int64)
    txt_ids = np.arange(txt_vecs.shape[0], dtype=np.int64)

    t2i_col.insert([img_ids, img_vecs])
    i2t_col.insert([txt_ids, txt_vecs])
    print("✓ Insert Data")

    _create_diskann_index(t2i_col)
    _create_diskann_index(i2t_col)
    print("✓ Create Index")

    t2i_col.load()
    i2t_col.load()
    print("✓ Load Memory")

# ===== search helpers =====
def i2t_search(query_vec: np.ndarray, topk: int = SEARCH_TOPK):
    res = i2t_col.search(
        data=query_vec,
        anns_field="vec",
        param={"search_list": max(SEARCH_LIST, topk)},
        limit=topk,
        output_fields=[]
    )
    hits = res[0]
    print(f"[i2t] ids={hits.ids}, distances={hits.distances}")

def t2i_search(query_vec: np.ndarray, topk: int = SEARCH_TOPK):
    res = t2i_col.search(
        data=query_vec,
        anns_field="vec",
        param={"search_list": max(SEARCH_LIST, topk)},
        limit=topk,
        output_fields=[]
    )
    hits = res[0]
    print(f"[t2i] ids={hits.ids}, distances={hits.distances}")

# ===== teardown =====
def end():
    for col in (t2i_col, i2t_col):
        try:
            col.release()
        finally:
            if utility.has_collection(col.name):
                utility.drop_collection(col.name)

# ===== main =====
if __name__ == "__main__":
    init_milvus()

    # 예시: 쿼리 벡터가 이미 계산되어 있다고 가정
    random_query = np.random.rand(1, DIM).astype(np.float32)
    random_query = _l2_normalize(random_query)

    t2i_search(random_query, topk=3)
    i2t_search(random_query, topk=3)

    end()
