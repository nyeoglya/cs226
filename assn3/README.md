## Introduction
This is the *CSED226 Intro. to Data Analysis* **assignment 3** project folder created by 20240505 Hyunseong Kong (공현성).

## Quick Start
### 1. Create conda env
```shell
conda create --name milvus python=3.11 -y
conda activate milvus
```

### 2. Install necessarily packages
```shell
pip install "pymilvus==2.4.9" "marshmallow>=3.13,<4" "environs>=9.5,<12" "grpcio>=1.59,<2"
```
Make sure the followings are installed: `torch, pillow, transformers, tqdm, flask`.

#### 2-1. Use requirements.txt
```shell
pip install -r requirements.txt
```

### 3. Docker initialization
Run docker.
```shell
./start.sh
```

### 4. Run
```shell
python init_milvus.py
python app.py
```
