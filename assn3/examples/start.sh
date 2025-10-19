mkdir $(pwd)/milvus
docker run -d --name milvus-standalone \
  --security-opt seccomp:unconfined \
  -e ETCD_USE_EMBED=true \
  -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
  -e COMMON_STORAGETYPE=local \
  -e DEPLOY_MODE=STANDALONE \
  -v $(pwd)/milvus:/var/lib/milvus \
  -p 19530:19530 -p 9091:9091 -p 2379:2379 \
  --health-cmd="curl -f http://localhost:9091/healthz" \
  --health-interval=30s --health-start-period=90s --health-timeout=20s --health-retries=3 \
  milvusdb/milvus:v2.4.9 milvus run standalone
