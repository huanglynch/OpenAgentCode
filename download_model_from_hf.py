from huggingface_hub import snapshot_download

# 指定模型 ID 和本地保存路径
repo_id = "sentence-transformers/all-mpnet-base-v2"
local_dir = "./cached_models/sentence-transformers_all-mpnet-base-v2"  # 自定义路径，例如您的项目中的 cached_models/

snapshot_download(repo_id=repo_id, local_dir=local_dir)