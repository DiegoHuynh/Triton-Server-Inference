import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Cập nhật đường dẫn chính xác đến file cấu hình hf.json
HF_CONFIG_FILE = "C:/Users/Duc Huynh/Triton-Server-Inference/tritonserver/configs/hf.json"
HF_MODEL_REPO = "/models"

def check_config_path():
    file = Path(HF_CONFIG_FILE).expanduser().resolve()
    if file.is_file():
        print(f"Config file found at: {file}")
    else:
        print(f"Config file not found! Please check the path: {file}")

if __name__ == "__main__":
    # Kiểm tra file cấu hình
    check_config_path()
    
    file = Path(HF_CONFIG_FILE).expanduser().resolve()
    if not file.is_file():
        print("No huggingface config found!")
    else:
        repo = Path(HF_MODEL_REPO)
        repo.mkdir(parents=True, exist_ok=True)
        
        # Đọc và xử lý cấu hình
        with file.open("r") as f:
            conf = json.load(f)
        
        # Lấy token từ config
        token = conf.get("token", None)
        models = conf.get("models", [])
        
        # Tải model từ Hugging Face
        for model in models:
            _name = model.get("name", None)
            assert _name is not None, "Invalid huggingface config! Model name cannot be none!"
            _token = model.get("token", token)
            _ref = model.get("ref", None)
            snapshot_download(repo_id=_name, revision=_ref, token=_token, local_dir=repo, ignore_patterns=[".*"])
        
        # Xóa cache
        cache = Path(repo, ".cache")
        shutil.rmtree(cache)
