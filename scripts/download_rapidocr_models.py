import os
from modelscope import snapshot_download

model_dir = os.environ.get("RAPIDOCR_DOWNLOAD_DIR", "/models")

os.makedirs(model_dir, exist_ok=True)

if not os.listdir(model_dir):
    print("Downloading RapidOCR models...")
    snapshot_download(
        repo_id="RapidAI/RapidOCR",
        cache_dir=model_dir
    )
else:
    print("Models already present, skipping download.")
