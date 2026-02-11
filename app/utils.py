import os
import tempfile
from pathlib import Path


def save_uploaded_files(path, uploaded_files):
    paths = []

    for uploaded_file in uploaded_files:
        save_path = Path(path) / (uploaded_file.name)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        paths.append(save_path)

    return paths