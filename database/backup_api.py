from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import subprocess
import uuid
import os

app = FastAPI()

PG_CONN_STR = "postgresql://postgres@localhost:5432/postgres"
BACKUP_DIR = "/tmp"

@app.post("/backup")
def backup():
    backup_id = str(uuid.uuid4())
    path = f"{BACKUP_DIR}/{backup_id}.dump"

    subprocess.run([
        "pg_dump",
        "-Fc",
        PG_CONN_STR,
        "-f", path
    ], check=True)

    return {"backup_id": backup_id}

@app.get("/backup/{backup_id}")
def download_backup(backup_id: str):
    path = f"{BACKUP_DIR}/{backup_id}.dump"
    return FileResponse(path, filename="backup.dump")

@app.post("/restore")
def restore(file: UploadFile):
    path = f"{BACKUP_DIR}/restore.dump"
    with open(path, "wb") as f:
        f.write(file.file.read())

    subprocess.run([
        "pg_restore",
        "--clean",
        "--if-exists",
        "-d", PG_CONN_STR,
        path
    ], check=True)

    return {"status": "restored"}
