FROM paradedb/paradedb:latest

# Install pip
RUN apt-get update && apt-get install -y python3-pip

# Install Python deps
RUN pip3 install fastapi uvicorn python-multipart

COPY backup_api.py /backup_api.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8001

ENTRYPOINT ["/start.sh"]
