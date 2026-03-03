FROM paradedb/paradedb:latest

# Install pip
#RUN apt-get update && apt-get install -y python3-pip

USER root

# Fix Debian keyring before apt install
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update --allow-releaseinfo-change \
    && apt-get install -y --no-install-recommends \
       debian-archive-keyring \
       gnupg \
    && apt-get update \
    && apt-get install -y python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip3 install fastapi uvicorn python-multipart

COPY backup_api.py /backup_api.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8001
EXPOSE 5432

ENTRYPOINT ["/start.sh"]
