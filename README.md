# parse-and-search

## Running the Converter locally for MPS enabled MacOCR

### Prerequisites
- Python with pip

### Installing dependencies
```
pip install -r frontend/requirements.txt
pip install -r converter/requirements.txt
```

### Running the frontend from terminal
In one shell session:
```
export CONVERTER_PIPELINE_TYPE=macocr
cd frontend
streamlit run streamlit_app.py --server.port=8501
```
In another shell session:
```
cd converter/app/
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Then, open a browser and go to localhost:8501



