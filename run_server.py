import uvicorn
from api_server import app

if __name__ == "__main__":
    print("Starting Dental_Panoramic_Reader FastAPI Gateway on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
