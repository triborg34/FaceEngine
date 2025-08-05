import io
import json
import logging
import os
import shutil
import socket
import time
import webbrowser
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from fastapi import FastAPI, File, Query, Response, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import uvicorn
from engine import generate_frames, graceful_shutdown,imageSearcher,imageCrop, loadConfig, loadDb, loadModel
from onvifmaneger import get_rtsp_url
from savatoDb import reciveFromUi

class RtspFields(BaseModel):
    ip:str
    port:str
    username:str
    password:str


class KnownPersonFields(BaseModel):
    name:str
    gender:str
    imagePath:str
    age:str
    role:str
    socialnumber:str
    

app = FastAPI()


origins = ["*"]  # Change this to specific domains in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH",
                   "DELETE"],  # Allowed HTTP methods
    allow_headers=["Origin", "X-Requested-With",
                   "Content-Type", "Accept"],  # Allowed headers
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    graceful_shutdown()
    logging.info("Application shutdown, resources released")


@app.get("/{camera_id}")
async def video_feed(camera_id: str, request: Request, source: str = Query(...)):
    if source == '0':
        source = int(source)
    """Stream video from a specific camera"""
    if not camera_id.startswith("rt"):
        return Response("Invalid camera ID format. Use rt1, rt2, etc.", status_code=400)

    try:
        # Extract camera index from ID (rt1 -> 1)
        camera_idx = int(camera_id[2:])

        # Check if camera index is valid

        return StreamingResponse(

            generate_frames(camera_idx, source, request),


            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store"
            }
        )
    except ValueError:
        return Response(f"Invalid camera ID: {camera_id}. Use format: rt1, rt2, etc.",
                        status_code=400)


def discover_onvif_stream():
    ip_base = "192.168.1"

    def event_generator():
        for i in range(1, 255):
            ip = f"{ip_base}.{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.3)
                result = sock.connect_ex((ip, 80))
                if result == 0:
                    yield f"data: {json.dumps({'ip': ip, 'port': 80})}\n\n"
                sock.close()
            except:
                continue
            time.sleep(0.1)
    return event_generator()


@app.get("/onvif/get-stream")
def get_camera_stream():
    return StreamingResponse(discover_onvif_stream(), media_type="text/event-stream")


@app.post('/onvif/get-rtsp')
async def get_camra_rtsp(request:RtspFields):
    print(request.ip)
    request.port=int(request.port)
    rtspUrl=get_rtsp_url(request.ip,request.port,request.username,request.password)
    return {'rtsp':rtspUrl}
    
    


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    UPLOAD_DIR_VIDEO = "uploads"
    os.makedirs(UPLOAD_DIR_VIDEO, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR_VIDEO, file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(file_location)
    img_encoded = imageCrop(file_location)
    
    # Convert image to base64
    import base64
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
    
    return {
        "file_location": file_location,
        "image_data": img_base64,
        "media_type": "image/jpeg"
    }
    




@app.post("/insertKToDp")
async def insertKtoDp(data:KnownPersonFields):
    if(data.imagePath.startswith('http://')):
        isurl=True
    else:
        isurl=False

    try:
        reciveFromUi(data.name,data.imagePath,data.age,data.gender,data.role,data.socialnumber,isurl)
        return Response("successful",200)
    except Exception as e:
        return Response(e,400)
    
    
    

if __name__ == "__main__":
    loadDb()
    host = '0.0.0.0'
    # port = int(loadConfig())
    port=8000

    loader = loadModel()
    if (loader):

        # webbrowser.open(f'http://127.0.0.1:{port}/web/app')
        uvicorn.run("app:app", log_level='info',log_config=None,
                    reload=False)