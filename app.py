from fastapi import FastAPI,Query,Response,Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response,StreamingResponse
from engine import generate_frames


app = FastAPI()
rtsp=['rtsp://192.168.1.245:554/stream']

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



@app.get("/{camera_id}")
async def video_feed(camera_id: str,request: Request, source: str = Query(...)):
    if source =='0':
        source=int(source)
    """Stream video from a specific camera"""
    if not camera_id.startswith("rt"):
        return Response("Invalid camera ID format. Use rt1, rt2, etc.", status_code=400)

    try:
        # Extract camera index from ID (rt1 -> 1)
        camera_idx = int(camera_id[2:])

        # Check if camera index is valid
        if camera_idx < 1 or camera_idx > len(rtsp):
            return Response(f"Camera {camera_id} not found. Valid range: rt1-rt{len(rtsp)}",
                            status_code=404)

        return StreamingResponse(

            generate_frames(camera_idx,source,request),


            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store"
            }
        )
    except ValueError:
        return Response(f"Invalid camera ID: {camera_id}. Use format: rt1, rt2, etc.",
                        status_code=400)