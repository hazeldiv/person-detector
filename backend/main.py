"""
FastAPI backend for person detection via WebSocket.
Receives JPEG frames (640x480 letterboxed), runs HOG+MOG2 detection,
returns bounding boxes as JSON.
"""

import json
import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from detector import PersonDetector

app = FastAPI(title="Person Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    detector = PersonDetector()
    print("[WS] Client connected")

    frame_count = 0
    fps = 0.0
    start_time = time.time()

    try:
        while True:
            message = await ws.receive()

            # Handle text messages (config commands)
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "set_line":
                    detector.set_detection_line(
                        points=data["points"],
                        side_parity=data["side_parity"],
                    )
                    await ws.send_json({"type": "line_set", "status": "ok"})
                    print(f"[WS] Detection polyline set: {len(data['points'])} pts, parity={data['side_parity']}")

                elif msg_type == "clear_line":
                    detector.clear_detection_line()
                    await ws.send_json({"type": "line_cleared", "status": "ok"})
                    print("[WS] Detection line cleared")

                continue

            # Handle binary messages: [4 bytes frameId uint32 BE] + [JPEG bytes]
            if "bytes" in message:
                raw = message["bytes"]
                if len(raw) < 4:
                    continue

                # Extract frameId from first 4 bytes
                frame_id = int.from_bytes(raw[:4], byteorder="big")
                jpeg_bytes = raw[4:]

                np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                # Run detection
                boxes = detector.detect(frame)

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    start_time = time.time()

                # Send back bounding boxes + echo frameId for client sync
                await ws.send_json(
                    {
                        "type": "detections",
                        "frameId": frame_id,
                        "boxes": boxes,
                        "fps": round(fps, 1),
                    }
                )

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await ws.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
