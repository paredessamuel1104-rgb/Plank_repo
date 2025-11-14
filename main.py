import os
import joblib
import numpy as np
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

app = FastAPI()

# Permitir CORS para desarrollo y producción
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y scaler al iniciar
MODEL_PATH = os.path.join("model", "plank_classifier_model.pkl")
SCALER_PATH = os.path.join("model", "plank_scaler.pkl")
LABELS_PATH = os.path.join("model", "plank_label_encoder.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABELS_PATH)

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            t0 = time.perf_counter()
            data = await websocket.receive_json()
            t1 = time.perf_counter()
            features = np.array(data["features"]).reshape(1, -1)
            features_scaled = scaler.transform(features)
            proba = model.predict_proba(features_scaled)[0].tolist()
            pred_idx = int(np.argmax(proba))
            pred_label = label_encoder.classes_[pred_idx]
            t2 = time.perf_counter()
            await websocket.send_json({
                "proba": proba,
                "pred": pred_label,
                "pred_idx": pred_idx,
                "timing": {
                    "receive": round((t1-t0)*1000, 2),
                    "inference": round((t2-t1)*1000, 2),
                    "total": round((t2-t0)*1000, 2)
                }
            })
    except WebSocketDisconnect:
        pass
