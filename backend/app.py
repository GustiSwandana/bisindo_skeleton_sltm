from __future__ import annotations

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.inference import SignPredictor


app = FastAPI(
    title="BISINDO Skeleton + LSTM API",
    version="1.0.0",
    description="API prediksi alfabet BISINDO berbasis landmark tangan MediaPipe dan LSTM.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = SignPredictor()


@app.get("/health")
def health_check() -> dict:
    return {
        "success": True,
        "model_ready": predictor.model_loaded,
        "message": "API aktif." if predictor.model_loaded else "API aktif, model belum dilatih.",
    }


@app.get("/labels")
def get_labels() -> JSONResponse:
    if not predictor.model_loaded:
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": "Model belum tersedia."},
        )
    return JSONResponse(content={"success": True, "labels": predictor.class_names})


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    if not predictor.model_loaded:
        return JSONResponse(
            status_code=503,
            content={"success": False, "message": "Model belum tersedia. Jalankan training dulu."},
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "File harus berupa gambar."},
        )

    image_bytes = await file.read()
    result = predictor.predict_bytes(image_bytes)

    status_code = 200 if result.success else 422
    return JSONResponse(
        status_code=status_code,
        content={
            "success": result.success,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "top_predictions": result.top_predictions,
            "landmarks": result.landmarks,
            "skeleton_image_base64": result.skeleton_image_base64,
            "message": result.message,
        },
    )
