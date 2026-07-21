"""
OCR Platform — one FastAPI application serving multiple OCR models.

No model is downloaded or loaded at startup. A model is loaded on its first
OCR request and then reused for every following request.

The /ocr response is produced by the selected model's own service and is
byte-identical to what that model's original project returned.
"""

import logging
import os
import shutil
import uuid
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

import config
import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ocr.api")

app = FastAPI(
    title="OCR Platform",
    description="Multi-model OCR service.",
    version=config.SERVICE_VERSION,
)

os.makedirs(config.UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Errors
#
# The two original projects returned different error bodies:
#   handwritten -> {"success": false, "message": ..., "details": ...}
#   legal       -> {"success": false, "error": ...}
# Both key sets are emitted so neither existing client breaks.
# --------------------------------------------------------------------------- #

def _error_response(status_code: int, message: str, details: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": message,
            "message": message,
            "details": details,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        return _error_response(
            exc.status_code, detail.get("message", ""), detail.get("details", "")
        )
    return _error_response(exc.status_code, str(detail))


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error_response(422, "Invalid request.", str(exc.errors()))


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    return _error_response(500, "OCR processing failed.", f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Routing helper
# --------------------------------------------------------------------------- #

def _resolve(model: Optional[str]):
    """Pick the service for ``model``, loading its weights if not yet loaded."""
    name = model or config.DEFAULT_MODEL
    service = registry.get(name)

    if service is None:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unknown model '{name}'.",
                "details": f"Available models: {sorted(registry.AVAILABLE_MODELS)}",
            },
        )

    if not service.is_loaded:
        logger.info("Model '%s' not loaded — loading now.", name)
        service.load()

    return service


def _run(service, image_path, task, preprocess, max_new_tokens) -> dict:
    try:
        return service.run(
            image_path,
            task=task,
            preprocess=preprocess,
            max_new_tokens=max_new_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("OCR failed for %s", image_path)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def root():
    return {"service": config.SERVICE_NAME, "status": "running"}


@app.get("/health")
def health():
    cuda = torch.cuda.is_available()
    return {
        "status": "healthy",
        "cuda": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


# --------------------------------------------------------------------------- #
# Model management
# --------------------------------------------------------------------------- #

@app.get("/models")
def list_models():
    return registry.status()


@app.post("/models/{model}/download")
def download_model(model: str):
    service = registry.get(model)
    if service is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown model '{model}'."
        )

    if service.is_downloaded():
        return {"success": True, "model": model, "downloaded": True, "message": "Already downloaded."}

    service.download()
    return {"success": True, "model": model, "downloaded": True, "message": "Download complete."}


@app.post("/models/{model}/unload")
def unload_model(model: str):
    service = registry.get(model)
    if service is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown model '{model}'."
        )

    service.unload()
    return {"success": True, "model": model, "loaded": False}


# --------------------------------------------------------------------------- #
# OCR — one endpoint, routed by `model`
# --------------------------------------------------------------------------- #

@app.post("/ocr")
def ocr(
    image: UploadFile = File(..., description="Image file to run OCR on."),
    model: Optional[str] = Query(None, description="Which OCR model to use."),
    task: Optional[str] = Query(None),
    preprocess: Optional[bool] = Query(None),
    max_new_tokens: Optional[int] = Query(None),
):
    """Run OCR on an uploaded image. The temporary file is always removed."""
    service = _resolve(model)

    suffix = os.path.splitext(image.filename or "")[1] or ".jpg"
    temp_path = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")

    try:
        with open(temp_path, "wb") as dest:
            shutil.copyfileobj(image.file, dest)
        return _run(service, temp_path, task, preprocess, max_new_tokens)
    finally:
        image.file.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)


class OCRFileRequest(BaseModel):
    path: str
    model: Optional[str] = None
    task: Optional[str] = None
    preprocess: Optional[bool] = None
    max_new_tokens: Optional[int] = None


@app.post("/ocr/file")
def ocr_file(request: OCRFileRequest):
    """Run OCR on an image that already exists on the server's filesystem."""
    service = _resolve(request.model)

    if not os.path.isfile(request.path):
        raise HTTPException(
            status_code=404,
            detail={"message": "File not found.", "details": request.path},
        )

    return _run(
        service, request.path, request.task, request.preprocess, request.max_new_tokens
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=config.HOST, port=config.PORT)
