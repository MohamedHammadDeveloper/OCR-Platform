# OCR Platform

One FastAPI application serving multiple OCR models. Each model keeps its own
preprocessing, prompt, processor, generation parameters, inference code and
post-processing — nothing is shared between them.

## Layout

```
app.py                              FastAPI app: /ocr, /models, /health
registry.py                         AVAILABLE_MODELS -> service mapping
config.py                           app-level settings only
services/
    arabic_handwritten/
        config.py
        download_model.py
        service.py                  copied from Arabic-English-handwritten-OCR-v3
    legal_documents/
        config.py
        download_model.py
        service.py                  copied from arabic-legal-documents-ocr-1.0
```

## Model storage

```
/workspace/models/
    arabic-handwritten/Arabic-English-handwritten-OCR-v3/
    legal-documents/arabic-legal-documents-ocr-1.0/
```

Models never overwrite each other. Override the root with `OCR_MODEL_ROOT`.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env        # set HF_TOKEN
python app.py               # or: uvicorn app:app --host 0.0.0.0 --port 10100
```

Nothing is downloaded or loaded at startup.

## Endpoints

### `POST /ocr`

Multipart upload. Query params:

| param            | notes                                                      |
| ---------------- | ---------------------------------------------------------- |
| `model`          | `arabic-handwritten` or `legal-documents`. Defaults to `OCR_DEFAULT_MODEL`. |
| `task`           | `task_1` / `task_2` — legal-documents only, ignored by the other |
| `preprocess`     | legal-documents only, ignored by the other                  |
| `max_new_tokens` | legal-documents only, ignored by the other                  |

```bash
curl -X POST "http://localhost:10100/ocr?model=arabic-handwritten" -F "image=@page.png"
curl -X POST "http://localhost:10100/ocr?model=legal-documents&task=task_2" -F "image=@doc.jpg"
```

The model is loaded automatically on the first request that needs it and stays
in memory afterwards.

**The response is unchanged from each original project.**

`arabic-handwritten`:

```json
{ "success": true, "text": "...", "data": { "output": { "full_text": "..." } } }
```

`legal-documents`:

```json
{ "success": true, "text": "...", "data": { } }
```

### `POST /ocr/file`

Same, with a JSON body: `{"path": "...", "model": "...", "task": "...", ...}`

### `GET /models`

```json
[
  { "name": "arabic-handwritten", "downloaded": true, "loaded": true,  "device": "cuda:0" },
  { "name": "legal-documents",    "downloaded": true, "loaded": false, "device": null }
]
```

### `POST /models/{model}/download`

Returns immediately if already present, otherwise runs that model's own
`download_model.py`.

### `POST /models/{model}/unload`

Drops the model from memory and calls `torch.cuda.empty_cache()`.

Each `download_model.py` also runs standalone:

```bash
python -m services.legal_documents.download_model
python -m services.arabic_handwritten.download_model
```

## Adding a model

1. `mkdir services/new_model/` (with an `__init__.py`)
2. Copy that project's `service.py` — expose `is_downloaded()`, `download()`,
   `load()`, `unload()`, `is_loaded`, `device()` and
   `run(image_path, task, preprocess, max_new_tokens) -> dict`
3. Copy its `download_model.py`
4. Add one line to `AVAILABLE_MODELS` in `registry.py`

No endpoint or routing code changes.
