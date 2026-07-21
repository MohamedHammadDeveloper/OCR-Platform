"""
Model registry.

Maps a model name to its service and nothing else. No inference logic lives
here. To add a model: create ``services/<new_model>/``, copy its ``service.py``
and ``download_model.py``, then add one line to ``AVAILABLE_MODELS``.
"""

from services.arabic_handwritten.service import ArabicHandwrittenService
from services.legal_documents.service import LegalDocumentsService

AVAILABLE_MODELS = {
    "arabic-handwritten": ArabicHandwrittenService,
    "legal-documents": LegalDocumentsService,
}

# One long-lived instance per model. Constructing a service does NOT load or
# download anything — the weights are only touched by load() / download().
_SERVICES = {name: cls() for name, cls in AVAILABLE_MODELS.items()}


def get(name: str):
    """Return the service for ``name``, or None if the model is unknown."""
    return _SERVICES.get(name)


def status() -> list:
    """Downloaded / loaded / device for every registered model."""
    return [
        {
            "name": name,
            "downloaded": service.is_downloaded(),
            "loaded": service.is_loaded,
            "device": service.device(),
        }
        for name, service in _SERVICES.items()
    ]
