"""
OpenAI-Compatible REST API server for Ghana Voice.
Allows any standard OpenAI TTS client or frontend to integrate seamlessly.
"""

import io
import soundfile as sf
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    FastAPI = None

from .constants import SAMPLE_RATE_24K
from .engine import GhanaVoice

_ENGINES = {}


def create_app():
    if FastAPI is None:
        raise ImportError("FastAPI is required for the REST server. Run: pip install fastapi uvicorn")

    app = FastAPI(
        title="Ghana Voice API",
        description="OpenAI-compatible text-to-speech API for Ghanaian and African languages",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class SpeechRequest(BaseModel):
        input: str
        model: Optional[str] = "ghana-voice-base"
        voice: Optional[str] = "twi"
        response_format: Optional[str] = "wav"
        speed: Optional[float] = 1.0

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "ghana-voice"}

    @app.post("/v1/audio/speech")
    def text_to_speech(req: SpeechRequest):
        lang = req.voice or "twi"
        if lang not in _ENGINES:
            _ENGINES[lang] = GhanaVoice(lang=lang, model=req.model)

        engine = _ENGINES[lang]
        try:
            wav = engine.synthesize(req.input)
            buf = io.BytesIO()
            sf.write(buf, wav, SAMPLE_RATE_24K, format="WAV")
            buf.seek(0)
            return Response(content=buf.read(), media_type="audio/wav")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def start_server(host: str = "0.0.0.0", port: int = 8000):
    try:
        import uvicorn
    except ImportError:
        raise ImportError("Uvicorn is required to run the server. Run: pip install uvicorn")

    app = create_app()
    uvicorn.run(app, host=host, port=port)
