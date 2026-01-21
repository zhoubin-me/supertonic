import io
import os
import zipfile
from typing import List, Union

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from helper import AVAILABLE_LANGS, load_text_to_speech, load_voice_style, sanitize_filename


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


ONNX_DIR = os.getenv("TTS_ONNX_DIR", "assets/onnx")
USE_GPU = _env_flag("TTS_USE_GPU", "0")

text_to_speech = load_text_to_speech(ONNX_DIR, USE_GPU)

app = FastAPI(title="Supertonic TTS Service")


class TTSRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="Text to synthesize.")
    lang: Union[str, List[str]] = Field("en", description="Language(s) for text.")
    voice_style: Union[str, List[str]] = Field(
        "assets/voice_styles/M1.json", description="Voice style path(s)."
    )
    total_step: int = Field(5, ge=1, le=50)
    speed: float = Field(1.05, gt=0.0)
    batch: bool = False
    silence_duration: float = Field(
        0.3, ge=0.0, description="Silence between chunks for non-batch mode."
    )


def _ensure_list(value: Union[str, List[str]]) -> List[str]:
    return value if isinstance(value, list) else [value]


def _validate_lengths(texts: List[str], langs: List[str], styles: List[str]) -> None:
    if not (len(texts) == len(langs) == len(styles)):
        raise HTTPException(
            status_code=400,
            detail="text, lang, and voice_style must have the same length.",
        )


def _validate_langs(langs: List[str]) -> None:
    invalid = sorted({lang for lang in langs if lang not in AVAILABLE_LANGS})
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language(s): {', '.join(invalid)}",
        )


def _slice_audio(
    wav: np.ndarray, durations: np.ndarray, sample_rate: int
) -> List[np.ndarray]:
    sliced = []
    for idx in range(wav.shape[0]):
        dur = float(np.atleast_1d(durations)[idx])
        end = int(sample_rate * dur)
        sliced.append(wav[idx, :end])
    return sliced


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/tts")
def synthesize(req: TTSRequest) -> StreamingResponse:
    texts = _ensure_list(req.text)
    langs = _ensure_list(req.lang)
    styles = _ensure_list(req.voice_style)

    if req.batch:
        _validate_lengths(texts, langs, styles)
    else:
        if len(texts) != 1 or len(langs) != 1 or len(styles) != 1:
            raise HTTPException(
                status_code=400,
                detail="Non-batch mode requires single text, lang, and voice_style.",
            )

    _validate_langs(langs)
    style = load_voice_style(styles, verbose=False)

    if req.batch:
        wav, dur = text_to_speech.batch(
            texts, langs, style, req.total_step, req.speed
        )
    else:
        wav, dur = text_to_speech(
            texts[0],
            langs[0],
            style,
            req.total_step,
            req.speed,
            req.silence_duration,
        )

    audio_chunks = _slice_audio(wav, dur, text_to_speech.sample_rate)

    if len(audio_chunks) == 1:
        buf = io.BytesIO()
        sf.write(buf, audio_chunks[0], text_to_speech.sample_rate, format="WAV")
        buf.seek(0)
        filename = sanitize_filename(texts[0], 40) or "tts"
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}.wav"'},
        )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, chunk in enumerate(audio_chunks):
            filename = sanitize_filename(texts[idx], 40) or f"tts_{idx+1}"
            wav_buf = io.BytesIO()
            sf.write(wav_buf, chunk, text_to_speech.sample_rate, format="WAV")
            zf.writestr(f"{filename}.wav", wav_buf.getvalue())
    zip_buf.seek(0)
    return StreamingResponse(
        zip_buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="tts_outputs.zip"'},
    )
