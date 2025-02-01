from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel  # o la librería que estés utilizando
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

app = FastAPI()

model = WhisperModel("tiny", device="cpu", compute_type="int8")

templates = Jinja2Templates(directory="templates")


class TranscriptionResponse(BaseModel):
    transcription: str


@app.post("/api/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    audio = await file.read()
    extension = os.path.splitext(file.filename)[1]
    with NamedTemporaryFile(delete=False, suffix=extension) as temp_audio:
        temp_audio.write(audio)
        temp_audio_path = temp_audio.name

    try:
        # Llamada al modelo de transcripción (puedes ajustar los parámetros)
        segments, info = model.transcribe(temp_audio_path, beam_size=5)
        transcription = " ".join([segment.text for segment in segments])
    finally:
        os.remove(temp_audio_path)

    return {"transcription": transcription}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
