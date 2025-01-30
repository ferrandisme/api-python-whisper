from typing import Union
from fastapi import FastAPI, File, UploadFile
import whisper
import os
from tempfile import NamedTemporaryFile
from pydantic import BaseModel

app = FastAPI()

model = whisper.load_model("tiny", device="cpu", fp16=False)


class TranscriptionResponse(BaseModel):
    transcription: str


@app.post(
    "/api/transcribe/",
    response_model=TranscriptionResponse,
    description="Transcribe audio files to text. Supported formats: wav, mp3, m4a, flac, ogg, opus, and more."
)
async def transcribe_audio(file: UploadFile = File(...)):
    audio = await file.read()
    with NamedTemporaryFile(delete=False, suffix=file.filename) as temp_audio:
        temp_audio.write(audio)
        temp_audio_path = temp_audio.name

    try:
        result = model.transcribe(temp_audio_path)
        transcription = result["text"]
    finally:
        os.remove(temp_audio_path)

    return {"transcription": transcription}
