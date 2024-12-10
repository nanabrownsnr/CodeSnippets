import os
from grok import check_compliance
from stt import speech_to_text
import json
import uvicorn
from fastapi import FastAPI, File, UploadFile
import shutil
audio_file4 = "audio_files\Call-Center-Sample-Recordings--Magellan-Solutions (4).mp3"

app = FastAPI(title="Voice Compliance Agent")


@app.post("/transcribe-audio/")
async def transcribe_audio(audio_file: UploadFile = File(...)) -> dict:

    try:
        save_path = f"./uploaded_files/{audio_file.filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)

        text = speech_to_text(save_path)
        if text:
            print("Call transcript: ",text)
            verdict = check_compliance(text)
            print("consumer clearly articulated their understanding of product and process: ",verdict)
        return {"text": text}
        
    except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
        print(f"Error processing emotion: {e}")
        return {"error": str(e)}


@app.post("/analyze-text/")
async def analyze_text(transcribed_test: str) -> dict:
    try:
        verdict = check_compliance(text)
        print("consumer clearly articulated their understanding of product and process: ",verdict)
        return {"verdict": verdict}
        
    except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
        print(f"Error processing emotion: {e}")
        return {"error": str(e)}
        
        
        
if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",  # Replace "main" with the module name where your app is defined
        host="0.0.0.0",
        port= 8500,  # Replace port with your desired port number
        reload=True  # Optional: Enables automatic reloading on file changes
    )
