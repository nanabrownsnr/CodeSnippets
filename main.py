from grok import check_customer_understanding, check_compliance
from stt import speech_to_text
import json

# #English
# audio_file1 = "audio_files\Call-Center-Sample-Recordings--Magellan-Solutions (1).mp3"
# audio_file2 = "audio_files\Call-Center-Sample-Recordings--Magellan-Solutions (2).mp3"
# audio_file3 = "audio_files\Call-Center-Sample-Recordings--Magellan-Solutions (4).mp3"
audio_file4 = "audio_files\Call-Center-Sample-Recordings--Magellan-Solutions (4).mp3"

#German
audio_file5 = "audio_files\Delivery--Logistics-Call-Center-Speech-Data-German-(Germany).mp3"


def check_voice_recording(audio):
    try:
        text = speech_to_text(audio)
        if text:
            print("Call transcript: ",text)
            verdict = check_compliance(text)
            print("consumer clearly articulated their understanding of product and process: ",verdict)

    except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
        print(f"Error processing emotion: {e}")


check_voice_recording(audio_file4)