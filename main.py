from openai import OpenAI
import base64
import pygame
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("TOKEN"))

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

fs = 44100  # Sample rate

print("Recording started. Press Ctrl+C to stop the recording.")
myrecording = sd.InputStream(samplerate=fs, channels=2)
myrecording.start()
recording = []
try:
    while True:
        data = myrecording.read(int(fs))
        recording.append(data[0])
except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    myrecording.stop()
    myrecording.close()
    recording = np.concatenate(recording, axis=0)
    wav.write('output.wav', fs, recording)  # Save as WAV file

audio = open('output.wav', 'rb')

transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio,
    response_format="text"
)

print(transcript)

returned = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You're Hannah. You act like a human at all costs and in all situations."},
        {"role": "user", "content": transcript}
    ]
)

resp = returned.choices[0].message.content
print(resp)

response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=resp
)

response.stream_to_file("response.mp3")

pygame.mixer.init()
pygame.mixer.music.load('response.mp3')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

