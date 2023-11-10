from openai import OpenAI
import base64
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import time

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

client = OpenAI(api_key=os.getenv("TOKEN"))

jarvis = client.beta.assistants.retrieve(
    assistant_id="asst_pTUi0R7FgyGPpLKm5QpLhnOF"
)

print(jarvis)

thread = client.beta.threads.create()

fs = 44100

try:
    while True:

        print("Listening ...")
        myrecording = sd.InputStream(samplerate=fs, channels=2)
        myrecording.start()
        recording = []
        try:
            while True:
                data = myrecording.read(int(fs))
                recording.append(data[0])
        except KeyboardInterrupt:
            print("Done listening!")
        finally:
            myrecording.stop()
            myrecording.close()
            recording = np.concatenate(recording, axis=0)
            wav.write('output.wav', fs, recording)

        audio = open('output.wav', 'rb')

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio,
            response_format="text"
        )

        print(transcript)

        msg = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=transcript
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=jarvis.id,
            instructions="Your response will be converted to speech, so avoid links and other non-speech content."
        )

        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            print("Stand by ...")
            time.sleep(2)
        msgs = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        recent = max(msgs.data, key=lambda msg: msg.created_at)
        msg = recent.content[0].text.value
        print(msg)

        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=msg
        )

        response.stream_to_file("response.mp3")

        pygame.mixer.init()
        pygame.mixer.music.load('response.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() == True:
            continue

except KeyboardInterrupt:
    print("Goodbye!")