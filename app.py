import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import asyncio
from tempfile import NamedTemporaryFile
from openai import AsyncOpenAI
from config import config



# Set API keys
openai_client = AsyncOpenAI(api_key=config.get("OPENAI_API_KEY"))
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=config.get("GROQ_API_KEY")
)

# Streamlit page configuration
st.set_page_config(
    page_title="Voice-to-LLM Chat",
    layout="centered",
)

# Initialize audio recording
def record_audio(duration=5, samplerate=16000):
    st.info("Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    return audio


def save_audio_to_file(audio, filename="recording.wav", samplerate=16000):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return filename


async def transcribe_audio(audio_file):
    """Transcribe audio using Groq's Whisper model."""
    try:
        with open(audio_file, "rb") as f:
            response = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text"
            )
        return response
    except Exception as e:
        st.error(f"Transcription failed: {str(e)}")
        return None


async def get_chat_response(text):
    """Get chat response using OpenAI's GPT model."""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Change the model name here
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Chat response generation failed: {str(e)}")
        return None



async def generate_speech(text):
    """Generate speech using OpenAI TTS."""
    try:
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        return response.content
    except Exception as e:
        st.error(f"Speech generation failed: {str(e)}")
        return None


# Streamlit UI
st.title("🎤 Voice-to-LLM Chat Application")

# Step 1: Record audio
duration = st.slider("Select Recording Duration (seconds):", min_value=1, max_value=10, value=5)
if st.button("Record"):
    audio_data = record_audio(duration=duration)
    audio_file_path = save_audio_to_file(audio_data)

    st.audio(audio_file_path, format="audio/wav")
    st.success(f"Audio recorded and saved to {audio_file_path}")

    # Step 2: Transcribe audio
    st.info("Transcribing audio...")
    transcription = asyncio.run(transcribe_audio(audio_file_path))
    if transcription:
        st.write(f"**Transcription:** {transcription}")

        # Step 3: Get chat response
        st.info("Generating chat response...")
        chat_response = asyncio.run(get_chat_response(transcription))
        if chat_response:
            st.write(f"**Chat Response:** {chat_response}")

            # Step 4: Generate speech
            st.info("Generating audio response...")
            speech_audio = asyncio.run(generate_speech(chat_response))
            if speech_audio:
                # Save speech audio
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                    temp_audio_file.write(speech_audio)
                    temp_audio_path = temp_audio_file.name

                # Play back generated audio
                st.audio(temp_audio_path, format="audio/wav")
                st.success("Audio response generated!")
