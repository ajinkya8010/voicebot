# Voice Assistant Application

## Prerequisites
- Python 3.9+
- OpenAI API Key
- Groq API Key

## Setup
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `config.py` file with your API keys:
   ```
   config = {
    "OPENAI_API_KEY": "your-openai-key",
    "GROQ_API_KEY": "your-groq-key"
   }
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

## Features
- Real-time voice activity detection
- Audio transcription using Groq Whisper
- AI-powered chat responses
- Text-to-speech responses
- History of conversation preserved



