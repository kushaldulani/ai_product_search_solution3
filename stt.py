import os
from typing import Literal
from pydantic import BaseModel, Field
from deepgram import DeepgramClient
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from groq import Groq
from fastapi import HTTPException

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# STT Model types
STTModel = Literal["deepgram", "groq"]


# Define output structure for translation
class TranslationOutput(BaseModel):
    translated_text: str = Field(description="English translation of the input text")
    detected_language: str = Field(description="Detected source language of the input")


# Initialize clients
deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Setup Groq LLM for translation
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=8192,
    api_key=GROQ_API_KEY
) if GROQ_API_KEY else None

# Groq: use json_mode method for structured output
groq_structured_llm = groq_llm.with_structured_output(TranslationOutput, method="json_mode") if groq_llm else None


# Translation prompt template
TRANSLATION_PROMPT = """You are a professional translator. Your task is to translate the given text to English.

Input text: {text}

Instructions:
- Translate the text to fluent, natural English
- Preserve the meaning and context
- Detect and identify the source language
- If the text is already in English, return it as-is

Return your response in JSON format with:
- translated_text: the English translation
- detected_language: the detected source language (e.g., "Spanish", "French", "Hindi", "English")
"""


def translate_to_english(text: str) -> TranslationOutput:
    """
    Translate text to English using Groq LLM

    Args:
        text: Text in any language

    Returns:
        TranslationOutput with translated text and detected language
    """
    if not groq_structured_llm:
        raise HTTPException(status_code=500, detail="Groq API key not configured")

    formatted_prompt = TRANSLATION_PROMPT.format(text=text)

    messages = [
        SystemMessage(content="You are a professional translator. Always respond in valid JSON format."),
        HumanMessage(content=formatted_prompt)
    ]

    response = groq_structured_llm.invoke(messages)
    return response


def transcribe_with_deepgram(audio_bytes: bytes) -> dict:
    """
    Transcribe audio using Deepgram with automatic language detection

    Args:
        audio_bytes: Audio file bytes

    Returns:
        dict with transcript, detected_language, confidence
    """
    if not deepgram:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured")

    try:
        # Transcribe using the listen.v1.media API with new SDK 5.x
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model="nova-2",  # Fast & accurate
            detect_language=True,  # Auto-detect language (adds ~200-300ms)
            punctuate=True,
            smart_format=False,  # Disabled for speed
        )

        transcript = response.results.channels[0].alternatives[0].transcript
        detected_language = response.results.channels[0].detected_language
        confidence = response.results.channels[0].alternatives[0].confidence

        return {
            "transcript": transcript,
            "detected_language": detected_language,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deepgram transcription failed: {str(e)}")


def transcribe_with_groq_whisper(audio_bytes: bytes, filename: str = "audio.mp3") -> dict:
    """
    Transcribe audio using Groq's Whisper Large v3 model

    Args:
        audio_bytes: Audio file bytes
        filename: Original filename (for file type detection)

    Returns:
        dict with transcript, detected_language, confidence
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API key not configured")

    try:
        # Groq Whisper API expects a file-like object
        # Create a temporary file-like object from bytes
        from io import BytesIO

        audio_file = BytesIO(audio_bytes)
        audio_file.name = filename  # Set filename for proper content-type detection

        # Transcribe using Groq Whisper
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_file),
            model="whisper-large-v3",
            response_format="verbose_json",  # Get detailed response with language
        )

        return {
            "transcript": transcription.text,
            "detected_language": transcription.language if hasattr(transcription, 'language') else None,
            "confidence": None  # Whisper doesn't provide confidence scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq Whisper transcription failed: {str(e)}")
