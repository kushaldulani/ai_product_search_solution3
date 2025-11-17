# -*- coding: utf-8 -*-
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define output structure for language detection
class LanguageDetectionOutput(BaseModel):
    is_english: bool = Field(description="True if text is in English, False otherwise")
    detected_language: str = Field(description="Detected language name (e.g., 'English', 'Spanish', 'French')")
    translated_text: str = Field(description="English translation if non-English, 'N/A' if already English")


# Setup Groq LLM for language detection
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=8192,
    api_key=GROQ_API_KEY
)

# Groq: use json_mode method for structured output
groq_structured_llm = groq_llm.with_structured_output(LanguageDetectionOutput, method="json_mode")


# Language detection and translation prompt
LANGUAGE_DETECT_PROMPT = """You are a language detection and translation expert. Your task is to:
1. Detect if the input text is in English or not
2. Identify the language of the input text
3. If the text is NOT in English, translate it to English
4. If the text IS in English, return 'N/A' for the translation
5. If you are unable to translate, return the original query as it is

Input text: {text}

Instructions:
- Carefully analyze the text to determine if it's in English
- If it's English (including variations like American, British, Australian English), set is_english to true
- If it's any other language, set is_english to false
- For non-English text, provide a fluent, natural English translation
- For English text, set translated_text to 'N/A'
- If the text is corrupted, gibberish, or cannot be translated, return the original text as translated_text
- Always identify the detected language name (use "Unknown" if unable to detect)

Return your response in JSON format with:
- is_english: boolean (true/false)
- detected_language: string (e.g., "English", "Spanish", "French", "Hindi", "Chinese", "Unknown")
- translated_text: string (English translation, "N/A" for English, or original text if unable to translate)
"""


def detect_and_translate(text: str) -> LanguageDetectionOutput:
    """
    Detect if text is English and translate if necessary

    Args:
        text: Input text in any language

    Returns:
        LanguageDetectionOutput with:
        - is_english: True/False
        - detected_language: Language name
        - translated_text: English translation or 'N/A'
    """
    formatted_prompt = LANGUAGE_DETECT_PROMPT.format(text=text)

    response = groq_structured_llm.invoke(formatted_prompt)
    return response


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Hello, how are you today?", True),  # English
        ("Hola, �c�mo est�s?", False),  # Spanish
        ("Bonjour, comment allez-vous?", False),  # French
        (".H  >(> ,(> 09> 9B", False),  # Hindi
        ("�)�Zm", False),  # Chinese
        ("This is a test message", True),  # English
        ("Guten Tag, wie geht es Ihnen?", False),  # German
        ("S�kaoCgYK", False),  # Japanese
    ]

    print("=== Language Detection & Translation Test Cases ===\n")

    for text, expected_is_english in test_cases:
        start = time.time()
        result = detect_and_translate(text)
        end = time.time()

        status = "" if result.is_english == expected_is_english else ""

        print(f"{status} Original: '{text}'")
        print(f"  Is English: {result.is_english} (Expected: {expected_is_english})")
        print(f"  Detected Language: {result.detected_language}")
        print(f"  Translation: {result.translated_text}")
        print(f"  Time: {end - start:.2f}s\n")
