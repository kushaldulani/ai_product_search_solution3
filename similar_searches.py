import time
import os
from pydantic import BaseModel, Field, List
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from prompts import similar_searches

from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Define output structure
class SimilarSearches(BaseModel):
    searches: List['str'] = Field(description="similar searches related to the input query")

parser = PydanticOutputParser(pydantic_object=SimilarSearches)

# Setup

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low",
    reasoning_format="hidden",
    api_key=GROQ_API_KEY
    )