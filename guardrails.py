import time
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from prompts import guardrails_prompt

from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Define output structure
class InputGuardrails(BaseModel):
    allowed: bool = Field(description="True if query relates to allowed categories, False otherwise")

parser = PydanticOutputParser(pydantic_object=InputGuardrails)

# Setup

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low",
    reasoning_format="hidden",
    api_key=GROQ_API_KEY
    )

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    model_kwargs={"thinkingBudget": 0},
    api_key=GEMINI_API_KEY
    )

structured_llm = gemini_llm.with_structured_output(InputGuardrails)


categories = """
- BATHTUBS & SHOWERS
- TILES
- BATHROOM FAUCETS
- AUDIO EQUIPMENT
- BATHROOM ACCESSORIES
- SPEAKERS
- AUDIO ACCESSORIES
- LIGHTING
- TOILETS & BIDETS
- KITCHEN FAUCETS & SINKS
- BATHROOM VANITIES & SINKS
- HEADPHONES
- HOME DECOR
- FURNITURE
"""


# Function to check input
def check_guardrails(user_query, provider="groq"):
    formatted_prompt = guardrails_prompt.format(user_query=user_query, categories=categories)
    
    if provider == "groq":
        response = groq_llm.invoke(formatted_prompt)
        return response.content.strip()
    
    elif provider == "gemini":
        # for gemini
        # response = gemini_llm.invoke(formatted_prompt)
        # return response.content.strip()

        # for gemini with structured output 
        response = structured_llm.invoke(formatted_prompt)
        return response.allowed

if __name__ == "__main__":

    user_input = "show me something cool bathtub designs"
    start = time.time()
    is_allowed = check_guardrails(user_input, provider="groq")
    end = time.time()
    print(f"Guardrail check took {end - start:.2f} seconds.")
    print(is_allowed)  # True


# Sorry, I can only help with bathroom, kitchen, lighting, furniture, and audio products. How can I assist with your home improvement needs?