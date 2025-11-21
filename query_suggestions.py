# -*- coding: utf-8 -*-
import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load brands from companies.txt
def load_brands():
    brands_file = os.path.join(os.path.dirname(__file__), "data", "companies.txt")
    try:
        with open(brands_file, "r") as f:
            brands = [line.strip() for line in f if line.strip()]
        return brands
    except Exception as e:
        print(f"Error loading brands: {e}")
        return []

BRANDS = load_brands()
BRANDS_LIST = ", ".join(BRANDS[:20])  # Use first 20 brands for prompt


# Define output structure for query suggestions
class QuerySuggestionsOutput(BaseModel):
    suggestions: List[str] = Field(description="List of 5 alternative query suggestions with variations in brand, color, or price")


# Setup Groq LLM for suggestions
groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

# Groq: use json_mode method for structured output
groq_structured_llm = groq_llm.with_structured_output(QuerySuggestionsOutput, method="json_mode")


# Query suggestions prompt
QUERY_SUGGESTIONS_PROMPT = """You are an expert at generating helpful product search query variations for bathroom, kitchen, lighting, furniture, and audio products.

Your task is to generate 5 alternative search queries based on the user's original query. Create variations by modifying:
- Brand names (use ONLY brands from this list: {brands})
- Colors (suggest alternative colors that work well)
- Price ranges (suggest both lower and higher price points)
- Finish types (matte, polished, brushed, chrome, nickel, etc.)

Original query: {query}

Guidelines:
- Generate exactly 5 suggestions
- Make each suggestion meaningfully different from the original
- Keep the core product type the same (e.g., if they asked for faucet, keep it as faucet)
- ONLY use brand names from the provided list
- Variations should be realistic and useful
- Mix up the variations (don't just change price 5 times)
- Keep suggestions concise and natural
- Vary price ranges by ±20-50%

Examples:
Input: "Riobel brand black faucet under 500"
Output suggestions:
1. "Kohler brand matte black faucet under 600"
2. "Riobel brand chrome faucet under 400"
3. "Delta brand black faucet under 300"
4. "Grohe brand brushed nickel faucet under 500"
5. "Hansgrohe brand black faucet under 450"

Input: "white bathtub under 1000"
Output suggestions:
1. "white bathtub under 1500"
2. "beige bathtub under 1000"
3. "Neptune white freestanding bathtub under 1200"
4. "Maax white acrylic bathtub under 800"
5. "white soaking bathtub under 900"

Return your response in JSON format with:
- suggestions: array of 5 strings (each alternative query)
"""


def generate_query_suggestions(query: str) -> QuerySuggestionsOutput:
    """
    Generate 5 alternative query suggestions based on user's original query

    Args:
        query: Original user query

    Returns:
        QuerySuggestionsOutput with 5 suggestions
    """
    if not query or not query.strip():
        return QuerySuggestionsOutput(suggestions=[])

    formatted_prompt = QUERY_SUGGESTIONS_PROMPT.format(
        query=query,
        brands=BRANDS_LIST
    )

    try:
        response = groq_structured_llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return QuerySuggestionsOutput(suggestions=[])


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "Riobel brand black faucet under 500",
        "white bathtub under 1000",
        "modern chandelier for dining room",
        "leather sofa under 2000",
        "Kohler chrome faucet under 800",
    ]

    print("=== Query Suggestions Test Cases ===\n")
    print(f"Available brands: {len(BRANDS)}\n")

    for query in test_queries:
        print(f"Original Query: '{query}'")
        result = generate_query_suggestions(query)
        print(f"Suggestions:")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()
