# -*- coding: utf-8 -*-
import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import time

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define output structure for multi-query detection
class MultiQueryDetectionOutput(BaseModel):
    has_multiple_queries: bool = Field(description="True if input contains multiple distinct queries, False otherwise")
    query_count: int = Field(description="Number of distinct queries detected")
    queries: List[str] = Field(description="List of individual queries. If single query, list contains one item")


# Setup Groq LLM for multi-query detection
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=8192,
    api_key=GROQ_API_KEY
)

# Groq: use json_mode method for structured output
groq_structured_llm = groq_llm.with_structured_output(MultiQueryDetectionOutput, method="json_mode")


# Multi-query detection prompt
MULTI_QUERY_DETECT_PROMPT = """You are an expert at analyzing user input and detecting if it contains multiple distinct queries or requests.

Your task is to:
1. Analyze the input text to determine if it contains one or multiple distinct queries
2. Count the number of distinct queries
3. Separate and extract each individual query

Input text: {text}

Instructions:
- A single query is one request, question, or search intent
- Multiple queries are when the user asks for several different things, products, or information in one input
- Examples of MULTIPLE queries:
  * "Show me laptops and headphones" (2 queries)
  * "I want a red dress, blue shoes, and a handbag" (3 queries)
  * "Find me iPhone 15, Samsung TV, and Nike shoes" (3 queries)
  * "Search for smartphones and also show me tablets" (2 queries)
  * "Show me white bathtub under 500 and black faucet under 300" (2 queries - two different products with their own constraints)
- Examples of SINGLE queries:
  * "Show me red Nike shoes" (1 query - single product with attributes)
  * "I want a laptop with 16GB RAM" (1 query - single product with specifications)
  * "Find wireless headphones under $100" (1 query - single product with constraints)
  * "Black leather jacket for men" (1 query - single product with attributes)
- If queries are connected with "and", "also", commas separating different items, they are typically multiple queries
- If the text describes attributes or specifications of a single item, it's a single query
- Preserve the original meaning and context of each query when separating
- If uncertain, treat as a single query

Return your response in JSON format with:
- has_multiple_queries: boolean (true if more than one query, false if single query)
- query_count: integer (number of distinct queries found)
- queries: array of strings (each individual query as a separate string)
"""


def detect_multiple_queries(text: str) -> MultiQueryDetectionOutput:
    """
    Detect if user input contains multiple queries and separate them

    Args:
        text: Input text from user (may contain single or multiple queries)

    Returns:
        MultiQueryDetectionOutput with:
        - has_multiple_queries: True/False
        - query_count: Number of queries detected
        - queries: List of individual queries
    """
    formatted_prompt = MULTI_QUERY_DETECT_PROMPT.format(text=text)

    response = groq_structured_llm.invoke(formatted_prompt)
    return response


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("Show me laptops", False, 1),  # Single query
        ("I want laptops and headphones", True, 2),  # Multiple queries
        ("Find me iPhone 15, Samsung TV, and Nike shoes", True, 3),  # Multiple queries
        ("Red Nike running shoes size 10", False, 1),  # Single query with attributes
        ("Black leather jacket for men", False, 1),  # Single query
        ("Show me smartphones and also tablets", True, 2),  # Multiple queries
        ("Laptop with 16GB RAM and SSD", False, 1),  # Single query with specs
        ("I need a dress, shoes, and a handbag", True, 3),  # Multiple queries
        ("Wireless headphones under $100", False, 1),  # Single query
        ("Show me gaming laptops and mechanical keyboards", True, 2),  # Multiple queries
        ("Show me white bathtub under 500 and black faucet under 300", True, 2),  # Multiple queries with price constraints
    ]

    print("=== Multi-Query Detection Test Cases ===\n")

    for text, expected_multiple, expected_count in test_cases:
        start = time.time()
        result = detect_multiple_queries(text)
        end = time.time()

        status = "✓" if (result.has_multiple_queries == expected_multiple and
                         result.query_count == expected_count) else "✗"

        print(f"{status} Input: '{text}'")
        print(f"  Has Multiple Queries: {result.has_multiple_queries} (Expected: {expected_multiple})")
        print(f"  Query Count: {result.query_count} (Expected: {expected_count})")
        print(f"  Separated Queries:")
        for i, query in enumerate(result.queries, 1):
            print(f"    {i}. {query}")
        print(f"  Time: {end - start:.2f}s\n")
