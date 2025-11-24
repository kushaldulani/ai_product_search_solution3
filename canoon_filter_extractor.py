# -*- coding: utf-8 -*-
import os
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import json

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Define output structure for filter extraction
# Field names match the API response format (friendly display names)
class PlantFilterOutput(BaseModel):
    hardiness_zone: Optional[List[str]] = Field(
        default=None,
        description="Hardiness zone values from 0-7. Examples: ['3'], ['4', '5'], None",
        alias="Hardiness Zone"
    )
    foliage_type: Optional[List[str]] = Field(
        default=None,
        description="Foliage type: 'deciduous' or 'evergreen'. Can have multiple values. Examples: ['deciduous'], ['evergreen'], ['deciduous', 'evergreen'], None",
        alias="Foliage Type"
    )
    fall_colour: Optional[List[str]] = Field(
        default=None,
        description="Fall color from: brown, green, orange, pink, purple, red, yellow. Can have multiple values. Examples: ['red'], ['red', 'yellow'], ['orange', 'yellow'], None",
        alias="Fall Colour"
    )
    flower_colour: Optional[List[str]] = Field(
        default=None,
        description="Flower color from: black, blue, brown, green, orange, pink, purple, red, white, yellow. Can have multiple values. Examples: ['yellow'], ['pink', 'white'], ['red', 'yellow'], None",
        alias="Flower Colour"
    )
    low_maintenance: Optional[List[str]] = Field(
        default=None,
        description="Low maintenance: ['true'] or ['false']. Must be array. Examples: ['true'], ['false'], None",
        alias="Low Maintenance"
    )
    salt_tolerant: Optional[List[str]] = Field(
        default=None,
        description="Salt tolerant: ['true'] or ['false']. Must be array. Examples: ['true'], ['false'], None",
        alias="Salt Tolerant"
    )
    soil_moisture: Optional[List[str]] = Field(
        default=None,
        description="Soil moisture from: 'average to moist', 'average to wet', 'dry to average', 'dry to moist', 'dry to wet', 'moist', 'moist to wet'. Can have multiple values. Examples: ['moist'], ['average to moist'], None",
        alias="Soil Moisture"
    )
    sun_exposure: Optional[List[str]] = Field(
        default=None,
        description="Sun exposure from: 'full sun', 'full sun to partial shade', 'full sun to shade', 'partial shade', 'partial shade to shade', 'shade'. Can have multiple values. Examples: ['full sun'], ['partial shade', 'shade'], None",
        alias="Sun Exposure"
    )
    height: Optional[float] = Field(
        default=None,
        description="Height in centimeters. For ranges like 'between 50-100cm', use the maximum value. Examples: 100.0, 250.0, None",
        alias="Height"
    )
    spread: Optional[float] = Field(
        default=None,
        description="Spread/width in centimeters. For ranges like 'between 30-50cm wide', use the maximum value. Examples: 60.0, 150.0, None",
        alias="Spread"
    )
    category: Optional[List[str]] = Field(
        default=None,
        description="Plant category from: 'broadleafs', 'fruits', 'grasses', 'groundcovers', 'houseplants', 'perennials', 'roses', 'shrubs', 'trees', 'tropicals', 'vines'. Can have multiple values. Examples: ['trees'], ['shrubs', 'trees'], ['grasses'], None",
        alias="Category"
    )

    class Config:
        populate_by_name = True


# Setup Groq LLM for filter extraction
groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=8192,
    api_key=GROQ_API_KEY
)

# Groq: use json_mode method for structured output
groq_structured_llm = groq_llm.with_structured_output(PlantFilterOutput, method="json_mode")


# Filter extraction prompt
FILTER_EXTRACTION_PROMPT = """You are a plant search filter extraction expert. Your task is to analyze user queries and extract relevant plant filters.

Available filters and their valid values:

1. **Hardiness Zone**: ["0", "1", "2", "3", "4", "5", "6", "7"]
2. **Foliage Type**: ["deciduous", "evergreen"]
3. **Fall Colour**: ["brown", "green", "orange", "pink", "purple", "red", "yellow"]
4. **Flower Colour**: ["black", "blue", "brown", "green", "orange", "pink", "purple", "red", "white", "yellow"]
5. **Low Maintenance**: ["true", "false"] (must be array)
6. **Salt Tolerant**: ["true", "false"] (must be array)
7. **Soil Moisture**: ["average to moist", "average to wet", "dry to average", "dry to moist", "dry to wet", "moist", "moist to wet"]
8. **Sun Exposure**: ["full sun", "full sun to partial shade", "full sun to shade", "partial shade", "partial shade to shade", "shade"]
9. **Height**: Single value in centimeters (100cm = 1 meter, 1 foot = 30.48cm, 1 inch = 2.54cm)
10. **Spread**: Single value in centimeters
11. **Category**: ["broadleafs", "fruits", "grasses", "groundcovers", "houseplants", "perennials", "roses", "shrubs", "trees", "tropicals", "vines"]

User Query: {query}

Instructions:
- Extract ONLY the filters mentioned in the user query
- For filters NOT mentioned, return null/None
- Use exact values from the valid options (case-sensitive)
- **CRITICAL**: ALL fields except Height and Spread MUST be arrays/lists, even for single values
  - CORRECT: "Soil Moisture": ["average to moist"]
  - WRONG: "Soil Moisture": "average to moist"
  - CORRECT: "Low Maintenance": ["true"]
  - WRONG: "Low Maintenance": "true"
- **IMPORTANT**: For multiple values in one filter category (like "red and yellow flowers" or "red or yellow fall color"), return as array/list: ["red", "yellow"]
- For boolean-like filters (Low Maintenance, Salt Tolerant), use ["true"] or ["false"] as arrays, NOT boolean or string
- For Height and Spread (SINGLE NUMERIC VALUE ONLY, use maximum for ranges):
  - Convert all measurements to centimeters
  - "under 1 meter" → Height: 100
  - "between 50-100cm" → Height: 100 (take maximum)
  - "over 2 feet" → Height: 61 (minimum constraint, use the value)
  - "small plants" → Height: 100 (interpret based on context)
  - "medium shrubs" → Height: 300 (take typical maximum)
  - "tall trees" → Height: 1000 (typical tall tree)
  - For ranges like "50-100cm", ALWAYS use the MAXIMUM value (100)
- For colors, match to closest valid option
- For sun exposure, match user terms like "sunny" → "full sun", "shady" → "shade"
- For moisture, match terms like "wet soil" → "moist to wet", "dry conditions" → "dry to average", "average" → "average to moist"
- For category, detect from:
  - Direct mentions: "tree", "shrub", "grass", "perennial", "vine", "rose", "groundcover", "houseplant", "tropical", "fruit", "broadleaf"
  - Plant names containing category words: "Feather Reed Grass" → ["grasses"], "Maple Tree" → ["trees"], "Climbing Rose" → ["roses"]
  - Plurals: "trees", "shrubs", "grasses", "perennials", "vines", "roses", "groundcovers", "houseplants", "tropicals", "fruits"
  - Match: "tree/trees" → "trees", "shrub/shrubs" → "shrubs", "grass/grasses" → "grasses", "perennial/perennials" → "perennials", etc.
- Return your response in JSON format with keys matching the filter names exactly (use spaces and capital letters)

Examples:

Query: "I want plants with hardiness zone 7, red fall color, and yellow flowers"
Response: {{
  "Hardiness Zone": ["7"],
  "Fall Colour": ["red"],
  "Flower Colour": ["yellow"]
}}

Query: "I want plants whose fall colors are red and yellow"
Response: {{
  "Fall Colour": ["red", "yellow"]
}}

Query: "Low maintenance plants whose moisture is average"
Response: {{
  "Low Maintenance": ["true"],
  "Soil Moisture": ["average to moist"]
}}

Query: "Show me small evergreen plants under 1 meter that are low maintenance"
Response: {{
  "Foliage Type": ["evergreen"],
  "Height": 100,
  "Low Maintenance": ["true"]
}}

Query: "I need plants for zone 5, preferably with orange or yellow fall colors, full sun"
Response: {{
  "Hardiness Zone": ["5"],
  "Fall Colour": ["orange", "yellow"],
  "Sun Exposure": ["full sun"]
}}

Query: "Compact plants, 50-80cm tall, 30-50cm wide, pink flowers"
Response: {{
  "Height": 80,
  "Spread": 50,
  "Flower Colour": ["pink"]
}}

Query: "Plants with pink or white flowers"
Response: {{
  "Flower Colour": ["pink", "white"]
}}

Query: "Show me trees and shrubs for zone 5"
Response: {{
  "Category": ["trees", "shrubs"],
  "Hardiness Zone": ["5"]
}}

Query: "I want ornamental grasses that are low maintenance"
Response: {{
  "Category": ["grasses"],
  "Low Maintenance": ["true"]
}}

Query: "Looking for perennials with yellow flowers for full sun"
Response: {{
  "Category": ["perennials"],
  "Flower Colour": ["yellow"],
  "Sun Exposure": ["full sun"]
}}

Query: "show me Eldorado Feather Reed Grass"
Response: {{
  "Category": ["grasses"]
}}

Query: "native grasses that grow 3 to 4 feet and tolerate moist soil"
Response: {{
  "Category": ["grasses"],
  "Height": 122,
  "Soil Moisture": ["moist"]
}}

Query: "I want Red Maple Tree for zone 5"
Response: {{
  "Category": ["trees"],
  "Hardiness Zone": ["5"]
}}

Now extract filters from the user query above.
"""


def extract_filters(query: str) -> PlantFilterOutput:
    """
    Extract plant filters from natural language query

    Args:
        query: User's natural language search query

    Returns:
        PlantFilterOutput with extracted filters
    """
    formatted_prompt = FILTER_EXTRACTION_PROMPT.format(query=query)
    response = groq_structured_llm.invoke(formatted_prompt)
    return response


def filters_to_dict(filters: PlantFilterOutput) -> dict:
    """
    Convert PlantFilterOutput to a clean dictionary with aliases (display names), excluding None values

    Args:
        filters: PlantFilterOutput object

    Returns:
        Dictionary with only non-None filter values using display names as keys
    """
    # Use by_alias=True to get the display names (e.g., "Hardiness Zone" instead of "hardiness_zone")
    return filters.model_dump(by_alias=True, exclude_none=True)


def filters_to_json(filters: PlantFilterOutput, pretty: bool = True) -> str:
    """
    Convert filters to JSON string

    Args:
        filters: PlantFilterOutput object
        pretty: If True, return pretty-formatted JSON

    Returns:
        JSON string
    """
    clean_dict = filters_to_dict(filters)
    if pretty:
        return json.dumps(clean_dict, indent=2, ensure_ascii=False)
    return json.dumps(clean_dict, ensure_ascii=False)


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "I want plant whose Hardiness Zone is 7, fall color red, and flower color is yellow",
        "I want plants whose fall colors are red and yellow",
        "Show me small plants under 1 meter",
        "I need evergreen plants for zone 5 with low maintenance",
        "Compact plants, 50-100cm tall, pink or white flowers, full sun",
        "Plants with red and orange fall foliage and purple flowers",
        "Tall plants over 5 meters",
        "Plants for shady areas, salt tolerant",
        "Medium plants between 1-2 feet, yellow or white flowers, dry conditions",
        "Low maintenance plants",
        "Plants with orange or red fall color for zone 4, partial shade",
        "Show me trees and shrubs for zone 5",
        "I want ornamental grasses that are low maintenance",
        "Looking for perennials with yellow flowers for full sun",
        "Show me roses and vines for partial shade",
        "I need groundcovers for dry areas",
    ]

    print("=" * 90)
    print("PLANT FILTER EXTRACTION TEST CASES")
    print("=" * 90)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 90)

        try:
            result = extract_filters(query)
            print("Extracted Filters:")
            print(filters_to_json(result))

        except Exception as e:
            print(f"Error: {e}")

        print()

    # Interactive mode
    print("\n" + "=" * 90)
    print("INTERACTIVE MODE - Enter your queries (type 'exit' to quit)")
    print("=" * 90)

    while True:
        user_query = input("\nEnter your query: ").strip()

        if user_query.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break

        if not user_query:
            continue

        try:
            result = extract_filters(user_query)
            print("\nExtracted Filters (JSON):")
            print(filters_to_json(result))

            print("\nExtracted Filters (Python Dict):")
            print(filters_to_dict(result))

        except Exception as e:
            print(f"Error: {e}")
