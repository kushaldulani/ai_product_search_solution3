import time
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from prompts import filter_extraction_prompt
from store_companies_qdrant import search_companies
from store_colors_qdrant import search_colors
from dotenv import load_dotenv
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0,
    model_kwargs={"thinkingBudget": 0},
    api_key=GEMINI_API_KEY
    )

categories = """
1. BATHTUBS & SHOWERS
2. TILES
3. BATHROOM FAUCETS
4. AUDIO EQUIPMENT
5. BATHROOM ACCESSORIES
6. SPEAKERS
7. AUDIO ACCESSORIES
8. LIGHTING
9. TOILETS & BIDETS
10. KITCHEN FAUCETS & SINKS
11. BATHROOM VANITIES & SINKS
12. HEADPHONES
13. HOME DECOR
14. FURNITURE
"""

installation_types = """
- built-in-wall
- ceiling
- ceiling-wall
- floor
- floor-mounted
- free-standing
- tabletop
- wall
"""

# top_10_brands = """
# Aquabrass (score: 0.5157)
# Perrin & Rowe (score: 0.4343)
# Brizo (score: 0.4156)
# Riobel Pro (score: 0.4080)
# Grohe (score: 0.3916)
# Hansgrohe (score: 0.3894)
# Riobel (score: 0.3884)
# TUBS The Ultimate Bath Store (score: 0.3851)
# Kohler (score: 0.3810)
# Rubinet (score: 0.3732)
# """


# top_10_colors = """
# 1. Matte Black (score: 0.4206) 
# 2. Matte Black (score: 0.4206) 
# 3. brushed-nickel (score: 0.3779) 
# 4. brushed-nickel (score: 0.3779) 
# 5. Oil-Rubbed Bronze (score: 0.3778) 
# 6. Oil-Rubbed Bronze (score: 0.3778) 
# 7. Black and Chrome (score: 0.3752) 
# 8. Black and Chrome (score: 0.3752) 
# 9. Brushed Nickel (score: 0.3625) 
# 10. Brushed Nickel (score: 0.3625)
# """

def exract_filters_from_user_query(user_query):

    # Get top 10 values from Qdrant
    top_10_brands = search_companies(user_query)
    top_10_colors = search_colors(user_query)

    print("Top 10 brands from Qdrant:", top_10_brands)
    print("Top 10 colors from Qdrant:", top_10_colors)
    formatted_prompt = filter_extraction_prompt.format(
        user_query=user_query,
        categories=categories,
        installation_types=installation_types,
        top_10_brands=top_10_brands,
        top_10_colors=top_10_colors
    )

    # print(formatted_prompt)
    response = gemini_llm.invoke(formatted_prompt)
    print("Raw response filters from Gemini:" , response)
    # Handle both string and list of content blocks, sometimes .content does not return a string
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        # Extract text from the first text block
        for block in response.content:
            if isinstance(block, dict) and block.get('type') == 'text':
                return block['text']
        # Fallback: return the entire content if no text block found
        return str(response.content)
    else:
        return str(response.content)

if __name__ == "__main__":
    user_query = "show me bathroom sink in matt blake and baby pink in riobal brand under 500 dollars"
    filters = exract_filters_from_user_query(user_query)
    print("Extracted Filters:")
    print(filters)