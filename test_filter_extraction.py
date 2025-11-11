from extract_filters import exract_filters_from_user_query

# Test queries
test_queries = [
    # "I am looking for product having sku:PXS01-BK",
    # "can you provide me the bathtub of ribel company?",
    # "can you provide me non black color bathtubs?",
    "can you provide me the bathtub of ribel and aura company? and it should of black or gray color only",
    "can you provide me the black color faucets under 400 and also can you provide white color faucets under 799",
    "can you provide me the taps of ribel and aura company? and it should of black or gray color only",
    # "black vannity",
    # "I am looking for a vanity that is 22\" wide",
    # "wall light",
    # "can you provide me the commode of toto company? and it should of white color only",
    # "i am looking for freestanding bathtub under 2000$",
    # "i am looking for floor mounted bathtub under 2000$",
    # "i am looking for freestanding floor mounted bathtub under 2000$",
    # "I am looking for wallmount vanity for modern looking bathroom",
    # "I am looking for vanity for modern looking bathroom",
    # "I am looking for vanity for modern looking bathroom under 2000$",
    # "I am looking for vanity for modern looking bathroom under 2000$ of rectangular shape and white color",
    # "can you show me wall light for my bathroom?",
    # "can you provide me the wall light of matteo brand",
    # "can you provide me the wall light of Modern Forms Luminaires & Forms brand",
    # "can you provide me best shower kit?",
    # "can you provide best two piece toilet?",
    # "can you provide best one piece or two piece toilet?",
    # "black matte tap for kitchen not wall-mounted",
    # "I am looking for a toilet for my grandmother having arthritis",
    # "Show me a kitchen faucet with a pull down sprayer",
    # "Show me matte black faucets for kitchen",
    # "I want a vanity faucet with single-handle operation.",
    # "Find two-piece toilets suitable for small bathrooms.",
    # "Find bathtub faucets with a hand shower attachment.",
    # "Show showers with sliding glass doors.",
    # "Show alcove bathtubs with built-in overflow protection.",
    # "Find shower heads with adjustable spray patterns.",
    # "Find floor mount vanities with storage cabinets.",
    # "I want bathroom mirrors with LED lighting.",
    # "Show medicine cabinets with anti-fog mirrors.",
    # "I want headphones with noise-cancellation.",
    # "Show wall light fixtures for living rooms.",
    # "can you provide me the white faucet with black handle",
    # "Can you show me bathtub which is not white color and it should not be freestanding"
]

def test_all_queries():
    """Test all queries and save results to a file"""
    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print(f"{'='*80}")

        try:
            response = exract_filters_from_user_query(query)
            results.append({
                'query': query,
                'response': response,
                'status': 'success'
            })
            print(f"Response:\n{response}")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            results.append({
                'query': query,
                'response': error_msg,
                'status': 'error'
            })
            print(error_msg)

    # Save results to file
    with open('test_queries2.txt', 'w', encoding='utf-8') as f:
        f.write("FILTER EXTRACTION TEST RESULTS\n")
        f.write("="*100 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\nQUERY {i}:\n")
            f.write(f"{result['query']}\n\n")
            f.write(f"STATUS: {result['status']}\n\n")
            f.write(f"RESPONSE:\n")
            f.write(f"{result['response']}\n")
            f.write("\n" + "-"*100 + "\n")

    print(f"\n\n{'='*80}")
    print(f"Testing complete! Results saved to test_queries.txt")
    print(f"Total queries: {len(test_queries)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_all_queries()
