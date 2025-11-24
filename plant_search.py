# -*- coding: utf-8 -*-
import pandas as pd
from typing import List, Optional, Tuple
from canoon_filter_extractor import extract_filters, filters_to_dict


def parse_zone(zone_value) -> Optional[List[int]]:
    """Parse zone values from various formats to list of integers"""
    if pd.isna(zone_value):
        return None

    zone_str = str(zone_value).strip()
    zones = []

    # Handle ranges like "3-7" or "3 to 7"
    if '-' in zone_str or ' to ' in zone_str:
        parts = zone_str.replace(' to ', '-').split('-')
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
            zones = list(range(start, end + 1))
        except:
            pass
    else:
        # Single zone
        try:
            zones = [int(zone_str)]
        except:
            pass

    return zones if zones else None


def parse_boolean_field(value) -> Optional[bool]:
    """Parse boolean fields from various formats"""
    if pd.isna(value):
        return None

    val_str = str(value).strip().lower()
    if val_str in ['true', 'yes', '1', 'y']:
        return True
    elif val_str in ['false', 'no', '0', 'n']:
        return False
    return None


def parse_color_list(color_value) -> Optional[List[str]]:
    """Parse color values which might be comma-separated"""
    if pd.isna(color_value) or str(color_value).strip() == '':
        return None

    color_str = str(color_value).strip().lower()
    # Split by common separators
    colors = [c.strip() for c in color_str.replace(';', ',').split(',')]
    return [c for c in colors if c]


def apply_filters_to_data(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply extracted filters to the DataFrame

    Args:
        df: DataFrame with plant data
        filters: Dictionary of filters from extract_filters

    Returns:
        Filtered DataFrame
    """
    result_df = df.copy()

    # Hardiness Zone filter
    if 'Hardiness Zone' in filters:
        filter_zones = [int(z) for z in filters['Hardiness Zone']]
        mask = result_df['zone'].apply(
            lambda x: any(z in filter_zones for z in parse_zone(x)) if parse_zone(x) else False
        )
        result_df = result_df[mask]

    # Foliage Type filter (deciduous/evergreen)
    if 'Foliage Type' in filters:
        filter_types = [ft.lower() for ft in filters['Foliage Type']]
        mask = result_df['attr_evergreen_or_deciduous'].apply(
            lambda x: str(x).lower() in filter_types if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Fall Colour filter
    if 'Fall Colour' in filters:
        filter_colors = [c.lower() for c in filters['Fall Colour']]
        mask = result_df['attr_fall_color'].apply(
            lambda x: any(fc in filter_colors for fc in parse_color_list(x)) if parse_color_list(x) else False
        )
        result_df = result_df[mask]

    # Flower Colour filter
    if 'Flower Colour' in filters:
        filter_colors = [c.lower() for c in filters['Flower Colour']]
        mask = result_df['attr_flower_color'].apply(
            lambda x: any(fc in filter_colors for fc in parse_color_list(x)) if parse_color_list(x) else False
        )
        result_df = result_df[mask]

    # Low Maintenance filter
    if 'Low Maintenance' in filters:
        filter_value = filters['Low Maintenance'][0].lower() == 'true'
        mask = result_df['attr_is_low_maintenance'].apply(
            lambda x: parse_boolean_field(x) == filter_value if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Salt Tolerant filter
    if 'Salt Tolerant' in filters:
        filter_value = filters['Salt Tolerant'][0].lower() == 'true'
        mask = result_df['attr_is_salt_tolerant'].apply(
            lambda x: parse_boolean_field(x) == filter_value if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Soil Moisture filter
    if 'Soil Moisture' in filters:
        filter_moisture = [m.lower() for m in filters['Soil Moisture']]
        mask = result_df['attr_moisture_descriptor'].apply(
            lambda x: any(fm in str(x).lower() for fm in filter_moisture) if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Sun Exposure filter
    if 'Sun Exposure' in filters:
        filter_sun = [s.lower() for s in filters['Sun Exposure']]
        mask = result_df['attr_sunlight_descriptor'].apply(
            lambda x: any(fs in str(x).lower() for fs in filter_sun) if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Height filter - use tolerance range for approximate matching
    if 'Height' in filters:
        target_height = filters['Height']
        # Use ±1% tolerance for height matching to handle rounding differences
        # This allows 152.4 to match when user specifies 152 or 153
        tolerance = target_height * 0.01
        min_height = target_height - tolerance
        max_height = target_height + tolerance
        mask = result_df['attr_height'].apply(
            lambda x: (min_height <= float(x) <= max_height) if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Spread filter - use tolerance range for approximate matching
    if 'Spread' in filters:
        target_spread = filters['Spread']
        # Use ±1% tolerance for spread matching
        tolerance = target_spread * 0.01
        min_spread = target_spread - tolerance
        max_spread = target_spread + tolerance
        mask = result_df['attr_spread_descriptor'].apply(
            lambda x: (min_spread <= float(x) <= max_spread) if pd.notna(x) else False
        )
        result_df = result_df[mask]

    # Category filter
    if 'Category' in filters:
        filter_categories = [c.lower() for c in filters['Category']]
        mask = result_df['sub_category_code'].apply(
            lambda x: str(x).lower() in filter_categories if pd.notna(x) else False
        )
        result_df = result_df[mask]

    return result_df


def search_plants(query: str, excel_path: str = None) -> Tuple[dict, List[str]]:
    """
    Search for plants based on natural language query

    Args:
        query: Natural language search query
        excel_path: Path to the Excel file with plant data

    Returns:
        Tuple of (filters_dict, sku_ids_list)
    """
    # Extract filters from query
    filter_output = extract_filters(query)
    filters = filters_to_dict(filter_output)

    # Use absolute path if not provided
    if excel_path is None:
        import os
        excel_path = os.path.join(os.path.dirname(__file__), 'data', 'data_cleaned.xlsx')

    # Load data
    df = pd.read_excel(excel_path)

    print(f"Loaded {len(df)} plants from database")
    print(f"Extracted filters: {filters}")

    # Apply filters
    filtered_df = apply_filters_to_data(df, filters)

    print(f"After filtering: {len(filtered_df)} plants match")

    # Get SKU IDs
    sku_ids = filtered_df['sku'].tolist()

    return filters, sku_ids


if __name__ == "__main__":
    # Test cases
    test_queries = [
        "I want plants with hardiness zone 3, low maintenance",
        "Show me plants with red fall color and yellow flowers",
        "Small evergreen plants under 1 meter",
        "Plants for zone 5, full sun, low maintenance",
        "Show me trees and shrubs for zone 5",
        "I want ornamental grasses that are low maintenance",
        "Looking for perennials with yellow flowers for full sun",
    ]

    print("=" * 90)
    print("PLANT SEARCH TEST CASES")
    print("=" * 90)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. {'-' * 86}")
        print(f"Query: {query}")
        filters, sku_ids = search_plants(query)
        print(f"\nExtracted Filters:")
        print(filters)
        print(f"\nMatching SKU IDs ({len(sku_ids)}):")
        print(sku_ids[:20])  # Show first 20
        if len(sku_ids) > 20:
            print(f"... and {len(sku_ids) - 20} more")
        print()
