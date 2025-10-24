# laptop_data_manager.py
import pandas as pd
import json
import os
import config
import llm_utils # For preprocessing functions

# Global DataFrame to hold laptop data
# df_laptops = pd.read_csv(r"C:\Users\SOWMILY DUTTA\Python_Projects\Upgrad\Course 6 - Gen AI\ShopAssist_ Data + Demo-20250522T120309Z-1-001\ShopAssist_ Data + Demo\ShopAssist_New\laptop_data.csv")

df_laptops = None

def _product_map_layer(laptop_description_str: str):
    """Internal helper for product map layer extraction."""
    lap_spec_keys = ["GPU intensity", "Display quality", "Portability", "Multitasking", "Processing speed"]
    expected_output_format_example = {key: "low/medium/high" for key in lap_spec_keys}
    values_enum = ['low', 'medium', 'high']

    prompt=f"""
    You are a Laptop Specifications Classifier. Your job is to extract key features from a laptop description,
    classify them according to predefined intensity levels (low, medium, high), and output them in a JSON dictionary format.
    The keys in the JSON dictionary must be exactly: {json.dumps(lap_spec_keys)}.
    The values for each key must be one of {json.dumps(values_enum)}.

    To analyze each laptop, perform the following steps:
    Step 1: Read the laptop's description carefully.
    Step 2: For each feature category below, determine if its intensity is low, medium, or high based on the rules.
    Step 3: Construct a JSON dictionary with the classifications. Output ONLY the JSON dictionary.

    {config.PROMPT_DELIMITER}
    Classification Rules:
    GPU Intensity:
    - low: Integrated graphics (e.g., Intel UHD, Intel Iris Xe [base models]), AMD Radeon Graphics (integrated).
    - medium: Mid-range dedicated graphics (e.g., NVIDIA GeForce MX series, lower-end GTX, Apple M-series GPU cores [e.g. M1/M2 non-Pro/Max]), Intel Iris Xe (when highlighted for light creative tasks).
    - high: High-end dedicated graphics (e.g., NVIDIA GeForce RTX series, AMD Radeon RX series, Apple M-series Pro/Max/Ultra GPU cores).
    Display Quality (consider resolution primarily, then other features like type/refresh rate):
    - low: Resolution below Full HD (e.g., HD 1366x768).
    - medium: Full HD resolution (1920x1080 or WUXGA 1920x1200).
    - high: Higher than Full HD (e.g., QHD 2560x1440, 4K 3840x2160, Retina) OR Full HD with premium features like OLED, high refresh rate (120Hz+), excellent color accuracy.
    Portability (based on Laptop Weight, extracted from description or specs):
    - high: Laptop weight less than 1.51 kg.
    - medium: Laptop weight between 1.51 kg and 2.51 kg.
    - low: Laptop weight greater than 2.51 kg.
    Multitasking (based on RAM Size):
    - low: 8GB RAM or less.
    - medium: 16GB RAM.
    - high: 32GB RAM or more.
    Processing Speed (based on CPU Type - prioritize generation and series):
    - low: Entry-level processors (e.g., Intel Celeron, Pentium, Core i3 [older gens], AMD Athlon, Ryzen 3 [older gens]).
    - medium: Mid-range processors (e.g., Intel Core i5, AMD Ryzen 5, Apple M1/M2/M3 base chips, newer Core i3/Ryzen 3).
    - high: High-performance processors (e.g., Intel Core i7/i9, AMD Ryzen 7/9, Apple M-series Pro/Max/Ultra chips).
    {config.PROMPT_DELIMITER}

    Example Input/Output:
    Input Description: "The Dell Inspiron is a versatile laptop... Intel Core i5 processor... 8GB of RAM... 15.6"" LCD display with 1920x1080... Weighing just 2.5 kg... Intel UHD GPU..."
    Expected Output: {{"GPU intensity": "low","Display quality":"medium","Portability":"medium","Multitasking":"low","Processing speed":"medium"}}
    {config.PROMPT_DELIMITER}
    Strictly output ONLY the JSON dictionary in the format {expected_output_format_example}.
    """
    input_prompt_content = f"""Based on the rules and examples, classify the following laptop description: "{laptop_description_str}"."""
    messages=[{"role": "system", "content":prompt },{"role": "user","content":input_prompt_content}]
    response = llm_utils.get_chat_completions_for_preprocessing(messages, json_format=True)

    if isinstance(response, dict) and "error" not in response and all(key in response for key in lap_spec_keys):
        return response
    else:
        print(f"Warning: _product_map_layer received unexpected or error response: {response} for description: {laptop_description_str[:50]}...")
        return {key: "unknown" for key in lap_spec_keys}

def _persona_tag(specs_dict: dict):
    """Internal helper for persona tagging."""
    if not isinstance(specs_dict, dict) or any(val == "unknown" or val == "error" for val in specs_dict.values()):
        return []

    persona_tag_format = {"persona": ["Persona Values"]}

    prompt = f"""You are a Laptop Persona Classifier. Your job is to identify relevant user personas for a laptop based on its classified specifications.
    The output should be a Python dictionary with a single key "persona" whose value is a list of strings from {json.dumps(config.PERSONA_VALUES)}.
    Output ONLY the JSON dictionary.

    Step 1: Analyze the input dictionary of laptop specifications ratings: {specs_dict}.
    Step 2: Based on the combination of these ratings, identify all applicable personas. A laptop can have multiple personas.

    Few-shot examples (Input is the spec ratings dict, Output is the persona dict):
	Input: {{'GPU intensity': 'medium', 'Display quality':'medium', 'Portability':'medium', 'Multitasking':'high', 'Processing speed':'medium'}}
	Output: {{"persona":["developer", "student", "casual_user"]}}
	Input: {{'GPU intensity': 'low', 'Display quality':'medium', 'Portability':'high', 'Multitasking':'medium', 'Processing speed':'low'}}
	Output: {{"persona":["traveler", "budget_conscious", "casual_user", "student"]}}
    {config.PROMPT_DELIMITER}"""

    input_prompt_content = f"""Follow the above instructions. Generate the persona list in the format {persona_tag_format} for the laptop specs: {specs_dict}."""
    messages=[{"role": "system", "content":prompt },{"role": "user","content":input_prompt_content}]
    response = llm_utils.get_chat_completions_for_preprocessing(messages, json_format=True)

    if isinstance(response, dict) and 'persona' in response and isinstance(response['persona'], list):
        return response['persona']
    else:
        print(f"Warning: _persona_tag received unexpected response: {response} for specs: {specs_dict}")
        return []

def _convert_column_from_string(df, column_name):
    """Safely converts a string representation of a list/dict in a DataFrame column back to its Python object."""
    def safe_literal_eval(val):
        if isinstance(val, str):
            try:
                # Attempt to fix common string issues for JSON parsing
                # Handles single quotes, None -> null, True/False -> true/false
                s = val.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
                return json.loads(s)
            except (json.JSONDecodeError, TypeError):
                return [] if column_name == 'Persona' else {} # Default for lists or dicts
        return val # Return as is if not a string (already parsed or other type)
    
    if column_name in df.columns:
        df[column_name] = df[column_name].apply(safe_literal_eval)
    return df


def initialize_data(force_reprocess=config.REPROCESS_DATA):
    global df_laptops
    if not force_reprocess and os.path.exists(config.PREPROCESSED_LAPTOP_DATA_CSV):
        print(f"Loading preprocessed data from {config.PREPROCESSED_LAPTOP_DATA_CSV}...")
        df_laptops = pd.read_csv(config.PREPROCESSED_LAPTOP_DATA_CSV)
        # Convert string representations of lists/dicts back to objects
        df_laptops = _convert_column_from_string(df_laptops, 'Specification_Ratings')
        df_laptops = _convert_column_from_string(df_laptops, 'Persona')
        print(f"Loaded {len(df_laptops)} preprocessed laptops.")
        return

    print(f"Preprocessing data from {config.LAPTOP_DATA_CSV}...")
    try:
        df_laptops = pd.read_csv(config.LAPTOP_DATA_CSV)
        print(df_laptops.head())
    except FileNotFoundError:
        print(f"Error: The file {config.LAPTOP_DATA_CSV} was not found. Cannot proceed.")
        df_laptops = pd.DataFrame() # Ensure it's an empty DataFrame
        return

    if 'Description' not in df_laptops.columns:
        print("Error: 'Description' column is missing. Cannot proceed with preprocessing.")
        df_laptops = pd.DataFrame()
        return
    
    df_laptops['Price'] = df_laptops['Price'].str.replace(",","")
    print(df_laptops.head())
    df_laptops['Price'] = pd.to_numeric(df_laptops['Price'], errors='coerce')
    df_laptops.dropna(subset=['Price'], inplace=True)

    print("Generating 'Specification_Ratings' (this may take a while)...")
    df_laptops['Specification_Ratings'] = df_laptops['Description'].apply(
        lambda x: _product_map_layer(x) if pd.notna(x) else {}
    )
    print("Generating 'Persona' tags (this may take a while)...")
    df_laptops['Persona'] = df_laptops['Specification_Ratings'].apply(
        lambda x: _persona_tag(x) if isinstance(x, dict) and x else []
    )
    
    try:
        df_laptops.to_csv(config.PREPROCESSED_LAPTOP_DATA_CSV, index=False)
        print(df_laptops.head())
        print(f"Preprocessed data saved to {config.PREPROCESSED_LAPTOP_DATA_CSV}")
    except Exception as e:
        print(f"Error saving preprocessed data: {e}")

    print("\nSample of preprocessed data:")
    print(df_laptops[['Model Name', 'Price', 'Specification_Ratings', 'Persona']].head())


def get_laptop_dataframe():
    """Returns the loaded and preprocessed laptop DataFrame."""
    global df_laptops
    if df_laptops is None:
        # Attempt to initialize if not already loaded (e.g., direct call without main_chatbot sequence)
        print("Laptop data not initialized. Attempting to load/preprocess...")
        initialize_data() 
    return df_laptops