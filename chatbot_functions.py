# chatbot_functions.py
import json
import pandas as pd
import laptop_data_manager
import config

def get_laptop_info(model_name: str):
    """Retrieves detailed information for a specific laptop model."""
    df = laptop_data_manager.get_laptop_dataframe()
    if df is None or df.empty:
        return json.dumps({"error": "Laptop data not loaded or is empty."})
    
    laptop_series = df[df['Model Name'].str.contains(model_name, case=False, na=False)]
    if not laptop_series.empty:
        # Convert all relevant data to dict, handle NaN
        laptop_details = laptop_series.iloc[0].where(pd.notna(laptop_series.iloc[0]), None).to_dict()
        return json.dumps({"status": "success", "data": laptop_details})
    else:
        return json.dumps({"status": "not_found", "message": f"Sorry, I couldn't find information for a laptop model like '{model_name}'."})

def recommend_laptops_by_criteria(budget_min: int = None, budget_max: int = None, personas: list = None):
    """Recommends laptops based on budget and/or personas."""
    df = laptop_data_manager.get_laptop_dataframe()
    if df is None or df.empty:
        return json.dumps({"error": "Laptop data not loaded or is empty."})

    filtered_df = df.copy()
    
    if budget_min is not None:
        filtered_df = filtered_df[filtered_df['Price'] >= budget_min]
    if budget_max is not None:
        filtered_df = filtered_df[filtered_df['Price'] <= budget_max]

    if personas and isinstance(personas, list) and len(personas) > 0:
        user_personas_lower = [p.lower() for p in personas]
        def match_personas(laptop_persona_list):
            if isinstance(laptop_persona_list, list): # Ensure it's a list
                return any(p_item.lower() in user_personas_lower for p_item in laptop_persona_list if isinstance(p_item, str))
            return False
        
        # Ensure 'Persona' column exists and apply filter
        if 'Persona' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Persona'].apply(match_personas)]
        else: # Should not happen if preprocessing is correct
            return json.dumps({"status":"error", "message":"Persona data is missing."})


    if not filtered_df.empty:
        # Select a subset of columns for recommendation to keep it concise
        recommendation_cols = ['Brand', 'Model Name', 'Price', 'RAM Size', 'Graphics Processor', 'Persona', 'Description']
        # Ensure selected columns exist
        existing_cols = [col for col in recommendation_cols if col in filtered_df.columns]
        recommendations = filtered_df[existing_cols].head(5).to_dict(orient='records')
        return json.dumps({"status": "success", "count": len(filtered_df), "data": recommendations})
    else:
        return json.dumps({"status": "not_found", "message": "Sorry, no laptops found matching your criteria. You might want to broaden your search."})

def end_conversation():
    """Signals the end of the conversation."""
    return json.dumps({"status": "ended", "message": "Okay, ending the conversation. If you need help again, just ask. Goodbye!"})


def get_available_functions_map():
    """Returns a map of function names to function objects."""
    return {
        "get_laptop_info": get_laptop_info,
        "recommend_laptops_by_criteria": recommend_laptops_by_criteria,
        "end_conversation": end_conversation,
    }

def get_tools_definition():
    """Returns the list of tool definitions for the OpenAI API."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_laptop_info",
                "description": "Use this function when the user wants to know about a specific product (laptop model).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "description": "The specific model name of the laptop the user is asking about (e.g., 'MacBook Air M2', 'Inspiron 15')."}
                    },
                    "required": ["model_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_laptops_by_criteria",
                "description": "Use this function when the user wants to buy a product or asks for laptop recommendations. Capture their budget and persona preferences if provided.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # "budget_min": {"type": "integer", "description": "The minimum budget (e.g., 30000). Optional."},
                        "budget_max": {"type": "integer", "description": "The maximum budget (e.g., 80000). Optional."},
                        "personas": {
                            "type": "array", "items": {"type": "string", "enum": config.PERSONA_VALUES},
                            "description": f"A list of user personas. Available options: {config.PERSONA_VALUES}. Optional."
                        },
                    },
                    "required": [], # Parameters are optional
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "Use this function if the user explicitly states they want to end the conversation, says 'goodbye', 'exit', or 'thanks, that's all'.",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    ]