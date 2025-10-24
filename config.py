OPENAI_API_KEY = "****"

# File Paths
LAPTOP_DATA_CSV = 'laptop_data.csv'
PREPROCESSED_LAPTOP_DATA_CSV = 'laptops_preprocessed.csv' # For caching

# Model IDs
HELPER_MODEL_ID = "gpt-3.5-turbo"  # Model for product_map_layer and persona_tag
CHATBOT_MODEL_ID = "gpt-3.5-turbo" # Model for the main chatbot logic

# Data Processing
REPROCESS_DATA = False # Set to True to force reprocessing of CSV data

# Chatbot Settings
MAX_HISTORY_MESSAGES = 15

# Persona Values
PERSONA_VALUES = [
    "gamer", "student", "business", "professional_creator",
    "developer", "casual_user", "traveler", "budget_conscious"
]

# Output Delimiters / Markers for Prompts (if needed globally)
PROMPT_DELIMITER = "#####"