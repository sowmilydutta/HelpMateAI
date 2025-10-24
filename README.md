# 🤖 ShopAssist AI 2.0: Function-Calling Laptop Advisor

ShopAssist AI 2.0 is an intelligent chatbot designed to provide accurate and personalized laptop recommendations. Leveraging the **OpenAI Function Calling API**, this system transforms the traditional conversational interface into a highly robust and efficient application capable of interacting directly with a structured laptop product database.

The system helps users navigate a complex catalog by gathering criteria like budget and user persona (e.g., gamer, student, developer) and executing specialized Python functions to filter and retrieve the best matches.

## ✨ Core Features and Enhancements

Version 2.0 implements a modular and robust architecture, featuring several key enhancements over previous iterations:

| Feature | Benefit |
| :--- | :--- |
| **OpenAI Function Calling** | Enables the LLM to precisely identify user intent and execute dedicated functions (`get_laptop_info`, `recommend_laptops_by_criteria`), drastically improving accuracy and reliability over manual text parsing. |
| **LLM-Powered Data Preprocessing** | Utilizes an LLM (`gpt-3.5-turbo`) to automatically classify laptop specs (e.g., "GPU intensity: high") and assign relevant **Persona Tags** based on product descriptions, enhancing filtering capability. |
| **Modular Architecture** | Code is separated into well-defined modules (`config`, `llm_utils`, `chatbot_functions`), improving maintainability, testability, and scalability. |
| **Data Caching** | Preprocessed and tagged laptop data is cached to `laptops_preprocessed.csv`, preventing redundant and costly LLM preprocessing calls on subsequent runs. |
| **Robust Error Handling** | Comprehensive error management is implemented for API failures, JSON parsing issues (even when the LLM outputs messy JSON), and unexpected function arguments, ensuring a smoother user experience. |
| **Dynamic Tool Definition** | The function definitions are dynamically generated from `config.py`, making it easy to add or modify available user personas (`PERSONA_VALUES`). |

## 🏗️ System Architecture

The project is structured into five integrated modules:

| File | Role | Description |
| :--- | :--- | :--- |
| `main_chatbot.py` | **Orchestration** | Implements the main conversation loop, manages history, sends prompts, handles function execution results, and directs the overall conversational flow based on the System Prompt. |
| `chatbot_functions.py` | **Tool Definitions** | Defines the Python functions (`get_laptop_info`, `recommend_laptops_by_criteria`) that the LLM is allowed to call. Also generates the necessary JSON schema for the OpenAI API. |
| `laptop_data_manager.py` | **Data & Preprocessing** | Manages the entire laptop data lifecycle. Handles raw CSV loading, LLM-based spec rating (`_product_map_layer`), persona tagging (`_persona_tag`), and data caching. |
| `llm_utils.py` | **API Abstraction** | Provides robust wrappers for `openai.ChatCompletion` calls, specializing in low-temperature JSON extraction for preprocessing and standard conversational responses. |
| `config.py` | **Configuration** | Centralized control file for API keys, model IDs, file paths, persona lists, and debugging flags (`REPROCESS_DATA`). |

---

## 🚀 Getting Started

### Prerequisites

1.  **Python:** Requires Python 3.x.
2.  **Dependencies:** Install libraries listed in the project's requirements (primarily `openai`, `pandas`).
3.  **Data:** A CSV file named `laptop_data.csv` must be present in the root directory.
4.  **OpenAI API Key:** Required for all LLM interactions.

### 1. Configuration

Set up your API key and file paths in `config.py`:

```python
# config.py

OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Ensure this path matches the location of your raw data
LAPTOP_DATA_CSV = 'laptop_data.csv' 
```

### 2. Data Initialization (First Run)

The first time you run the chatbot, the system will execute the **LLM-based preprocessing pipeline** to assign specification ratings and persona tags to every laptop, which can take some time. On subsequent runs, the cached file (`laptops_preprocessed.csv`) will be loaded instantly.

### 3. Run the Chatbot

Execute the main file to start the conversational interface:

```bash
python main_chatbot.py
```

## 💻 Usage Walkthrough

The chatbot uses the function calling logic to manage requests.

### Example Conversation Flow

| Step | User Input | LLM Action (Internal) |
| :--- | :--- | :--- |
| **1 (Goal)** | "I need a new laptop. I'm a student and my budget is around 60000." | Calls `recommend_laptops_by_criteria(budget_max=60000, personas=['student'])` |
| **2 (Info)** | "What about the MacBook Air M2?" | Calls `get_laptop_info(model_name='MacBook Air M2')` |
| **3 (Exit)** | "Thanks, that's all for now. Goodbye." | Calls `end_conversation()` |

The chatbot will interpret the user's intent, execute the appropriate function, and then synthesize the function's JSON output into a natural, conversational response for the user.
