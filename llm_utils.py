# llm_utils.py
import os
import json
import re
import openai
from openai import OpenAI
import config


client = OpenAI(api_key=config.OPENAI_API_KEY)# llm_utils.py

def get_chat_completions_for_preprocessing(messages, json_format=False, model=config.HELPER_MODEL_ID):
    """OpenAI API call wrapper for the preprocessing helper functions."""
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0, # Low temperature for deterministic extraction
        }
        if json_format:
            params["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content
        print(content)

        if json_format:
            try:
                # LLM might wrap JSON in markdown or have leading/trailing text
                match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    # Try to find JSON directly if no markdown block
                    json_match = re.search(r"\{.*\}", content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        json_str = content # Fallback if no clear JSON structure found

                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse JSON response for preprocessing: '{content}'. Error: {e}")
                return {"error": "Failed to parse JSON", "raw_content": content}
        return content
    except Exception as e:
        print(f"Error in get_chat_completions_for_preprocessing: {e}")
        return {"error": str(e)} if json_format else f"Error: {str(e)}"


def get_chatbot_completion(messages, tools=None, tool_choice="auto", model=config.CHATBOT_MODEL_ID):
    """OpenAI API call wrapper for the main chatbot logic."""
    try:
        params = {
            "model": model,
            "messages": messages,
            "temperature": 0.7, # Standard temperature for chatbot
        }
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message
    except Exception as e:
        print(f"Error in get_chatbot_completion: {e}")
        # Return a mock message object or raise error, depending on desired handling
        return openai.types.chat.ChatCompletionMessage(role="assistant", content=f"Sorry, an error occurred: {e}")