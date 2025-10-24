# main_chatbot.py
import json
import config
import llm_utils
import laptop_data_manager
import chatbot_functions

def run_chatbot():
    print("Initializing Laptop Advisor Chatbot...")
    laptop_data_manager.initialize_data(force_reprocess=config.REPROCESS_DATA)
    
    df = laptop_data_manager.get_laptop_dataframe()
    if df is None or df.empty:
        print("Exiting: Laptop data could not be loaded or processed.")
        return

    system_prompt = f"""You are "Laptop Advisor", a friendly and expert chatbot helping users find laptops.
        Follow this flow:
        1.  Initialize Conversation: Greet the user.
        2.  Capture User Intent: Understand if they want to (a) know about a specific product, or (b) buy a product (get recommendations).
        3.  Decision based on Intent:
            *   If (a) "know about a Product": Use `get_laptop_info` function. You must get the `model_name`.
            *   If (b) "buy a product": Use `recommend_laptops_by_criteria`.
                *   Politely ask for their budget and persona(s) if not provided. Available personas: {config.PERSONA_VALUES}.
                *   The function will filter laptops.
            *   If the user's query seems harmful or completely off-topic: Politely decline to engage with the harmful part and steer back to laptops. If they persist with harmful or abusive language, you can use `end_conversation`.
        4.  Products Found?:
            *   If `get_laptop_info` finds a product, present its details clearly and concisely.
            *   If `recommend_laptops_by_criteria` finds products, recommend them, highlighting key features like Brand, Model, Price, and main specs.
            *   If no products are found, inform the user with the message from the function.
        5.  Exit?: After providing information or recommendations, ask "Is there anything else I can help you with, or would you like to look for other options or exit?".
            *   If they want to exit or say thanks/goodbye, use `end_conversation`.
            *   Otherwise, go back to "Capture User Intent".

        Always be concise and helpful. When presenting laptop data, format it nicely.
        Do not make up information. Rely on the function outputs.
        If a function call returns an error or unexpected data, inform the user you encountered an issue and try to proceed or ask for clarification.
    """
    messages = [{"role": "system", "content": system_prompt}]

    initial_greeting = "Hello! I'm Laptop Advisor. How can I help you with laptops today? You can ask about a specific model, get recommendations, or type 'exit' to end our chat."
    print(f"Laptop Advisor: {initial_greeting}")
    messages.append({"role": "assistant", "content": initial_greeting})

    available_funcs_map = chatbot_functions.get_available_functions_map()
    tools_def = chatbot_functions.get_tools_definition()

    while True:
        user_input = input("You: ")
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

        try:
            response_message = llm_utils.get_chatbot_completion(
                messages=messages,
                tools=tools_def,
                tool_choice="auto"
            )
            messages.append(response_message) # Add assistant's response/tool_call

            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_funcs_map.get(function_name)
                    
                    if not function_to_call:
                        print(f"Laptop Advisor: I'm sorry, I tried to use an unknown action: {function_name}")
                        tool_response_content = json.dumps({"error": f"Unknown function '{function_name}' requested by model."})
                    else:
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                            print(f"Calling function: {function_name} with args: {function_args}")
                            tool_response_content = function_to_call(**function_args)
                        except json.JSONDecodeError:
                            error_msg = f"Invalid arguments format provided by model for {function_name}."
                            print(f"Laptop Advisor: {error_msg}")
                            tool_response_content = json.dumps({"error": error_msg, "arguments_received": tool_call.function.arguments})
                        except TypeError as e:
                            error_msg = f"Argument mismatch for {function_name}: {e}"
                            print(f"Laptop Advisor: {error_msg}")
                            tool_response_content = json.dumps({"error": error_msg, "arguments_received": tool_call.function.arguments})
                        except Exception as e:
                            error_msg = f"An unexpected error occurred while executing {function_name}: {e}"
                            print(f"Laptop Advisor: {error_msg}")
                            tool_response_content = json.dumps({"error": error_msg})
                    
                    function_response_data = json.loads(tool_response_content) # For internal logic (e.g., end_conversation)

                    if function_name == "end_conversation" and function_response_data.get("status") == "ended":
                        print(f"Laptop Advisor: {function_response_data.get('message', 'Goodbye!')}")
                        return # End conversation

                    messages.append(
                        {"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": tool_response_content}
                    )
                
                # Get final response from LLM after tool execution
                second_response_message = llm_utils.get_chatbot_completion(messages=messages) # No tools needed here generally
                final_response_content = second_response_message.content
                print(f"Laptop Advisor: {final_response_content}")
                messages.append({"role": "assistant", "content": final_response_content})

            else: # No tool call, direct response from LLM
                if response_message.content:
                    print(f"Laptop Advisor: {response_message.content}")
                else: 
                    print("Laptop Advisor: I'm not sure how to respond to that. Can you try rephrasing?")
                    messages.append({"role": "assistant", "content": "I'm not sure how to respond to that."})
            
            # History pruning
            if len(messages) > config.MAX_HISTORY_MESSAGES:
                # Keep system prompt, then the last N messages
                messages = [messages[0]] + messages[-(config.MAX_HISTORY_MESSAGES - 1):]

        except Exception as e:
            print(f"Laptop Advisor: An critical error occurred in the main loop: {e}")
            # Simple recovery: pop last user message if it might have caused issue and offer to restart or try again
            if messages and messages[-1]["role"] == "user": 
                messages.pop() 
            print("Laptop Advisor: I'm having some trouble. Please try rephrasing your request or type 'exit'.")


if __name__ == "__main__":
    run_chatbot()