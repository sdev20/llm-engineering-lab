import json
import inspect
from config import API_MODEL
from llm_client import client
import gradio as gr
from brochure_generation import create_brochure, stream_brochure_for_ui_chat

model = API_MODEL


system_message = "You are a helpful assistant"


def brochure_releated_images(website_url):
    """
    Returns image URLs for a company’s brochure. Use when the user asks for pictures, images, or visuals.

    Args:
        website_url (str): The URL of the company website to generate a brochure for.
    """
    return "https://www.example.com/brochure_image.jpg"  # Placeholder for actual image generation logic


 #Map functions for the agent execution loop
AVAILABLE_FUNCTIONS = {
    #"stream_brochure_for_ui_chat": stream_brochure_for_ui_chat
    "create_brochure": create_brochure,
    "brochure_releated_images": brochure_releated_images
}

tools_schema = [
    # {
    #     "type" : "function",
    #     "function": 
    #     {
    #         "name": "stream_brochure_for_ui_chat",
    #         "description": "Generates a company sales brochure for a given company website URL.",
    #         "parameters":
    #         {
    #             "type":"object",
    #             "properties": 
    #             {
    #                 "website_url": 
    #                 {
    #                     "type": "string",
    #                     "description": "The URL of the company website to generate a brochure for."
    #                 }
    #             },
    #             "required": ["website_url"]
    #         }
    #     }
    # },
    {
        "type" : "function",
         "function": 
                {
                    "name": "create_brochure",
                    "description": "Generates a company sales brochure for a given company website URL.",
                    "parameters":
                    {
                        "type":"object",
                        "properties": 
                        {
                            "website_url": 
                            {
                                "type": "string",
                                "description": "The URL of the company website to generate a brochure for."
                            }
                        },
                        "required": ["website_url"]
                    }
                }
    },
    {
        "type" : "function",
        "function": 
                {
                    "name": "brochure_releated_images",
                    "description": "Provides images related to a company sales brochure for a given company website URL.",
                    "parameters":
                    {
                        "type":"object",
                        "properties": 
                        {
                            "website_url": 
                            {
                                "type": "string",
                                "description": "The URL of the company website to generate a brochure for."
                            }
                        },
                        "required": ["website_url"]
                    }
                }
        }
]






# def tool_calls_with_streaming_no_llm_at_the_end(reply, tool_calls_dict):
    # shown = reply + "\n\n" if reply else ""
    
    # for tool_call in tool_calls_dict.values():
    #     func_name = tool_call["function_name"]
    #     try:
    #         func_args = json.loads(tool_call["function_arguments"])
    #     except json.JSONDecodeError:
    #         shown += f"Sorry, I couldn't read the arguments for `{func_name}`.\n\n"
    #         yield shown
    #         continue
    
    #     # Retrieve the actual python function based on name
    #     function_to_call = AVAILABLE_FUNCTIONS.get(func_name)
    #     if function_to_call is None:
    #         shown += f"Sorry, I don't know the tool `{func_name}`.\n\n"
    #         yield shown
    #         continue
    
    #     heading = f"## {func_args.get('website_url', func_name)}\n\n"
    #     output = function_to_call(**func_args)
    
    #     if inspect.isgeneratorfunction(function_to_call):
    #         last = ""
    #         for partial in output:
    #             last = partial
    #             yield shown + heading + partial  # live streaming, previous tools' text stays
    #     else:
    #         last = str(output)
    #         yield shown + heading + last
    
    #     # Tool finished: lock its final text into `shown` before the next tool starts
    #     shown += heading + last + "\n\n---\n\n"

        

def chat(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]

    while True:
        response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                stream=True)

        reply = ""
        tool_calls_dict = {}
        for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    delta = chunk.choices[0].delta.content
                    reply += delta
                    yield reply
                if chunk.choices[0].delta.tool_calls is not None:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        index = tool_call.index
                        if tool_calls_dict.get(index) is None:
                            tool_calls_dict[index] = {
                                "id": tool_call.id,
                                "function_name": tool_call.function.name,
                                "function_arguments": tool_call.function.arguments or ""
                            }
                        else:
                            # Merge the new tool_call with the existing one
                            existing_tool_call = tool_calls_dict[index]
                            existing_tool_call["function_arguments"] += tool_call.function.arguments or ""

        print("Final tool call dictionary:", tool_calls_dict)
        # Gradio replaces the message on every yield, so keep a running string of everything shown so far
        #tool_calls_with_streaming_no_llm_at_the_end(reply=reply, tool_calls_dict=tool_calls_dict)
        if tool_calls_dict:
            # The API needs the assistant's tool request in `messages` before any `tool` result messages
            tool_calls_message = {
                "role": "assistant",
                "content": reply or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function_name"],
                            "arguments": tc["function_arguments"],  # keep the raw JSON string
                        },
                    }
                    for tc in tool_calls_dict.values()
                ],
            }
            messages.append(tool_calls_message)

            for tool_call in tool_calls_dict.values():
                func_name = tool_call["function_name"]
                func_args = json.loads(tool_call["function_arguments"])
        
                # Retrieve the actual python function based on name
                function_to_call = AVAILABLE_FUNCTIONS.get(func_name)
            
                if function_to_call:
                # Execute the function
                    tool_output = function_to_call(**func_args)
        
                    messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": tool_output
                        })
                else:
                    # If the function is not found, append an error message to messages
                    messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": func_name,
                            "content": f"Error: Function '{func_name}' not found."
                        })
        else:
            break  # No tool calls, exit the loop



view = gr.ChatInterface(fn=chat, 
                        title="GPT Chat Messanger", 
                        description="A simple chat interface to interact with GPT model.",
                        flagging_mode="never")

view.launch(inbrowser=True)



