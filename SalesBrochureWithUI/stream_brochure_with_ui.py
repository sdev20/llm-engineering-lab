from config import API_MODEL
from llm_client import client
import gradio as gr
from brochure_generation import stream_brochure_for_ui_chat

model = API_MODEL
system_assistant = "You are a helpful assistant that responds in markdown without code blocks."


def message_gpt(prompt):
    messages = [{"role": "system", "content": system_assistant},
                {"role": "user", "content": prompt}
                ]
    response = client.chat.completions.create(
        model = model,
        messages=messages)

    return response.choices[0].message.content


def stream_gpt(prompt):
    messages = [{"role": "system", "content": system_assistant},
                {"role": "user", "content": prompt}
                ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True)
    result = ""
    for chunk in response:
         if chunk.choices[0].delta.content is not None:
            delta = chunk.choices[0].delta.content
            result += delta
            yield result


message_input = gr.Textbox(label="Company URL:", placeholder="Enter the company URL here...")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_brochure_for_ui_chat,
    title="GPT Message Shouter",
    inputs=[message_input], 
    outputs=[message_output],
    examples=[["Explain Transformer architecture to an aspiring AI engineer."]],
    flagging_mode="never")

view.launch(inbrowser=True)