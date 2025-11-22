import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

# Load the Flan-T5 model and tokenizer
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_response(input_text, context=""):
    prompt = f"Context: {context}\nUser: {input_text}\nAssistant:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def chat(user_input, history=None):
    if history is None:
        history = []

    context = ""
    for turn in history:
        context += f"User: {turn[0]}\nAssistant: {turn[1]}\n"

    response = generate_response(user_input, context)
    history.append((user_input, response))

    return history, history

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("## 💬 Flan-T5 Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message", placeholder="Type your message and press Enter")
    state = gr.State([])

    submit_btn = msg.submit(chat, [msg, state], [chatbot, state])
    submit_btn.then(lambda: "", None, msg)  # This line clears the textbox

app.launch()
