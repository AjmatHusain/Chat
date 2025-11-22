# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face tools for loading the model and tokenizer
import gradio as gr  # type: ignore # Gradio is used for building a simple web-based UI
#pyright: ignore[reportMissingImports]
# ===============================
# Load the Flan-T5 model and tokenizer from Hugging Face
# ===============================
model_name = "google/flan-t5-large"  # Pretrained model name
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Loads the tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Loads the model

# Set the model to evaluation mode (important for inference)
model.eval()

# ===============================
# Function to generate a response using the model
# ===============================
def generate_response(input_text, context=""):
    # Combine the conversation context and current user input into a prompt
    prompt = f"Context: {context}\nUser: {input_text}\nAssistant:"

    # Tokenize the prompt to convert it into model-readable IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # 'pt' means PyTorch tensor

    # Turn off gradient calculation since we're only doing inference
    with torch.no_grad():
        # Generate a response from the model
        output_ids = model.generate(
            input_ids,
            max_length=150,             # Limit the response length
            num_beams=5,                # Beam search with 5 beams for better responses
            no_repeat_ngram_size=2,     # Avoid repeating 2-gram phrases
            early_stopping=True         # Stop generation when all beams end
        )

    # Decode the generated token IDs back into text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

# ===============================
# Function to handle multi-turn chat logic
# ===============================
def chat(user_input, history=None):
    if history is None:
        history = []  # Initialize history if it's the first message

    # Build the conversation context from previous turns
    context = ""
    for turn in history:
        context += f"User: {turn[0]}\nAssistant: {turn[1]}\n"

    # Generate a new response based on current input and past context
    response = generate_response(user_input, context)

    # Add the new turn to the conversation history
    history.append((user_input, response))

    # Return updated history for display and storage
    return history, history

# ===============================
# Build the Gradio web UI
# ===============================
with gr.Blocks() as app:
    gr.Markdown("##  Flan-T5 Chatbot")  # Title/heading in the web app

    chatbot = gr.Chatbot()  # Component to display chat history
    msg = gr.Textbox(label="Your Message", placeholder="Type your message and press Enter")  # User input box
    state = gr.State([])  # Holds the chat history (list of messages)

    # When user submits a message:
    submit_btn = msg.submit(chat, [msg, state], [chatbot, state])  # Call 'chat' function and update the chatbot and state

    # Clear the input box after submission
    submit_btn.then(lambda: "", None, msg)

# ===============================
# Launch the web application
# ===============================
app.launch()
