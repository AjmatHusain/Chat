# pip install transformers
# pip install torch
#pip install huggingface-hub

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the GODEL model and tokenizer
model_name = "microsoft/GODEL-v1_1-large-seq2seq"  # Example model; adjust if necessary
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_response(input_text, grounding_text=""):
    """
    Generate a response from GODEL based on the input text and optional grounding context.
    grounding_text is the contextual knowledge or background information the model can use.
    """
    # Combine user input and grounding information
    conversation_input = f"[CONTEXT] {grounding_text} [USER] {input_text} [SYSTEM]"
    input_ids = tokenizer.encode(conversation_input, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        chat_history_ids = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    response = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
    return response

def main():
    print("Chatbot: Hi there! How can I help you?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Optional grounding context (could be updated to pull from a knowledge base)
        grounding_text = ""  # Set grounding context here if available
        response = generate_response(user_input, grounding_text)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
