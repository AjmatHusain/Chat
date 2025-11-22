import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the Flan-T5 model and tokenizer from Hugging Face model hub
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Converts text to tokens and back
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # The pre-trained sequence-to-sequence model

# Put the model in evaluation mode (disables training-specific operations like dropout)
model.eval()

def generate_response(input_text, context=""):
    """
    Generate a response given an input text and optional conversation context.
    
    Args:
        input_text (str): The current message from the user.
        context (str): Optional previous conversation or background information.
        
    Returns:
        str: The generated reply from the model.
    """
    # Create a prompt by combining context and user input, formatted like a conversation
    prompt = f"Context: {context}\nUser: {input_text}\nAssistant:"
    
    # Tokenize the prompt: convert the text into numerical IDs and create a tensor for the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # return_tensors="pt" tells the tokenizer to:
        # Convert the list of token IDs into a PyTorch tensor (hence "pt" for PyTorch).
        # This tensor can be directly fed into the PyTorch model for processing.

    # Generate output tokens without calculating gradients (faster and less memory usage)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,              # Input token IDs
            max_length=150,         # Maximum length of generated tokens including input
            num_beams=5,            # Beam search with 5 beams to improve generation quality
            no_repeat_ngram_size=2, # Prevent repeating sequences of 2 tokens
            early_stopping=True     # Stop generation early if the model decides output is complete
        )
    
    # Decode the generated token IDs back to readable text, skipping special tokens like <pad>
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return response

def main():
    print("Chatbot: Hello! I'm here to chat with you.")
    
    context = ""  # Optional: keep track of conversation history here
    
    while True:
        user_input = input("You: ")  # Get user input from console
        
        if user_input.lower() == "exit":  # Exit condition
            print("Chatbot: Goodbye!")
            break
        
        # Generate a response using current user input and context
        response = generate_response(user_input, context)
        
        print("Chatbot:", response)
        
        # Optional: update context by appending conversation history to provide continuity
        # context += f" User: {user_input} Assistant: {response}"

if __name__ == "__main__":
    main()
