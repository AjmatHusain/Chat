# Import necessary libraries Hugging Face Transformers
from transformers import  AutoModelForCausalLM, AutoTokenizer

# Specify the name of the pre-trained model to use
# "microsoft/DialoGPT-medium" is a version of GPT-2 fine-tuned for conversational dialogue
model_name = "microsoft/DialoGPT-medium"

# Load the tokenizer for the model (used to convert text to tokens and vice versa)
# Note: AutoTokenizer is a generic class that automatically selects the correct tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the pre-trained language model for causal language modeling (text generation)
# AutoModelForCausalLM is used for models that predict the next word in a sentence (like GPT-2 or DialoGPT)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode (disables dropout and similar training-specific behavior)
model.eval()

# Define a function to generate a response based on user input
def generate_response(input_text):
    # Encode the user's input and append the End Of Sentence (EOS) token
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # Generate a response using the model
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,               # Maximum number of tokens in the response
        pad_token_id=tokenizer.eos_token_id,  # Token to use for padding, these models are not trained with extra pad token so we use the eos itself.
        do_sample=True,               # Use sampling to generate more diverse responses
        temperature=0.9,              # Controls randomness: higher = more random
        top_p=0.9                     # Nucleus sampling: consider only top p probability mass
    )

    # Decode the output tokens to get the generated text (excluding the input part)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Define the main function to run the chatbot in a loop
def main():
    print("Chatbot: Hi there! How can I help you?")
    
    # Infinite loop to continuously interact with the user
    while True:
        # Take input from the user
        user_input = input("You: ")

        # If user types 'exit', end the chat
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        
        # Generate and display the chatbot's response
        response = generate_response(user_input)
        print("Chatbot:", response)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
