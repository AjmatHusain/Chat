from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chat():
    print("🤖 Chatbot is ready! Type 'exit', 'quit' or 'bye' to stop.")
    chat_history_ids = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Chatbot: Goodbye!")
            break

        # Encode user input and append to chat history
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)

        if chat_history_ids is None:
            chat_history_ids = new_input_ids
        else:
            chat_history_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)

        # Generate a response
        output_ids = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, do_sample=True, top_p=0.9, temperature=0.8)
        response_ids = output_ids[:, chat_history_ids.shape[-1]:]

        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        print(f"🤖 Chatbot: {response}")

        # Append generated response to chat history
        chat_history_ids = torch.cat([chat_history_ids, response_ids], dim=-1)

if __name__ == "__main__":
    chat()
