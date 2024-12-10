from transformers import AutoTokenizer, MistralForCausalLM
from model.mistral_custom import MistralForCausalLM_custom
import torch


def test():
    model = MistralForCausalLM_custom.from_pretrained("mistral-community/Mistral-7B-v0.2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2")

    prompt = "Hey, are you conscious? Can you talk to me?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    next_token_input_ids = input_ids.clone().cuda()  # Clone input_ids to avoid mutation
    max_new_tokens = 20  
        
    # Perform greedy decoding
    model.eval()
    predicted_ids = []

    for i in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass: Get logits for the current input
                        
            # Forward layer by layer
            outputs = model.forward_1(input_ids=next_token_input_ids)
            for i, _ in enumerate(model.model.layers):
                outputs = model.forward_2(idx=i, previous_outputs=outputs)
            outputs = model.forward_3(previous_outputs=outputs)
            outputs = model.forward_4(previous_outputs=outputs)
            logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
            
            # Get the last token logits
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            # Perform argmax to get the most likely next token
            next_token_id = torch.argmax(next_token_logits).item()
            predicted_ids.append(next_token_id)
            
            # Print the decoded token
            print(tokenizer.decode([next_token_id]), end=" ", flush=True)
            
            # Append the predicted token to the input_ids for the next iteration
            next_token_input_ids = torch.cat(
                [next_token_input_ids, torch.tensor([[next_token_id]], device=next_token_input_ids.device)],
                dim=-1
            )
            
    # Print the final generated text
    output_text = tokenizer.decode(next_token_input_ids[0], skip_special_tokens=True)
    print("\n\nFinal Output:\n", output_text)

    # Generate
    generate_ids = model.generate(input_ids, max_length=30)
    prediction = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(prediction)
    
    
if __name__ == "__main__":
    test()