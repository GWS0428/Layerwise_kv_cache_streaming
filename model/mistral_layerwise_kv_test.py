import os
import pickle

import torch
from transformers import AutoTokenizer
from mistral_custom import MistralForCausalLM_custom
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from src.utils import tensor_to_tuple


def test():
    os.environ["QUANT_LEVEL"] = "2"
    model = MistralForCausalLM_custom.from_pretrained("mistral-community/Mistral-7B-v0.2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2")

    prompt = "Hey, are you conscious? Can you talk to me?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    next_token_input_ids = input_ids.clone().cuda()  # Clone input_ids to avoid mutation
    max_new_tokens = 20  
        
    # Perform greedy decoding
    model.eval()
    predicted_ids = []
    
    # load kv cache
    layer_to_device_id = {}
    kv_raw_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mistral7b_longchat_data")
    kv_enc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "encoded")
    doc_id = 0
    kv_list = []
    
    kv_raw_file_path = os.path.join(kv_raw_dir, f"raw_kv_0_layer_0.pt")
    # kv_raw = pickle.load(open(kv_raw_file_path))
    kv_raw = torch.load(kv_raw_file_path)
    for layer in range(32):  # Assuming there are 32 layers
        layer_to_device_id[layer] = kv_raw[0][0].device.index # TODO: Check if this is correct
        
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_raw.shape[-2])
        meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        
        bytes = pickle.load(open(f"{kv_enc_dir}/{doc_id}_layer_{layer}.pkl", "rb"))
        decoded_kv = deserializer.from_bytes(bytes)
        decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        kv_list.append(decoded_kv)
        print(f"Layer {layer} KV cache loaded")
    print("KV cache loaded")
        
    outputs = None
    for i, kv in enumerate(kv_list):
        if i == 0:
            outputs = model.forward_1(input_ids=next_token_input_ids, past_key_values=kv)
        outputs = model.forward_2(idx=i, previous_outputs=outputs, past_key_values_layer=kv)
        print(f"Layer {i} done")
    outputs = model.forward_3(previous_outputs=outputs)
    outputs = model.forward_4(previous_outputs=outputs)
    logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
            
    exit()
    
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
    
    
if __name__ == "__main__":
    test()