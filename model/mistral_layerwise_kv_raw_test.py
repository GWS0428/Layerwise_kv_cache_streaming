import os
# import pickle

import torch
from transformers import AutoTokenizer
from mistral_custom import MistralForCausalLM_custom
# from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
# from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
# from src.utils import tensor_to_tuple

def tensor_to_tuple(kv, layer_to_device_id):
    """ Convert a tensor to a list of tuples
    Input tensor's shape should be (num_layers, 2, num_heads, seq_len, heads_dim)
    """
    new_kv = []
    for i in range(len(kv)):
        new_kv.append((kv[i][0].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}"), 
                       kv[i][1].unsqueeze(0).to(f"cuda:{layer_to_device_id[i]}")))
    return tuple(new_kv)


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
        print(f"layer_to_device_id[{layer}]: {layer_to_device_id[layer]}")
        # lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_raw.shape[-2])
        # meta_data = LMCacheEngineMetadata(model_name="mistral-community/Mistral-7B-v0.2", fmt="huggingface", world_size=1, worker_id=0)
        # deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        
        raw_kv = torch.load(f"mistral7b_longchat_data/raw_kv_{doc_id}_layer_{layer}.pt")
        raw_kv.to("cuda")
        raw_kv = tensor_to_tuple(raw_kv, layer_to_device_id)
        # bytes = pickle.load(open(f"mistral7b_longchat_data/raw_kv_0_layer_{layer}.pt", "rb"))
        # decoded_kv = deserializer.from_bytes(bytes)
        # print(f"shape of decoded_kv: {decoded_kv.shape}")
        # decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        kv_list.append(raw_kv)
        print(f"Layer {layer} KV cache loaded")
    print("KV cache loaded")
        
    outputs = None
    with torch.no_grad():
        for i, kv in enumerate(kv_list):
            if i == 0:
                outputs = model.forward_1(input_ids=next_token_input_ids, past_key_values=kv)
            outputs = model.forward_2(idx=i, previous_outputs=outputs, past_key_values_layer=kv)
            print(f"Layer {i} done")
        outputs = model.forward_3(previous_outputs=outputs)
        outputs = model.forward_4(previous_outputs=outputs)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    
if __name__ == "__main__":
    test()