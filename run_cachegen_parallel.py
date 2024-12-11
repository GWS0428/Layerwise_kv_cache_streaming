from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import os
import time
import pickle
import torch
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
from model.mistral_custom import MistralForCausalLM_custom

def worker_decode(q: Queue, encoded_dir: str, doc_id: int, layer: int, model_id: str, kv_tokens: list):
    with open(f"{encoded_dir}/{doc_id}_layer_{layer}.pkl", "rb") as f:
        encoded_bytes = pickle.load(f)
    lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_tokens[doc_id])
    meta_data = LMCacheEngineMetadata(model_name=model_id, fmt="huggingface", world_size=1, worker_id=0)
    deserializer = CacheGenDeserializer(lmcache_config, meta_data)
    decoded_kv = deserializer.from_bytes(encoded_bytes)
    q.put((doc_id, layer, decoded_kv))


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    p = argparse.ArgumentParser()

    p.add_argument("--model_id", type=str, default="lmsys/longchat-7b-16k")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
    p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=1)
    p.add_argument("--encoded_dir", type=str, default=None)
    p.add_argument("--results_dir", type=str, default=None)
    p.add_argument("--results_str", type=str, default="results")
    p.add_argument("--dataset_name", type=str)
    p.add_argument("--calculate_metric", type=int)

    args = p.parse_args()

    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)

    data = load_testcases(DATASET_TO_PATH[args.dataset_name])
    os.environ["QUANT_LEVEL"] = "2"

    num_layers = 32
    kv_tokens = []
    avg_size = []

    kv_example = torch.load(f"{args.save_dir}/raw_kv_{args.start}_layer_0.pt")
    layer_to_device_id = {}

    device_index = kv_example.device.index
    for l in range(num_layers):
        layer_to_device_id[l] = device_index

    for doc_id in range(args.start, args.end):
        kv_first_layer = torch.load(f"{args.save_dir}/raw_kv_{doc_id}_layer_0.pt")
        chunk_size = kv_first_layer.shape[-2]
        kv_tokens.append(chunk_size)

        for layer in range(num_layers):
            key_value = torch.load(f"{args.save_dir}/raw_kv_{doc_id}_layer_{layer}.pt")
            lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
            meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0)
            cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
            encoded_bytes = cachegen_serializer.to_bytes(key_value)
            pickle.dump(encoded_bytes, open(f"{args.encoded_dir}/{doc_id}_layer_{layer}.pkl", "wb"))

            if layer == 0:
                avg_size.append(len(encoded_bytes)/1e6)

    #model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)

    model = MistralForCausalLM_custom.from_pretrained("mistral-community/Mistral-7B-v0.2").cuda()
    tokenizer = AutoTokenizer.from_pretrained("mistral-community/Mistral-7B-v0.2")

    decoded_kvs = []
    average_acc = []

    for doc_id in range(args.start, args.end):
        os.environ['DOC_ID'] = str(doc_id)
        print("Running inference for doc_id:", doc_id)

        q = Queue()

        layer = 0
        p = Process(target=worker_decode, args=(q, args.encoded_dir, doc_id, layer, args.model_id, kv_tokens))
        p.start()

        for layer in range(num_layers):
            print(layer)
            curr_doc_id, curr_layer, decoded_kv = q.get()
            assert curr_doc_id == doc_id and curr_layer == layer, \
                f"Mismatch in doc_id or layer: expected (doc_id={doc_id}, layer={layer}) but got (doc_id={curr_doc_id}, layer={curr_layer})"

            if layer + 1 < num_layers:
                next_layer = layer + 1
                p = Process(target=worker_decode, args=(q, args.encoded_dir, doc_id, next_layer, args.model_id, kv_tokens))
                p.start()

            decoded_kv = decoded_kv.cuda()
            decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)

            text = data[doc_id]['prompt']
            input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()

            import IPython; IPython.embed()
            #output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
            prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            if args.calculate_metric == 1:
                if args.dataset_name == "longchat":
                    metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                    average_acc.append(metric)
                elif args.dataset_name in ["nqa", "tqa"]:
                    metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                    average_acc.append(metric)

            if args.dataset_name == "longchat":
                print(prediction, data[doc_id]['label'][0])

    if args.dataset_name == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"

    if args.calculate_metric == 1 and len(average_acc) > 0:
        print(f"Average cachegen {metric_name} is: ", np.mean(average_acc))

    if len(avg_size) > 0:
        print(f"Average size of KV cache: {np.mean(avg_size)}MB")
