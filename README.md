# Fast LLM Context Loading by Streaming Layerwise Compressed KV Cache

This is the code repo for the project 'Fast LLM Context Loading by Streaming Layerwise Compressed KV Cache'. This code is written based on [CacheGen: Fast Context Loading for Language Model Applications via KV Cache Streaming](https://arxiv.org/pdf/2310.07240.pdf) (SIGCOMM'24). 

The code structure is organized as follows:

- ```LMCache```: The modules for KV cache encoding / decoding with CacheGen's customized codec 
- ```test_data```: The example testing cases for CacheGen. 
- ```src```: Some helper functions used by CacheGen (e.g., transforming tensor to tuple, transforming tuple to tensor etc.)

## Installation

```
conda env create -f env.yaml
conda activate cachegen
pip install -e LMCache
cd LMCache/third_party/torchac_cuda 
python setup.py install
```

