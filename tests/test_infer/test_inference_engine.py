import random

import numpy as np
import pytest
import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig

import colossalai
from colossalai.inference.config import InferenceConfig
from colossalai.inference.core.engine import InferenceEngine
from colossalai.testing import spawn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def check_inference_engine(test_cai=False):
    setup_seed(20)
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=50000, hidden_size=512, intermediate_size=1536, num_attention_heads=4, num_hidden_layers=4
        )
    ).cuda()

    inputs = [
        "介绍一下北京,",
        "介绍一下武汉,",
    ]

    if test_cai:
        inference_config = InferenceConfig(max_output_len=1)
        inference_engine = InferenceEngine(model, tokenizer, inference_config, verbose=True)
        inference_engine.add_request(prompts=inputs)
        assert inference_engine.request_handler._has_waiting()
        generation_config = GenerationConfig(do_sample=True, top_p=0.5, top_k=50)
        outputs = inference_engine.generate(generation_config)
    else:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer.batch_encode_plus(inputs, padding=True, return_tensors="pt")["input_ids"]
        inputs = inputs.cuda()
        generation_config = GenerationConfig(
            do_sample=True, top_p=0.5, top_k=50, pad_token_id=tokenizer.pad_token_id, max_new_tokens=1
        )
        outputs = model.generate(inputs, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return outputs


def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, port=port, host="localhost")
    cai_outputs = check_inference_engine(True)
    transformer_outputs = check_inference_engine(False)

    for s1, s2 in zip(cai_outputs, transformer_outputs):
        assert s1 == s2


@pytest.mark.dist
def test_inference_engine():
    spawn(run_dist, 1)


if __name__ == "__main__":
    test_inference_engine()