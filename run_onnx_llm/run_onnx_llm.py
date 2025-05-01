# -*- coding:utf-8 -*-
"""
This is a script for running ONNX LLM models, including 'running with kv cache' and 'running without kv cache'."
"""
import sys
import logging
import argparse
import onnxruntime
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer

logging.basicConfig(format="[%(levelname)s] - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLLM:
    def __init__(self, onnx_file):
        # Create the ONNX session.
        logger.info(f"Creating ONNX session for [{onnx_file}].")
        if not onnx_file.endswith(".onnx"):
            logger.error(f"`{onnx_file}` is not a onnx file.")
            sys.exit(0)

        options = onnxruntime.SessionOptions()
        providers = ["CPUExecutionProvider"]
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.llm_session = onnxruntime.InferenceSession(
            onnx_file,
            sess_options=options,
            providers=providers
        )
        logger.info(f"Create ONNX session Done!")
        self.next_token = None
        self.pos = None  # The real position of next_token.

    def _prepare_input(self, inputs, outputs, use_kv_cache=False):
        # for next generation
        # 1.input_ids 2.attention_mask 3.kv_cache.
        logits = outputs[0]
        if logits.shape[1] > 1:
            # for prompt_processor: logits.shape=(1, max_tokens, vocab_size)
            next_token_logits = logits[:, self.pos - 1, :]
        else:
            # for token_generator: logits.shape=(1, 1, vocab_size)
            next_token_logits = logits[:, 0, :]

        next_token_scores = softmax(next_token_logits, axis=-1)
        self.next_token = np.argmax(next_token_scores, axis=-1)[0]

        if use_kv_cache:
            inputs["input_ids"] = self.next_token.reshape((1, 1))
            inputs["attention_mask"][:, self.pos] = 1
            for i in range(int((len(outputs) - 1) / 2)):
                key = outputs[i * 2 + 1][:, :, : self.pos, :]
                value = outputs[i * 2 + 2][:, :, : self.pos, :]
                inputs[f"past_key_values.{i}.key"] = key
                inputs[f"past_key_values.{i}.value"] = value
        else:
            inputs["input_ids"][:, self.pos] = self.next_token.reshape((1, 1))
            inputs["attention_mask"][:, self.pos] = 1
        return inputs

    def predict(self, inputs, use_kv_cache=False):
        outputs = self.llm_session.run(None, input_feed=inputs)
        kv_inputs = self._prepare_input(inputs, outputs, use_kv_cache)
        # update self.pos
        self.pos += 1
        return kv_inputs


class PromptProcessor(BaseLLM):
    def __init__(self, onnx_file):
        super().__init__(onnx_file)

    def pad_to_max_tokens(self, inputs, eos_token_id, pad_token_id=0, max_tokens=256):
        # This pad function only supports batch_size = 1.
        input_ids = inputs["input_ids"]
        if input_ids.shape[0] != 1:
            logger.error(f"The batch size({input_ids.shape[0]}) is not equal to 1.")
            sys.exit(0)
        attention_mask = inputs["attention_mask"]
        if self.pos is None:
            self.pos = input_ids.shape[1]
        new_shape = (input_ids.shape[0], max_tokens)
        pad_input_ids = eos_token_id * np.ones(new_shape, input_ids.dtype)
        pad_attention_mask = pad_token_id * np.ones(new_shape, attention_mask.dtype)

        pad_input_ids[:, 0: self.pos] = input_ids
        pad_attention_mask[:, 0: self.pos] = attention_mask
        pad_inputs = {"input_ids": pad_input_ids, "attention_mask": pad_attention_mask}

        return pad_inputs


class TokenGenerator(BaseLLM):
    def __init__(self, onnx_file):
        super().__init__(onnx_file)


def softmax(x: np.array, axis: int = -1):
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    sum_values = np.sum(exp_x_shifted, axis=axis, keepdims=True)
    return exp_x_shifted / sum_values


def run(
        inputs: Dict[str, np.ndarray],
        prompt_processor: PromptProcessor,
        eos_token_id: int,
        max_tokens: int = 256,
):
    """
    Run without kv cache, The input_ids update like this:
    prompt -> [2, 3, 4, 5, 2 ... 2] -> predict -> x1
           -> [2, 3, 4, 5, x1 ... 2] -> predict -> x2
            ...
    """
    for idx in range(max_tokens):
        # Update inputs
        inputs = prompt_processor.predict(inputs, use_kv_cache=False)
        next_token = prompt_processor.next_token
        pos = prompt_processor.pos
        yield next_token

        if next_token == eos_token_id or pos >= max_tokens:
            break


def run_with_kv_cache(
        inputs: Dict[str, np.ndarray],
        prompt_processor: PromptProcessor,
        token_generator: TokenGenerator,
        eos_token_id: int,
        max_tokens: int = 256,
):
    """
    Run with kv cache has 2 stage, the input_ids and kv cache update like this:
    1. prompt process stage
    prompt -> [2, 3, 4, 5, 2 ... 2] -> prompt_processor.predict -> x1, kv cache1

    2. autoregressive stage
        -> [x1], kv cache1 -> token_generator.predict -> x2, kv cache2
        -> [x2], kv cache2 -> token_generator.predict -> x3, kv cache3
            ...
    """
    for idx in range(max_tokens):
        if idx == 0:
            # 1. Prompt process stage
            kv_inputs = prompt_processor.predict(inputs, use_kv_cache=True)
            next_token = prompt_processor.next_token
            pos = token_generator.pos = prompt_processor.pos
            # The prompt_processor only runs once, so it can be deleted after running.
            del prompt_processor
        else:
            # 2. Autoregressive stage
            # Update kv_inputs
            kv_inputs = token_generator.predict(kv_inputs, use_kv_cache=True)
            next_token = token_generator.next_token
            pos = token_generator.pos

        yield next_token

        if next_token == eos_token_id or pos >= max_tokens:
            break


def main(
        prompt: str,
        model_path: str,
        model_with_kv_path: str,
        tokenizer_path: str,
        max_tokens: int = 256,
        use_kv_cache: bool = False,
):
    if use_kv_cache and not model_with_kv_path:
        logger.error(f"`use_kv_cache` is True while `model_with_kv_path` is {model_with_kv_path}.")
        sys.exit(0)
    logger.info(f"`use_kv_cache` is {use_kv_cache}.")
    logger.info(f"`max_tokens` is {max_tokens}.")

    input_list = [prompt]
    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    # Encoding prompt to token_id.
    inputs = tokenizer.batch_encode_plus(input_list, padding=True, return_tensors="np")
    inputs = dict(inputs)
    eos_token_id = tokenizer.eos_token_id

    # Load prompt_processor.
    prompt_processor = PromptProcessor(model_path)

    # Check whether onnx model input is valid.
    for inputs_meta in prompt_processor.llm_session._inputs_meta:
        if inputs_meta.name == "input_ids":
            x_shape = inputs_meta.shape
            batch_size, seq_len = x_shape
            if isinstance(seq_len, str):
                logger.info("Detected dynamic input shape in onnx model!")
            elif isinstance(seq_len, int) and seq_len != max_tokens:
                logger.warning(f"The input seq_len({seq_len}) of onnx model "
                               f"is not equal to max_tokens({max_tokens}), set max_tokens({seq_len}).")
                max_tokens = seq_len

    # Pad to max tokens.
    pad_inputs = prompt_processor.pad_to_max_tokens(
        inputs,
        eos_token_id,
        pad_token_id=0,
        max_tokens=max_tokens
    )

    if use_kv_cache:
        # Load token_generator and run with kv cache.
        token_generator = TokenGenerator(model_with_kv_path)
        gen_next = run_with_kv_cache(
            pad_inputs,
            prompt_processor,
            token_generator,
            eos_token_id,
            max_tokens
        )
    else:
        # Run without kv cache.
        gen_next = run(
            pad_inputs,
            prompt_processor,
            eos_token_id,
            max_tokens
        )

    result = []
    old_text = ""
    for next_token in gen_next:
        result.append(next_token)
        text = tokenizer.decode(result, skip_special_tokens=True)
        print(text[len(old_text):], end="", flush=True)
        old_text = text
    print()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="\n<|user|>\nWho are you?</s>\n<|assistant|>\n",
        help="The prompt from user."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="The onnx path of LLM model without kv cache."
    )
    parser.add_argument(
        "-mkv",
        "--model_with_kv",
        type=str,
        help="The onnx path of LLM model with kv cache."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_path",
        type=str,
        required=True,
        help="The folder containing tokenizer.model and config.json"
    )
    parser.add_argument(
        "-c",
        "--use_kv_cache",
        action="store_true",
        help="Wheather run with kv cache."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="The max tokens"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.prompt,
        args.model,
        args.model_with_kv,
        args.tokenizer_path,
        args.max_tokens,
        args.use_kv_cache,
    )
# Usage example:
# run with kv cache
# python3 run_llm.py -m ./with_kv_cache_2048/rank_0__decoder_model_fp32.onnx -mkv ./with_kv_cache_2048/rank_0__decoder_with_past_model_fp32.onnx -t ./gemma_tokenizer --max_tokens 2048 -c

# run without kv cache
# python3 run_llm.py -m ./with_kv_cache_2048/rank_0__decoder_model_fp32.onnx -t ./gemma_tokenizer --max_tokens 2048
