"""
Batched LLM Inference Engine
Single responsibility: Convert conversations -> responses
Supports: OpenAI API, HuggingFace Transformers, vLLM
"""

import time
from typing import List, Dict, Any, Optional
from multiprocessing.pool import ThreadPool
from openai import OpenAI, OpenAIError


class BatchedLLMInference:
    """
    Unified inference engine for batched LLM generation.
    Supports GPT API, HuggingFace transformers, and vLLM backends.

    Usage:
        # Initialize once - model loaded into memory
        inference = BatchedLLMInference(
            model_name="gpt-4",
            backend="gpt",
            temperature=0.7,
            batch_size=8
        )

        # Use many times - no reloading
        conversations = [[{"role": "user", "content": "Hello"}], ...]
        responses = inference.generate(conversations)
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "gpt",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
        num_gpus: int = 1,
        no_cuda: bool = False,
        # Model-specific settings
        enable_thinking: bool = False,
        gpt_reasoning: str = "medium",
        # vLLM-specific settings
        vllm_multiproc_method: str = "spawn",
        vllm_max_num_seqs: Optional[int] = None,
        vllm_max_model_len: Optional[int] = None,
        vllm_gpu_memory_utilization: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize inference engine with model/client.

        Args:
            model_name: Model identifier (e.g., "gpt-4", "meta-llama/Llama-2-7b")
            backend: "gpt" (OpenAI API), "hf" (HuggingFace), or "vllm"
            temperature: Sampling temperature (0.0-1.0), None uses model-specific default
            max_tokens: Maximum tokens to generate, None uses model-specific default
            top_p: Top-p sampling parameter, None uses model-specific default
            top_k: Top-k sampling parameter, None uses model-specific default
            min_p: Min-p sampling parameter, None uses model-specific default
            batch_size: Batch size for generation (GPT: parallel threads, local: true batching)
            cache_dir: Directory for model downloads (local models only)
            num_gpus: Number of GPUs for vLLM tensor parallelism
            no_cuda: Force CPU usage even if CUDA available
            enable_thinking: Enable thinking/reasoning mode (Qwen3, GPT-OSS, CWM models)
            gpt_reasoning: GPT-OSS reasoning level ("low", "medium", "high")
            vllm_multiproc_method: vLLM multiprocessing start method ("spawn" or "fork", default: "spawn")
                                   Use "spawn" to avoid CUDA re-initialization errors
            vllm_max_num_seqs: Maximum number of sequences to process in a batch (vLLM only, default: 256)
            vllm_max_model_len: Maximum sequence length in tokens (vLLM only, default: model's context length)
            vllm_gpu_memory_utilization: Fraction of GPU memory to use (vLLM only, default: 0.9, range: 0.0-1.0)
        """
        self.model_name = model_name
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.num_gpus = num_gpus
        self.enable_thinking = enable_thinking
        self.gpt_reasoning = gpt_reasoning
        self.vllm_multiproc_method = vllm_multiproc_method
        self.vllm_max_num_seqs = vllm_max_num_seqs
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization

        # Model type detection
        self.is_deepseek = "deepseek" in model_name.lower()
        self.is_qwen3 = "qwen3" in model_name.lower()
        self.is_phi = "phi" in model_name.lower()
        self.is_gpt_oss = "gpt-oss" in model_name.lower()
        self.is_cwm = "cwm" in model_name.lower()

        print(f"Initializing BatchedLLMInference:")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(f"  Temperature: {temperature if temperature is not None else 'model default'}")
        print(f"  Max tokens: {max_tokens if max_tokens is not None else 'model default'}")
        print(f"  Top-p: {top_p if top_p is not None else 'model default'}")
        print(f"  Top-k: {top_k if top_k is not None else 'model default'}")
        print(f"  Min-p: {min_p if min_p is not None else 'model default'}")
        print(f"  Batch size: {batch_size}")

        if backend == "vllm":
            print(f"  vLLM Settings:")
            print(f"    max_num_seqs: {vllm_max_num_seqs if vllm_max_num_seqs is not None else 'default (256)'}")
            print(f"    max_model_len: {vllm_max_model_len if vllm_max_model_len is not None else 'model default'}")
            print(f"    gpu_memory_utilization: {vllm_gpu_memory_utilization if vllm_gpu_memory_utilization is not None else 'default (0.9)'}")

        # Initialize model/client based on backend
        if backend == "gpt":
            self._init_gpt_client()
        else:
            self._init_local_model(no_cuda)

        print("Initialization complete!")

    def _init_gpt_client(self):
        """Initialize OpenAI client for API calls."""
        import os

        if self.model_name.startswith("deepseek"):
            self.client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY")
            )
            print("Initialized DeepSeek API client")
        else:
            # Use official OpenAI API
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Please set it with: export OPENAI_API_KEY='your-key-here'"
                )

            self.client = OpenAI(api_key=api_key)
            print("Initialized OpenAI API client")

        self.model = None
        self.tokenizer = None
        self.device = None

    def _init_local_model(self, no_cuda: bool):
        """Initialize local model and tokenizer for HF/vLLM backends."""
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Set vLLM multiprocessing method BEFORE any CUDA operations
        # This fixes "Cannot re-initialize CUDA in forked subprocess" errors
        if self.backend == "vllm":
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = self.vllm_multiproc_method
            print(f"Set VLLM_WORKER_MULTIPROC_METHOD={self.vllm_multiproc_method}")

        self.device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        print(f"Using device: {self.device}")
        print(f"Loading model {self.model_name}...")

        try:
            # Load tokenizer
            tokenizer_kwargs = {"padding_side": "left"}
            if self.cache_dir:
                tokenizer_kwargs["cache_dir"] = self.cache_dir
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
            if self.tokenizer.pad_token is None:
                print("Warning: Tokenizer missing pad_token, setting to eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model based on backend
            if self.backend == "hf":
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if self.device.type == 'cuda' else torch.float32,
                }
                if self.cache_dir:
                    model_kwargs["cache_dir"] = self.cache_dir
                if self.device.type == 'cuda':
                    model_kwargs["device_map"] = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

                if self.device.type == 'cpu':
                    self.model.to(self.device)

            elif self.backend == "vllm":
                from vllm import LLM

                vllm_kwargs = {
                    "model": self.model_name,
                    "download_dir": self.cache_dir,
                    "tensor_parallel_size": self.num_gpus,
                    "trust_remote_code": True,
                    "disable_custom_all_reduce": True,
                }

                # Add optional vLLM-specific parameters if provided
                if self.vllm_max_num_seqs is not None:
                    vllm_kwargs["max_num_seqs"] = self.vllm_max_num_seqs
                if self.vllm_max_model_len is not None:
                    vllm_kwargs["max_model_len"] = self.vllm_max_model_len
                if self.vllm_gpu_memory_utilization is not None:
                    vllm_kwargs["gpu_memory_utilization"] = self.vllm_gpu_memory_utilization

                # Model-specific vLLM settings (can be overridden by explicit parameters)
                if self.is_deepseek and "max_model_len" not in vllm_kwargs:
                    vllm_kwargs["max_model_len"] = 32768

                self.model = LLM(**vllm_kwargs)

            print("Model loaded successfully")
            self.client = None

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "nl"
    ) -> List[str]:
        """
        Generate responses for a batch of conversations.

        Args:
            conversations: List of conversation histories
                Example: [
                    [{"role": "system", "content": "You are helpful"},
                     {"role": "user", "content": "What is 2+2?"}],
                    [{"role": "user", "content": "Explain gravity"}],
                    ...
                ]
            temperature: Override default temperature for this call
            max_tokens: Override default max_tokens for this call
            task_type: Task type hint ("nl" or "code") for model-specific tuning

        Returns:
            List of generated response strings, one per conversation
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.backend == "gpt":
            return self._generate_gpt_batch(conversations, temp, max_tok)
        else:
            return self._generate_local_batch(conversations, temp, max_tok, task_type)

    def _generate_gpt_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int
    ) -> List[str]:
        """Generate using GPT API with parallel ThreadPool."""
        pool = ThreadPool(processes=self.batch_size)

        start_time = time.time()
        print(f"Starting GPT API calls for {len(conversations)} conversations...")

        responses = pool.starmap(
            self._run_gpt4_single,
            [(conv, self.model_name, max_tokens, temperature) for conv in conversations]
        )

        pool.close()
        pool.join()

        end_time = time.time()
        print(f"API calls completed in {end_time - start_time:.2f} seconds")

        return responses

    def _run_gpt4_single(
        self,
        conversation: List[Dict[str, str]],
        engine: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Run single GPT API call with error handling."""
        try:
            if engine.startswith("gpt-4"):
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            elif engine.startswith("gpt-5"):
                # GPT-5 models: use max_completion_tokens and only support temperature=1
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    # temperature not configurable for GPT-5 nano (always 1)
                    max_completion_tokens=max_tokens if max_tokens is not None else 4096
                )
            elif engine.startswith("deepseek"):
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    temperature=0.0,
                    max_tokens=4096,
                    stream=False
                )
            else:
                # Default case
                response = self.client.chat.completions.create(
                    model=engine,
                    messages=conversation,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

            return response.choices[0].message.content.strip()

        except OpenAIError as exc:
            print(f"OpenAI error occurred: {exc}, retrying...")
            return "FAIL"

    def _generate_local_batch(
        self,
        conversations: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        task_type: str
    ) -> List[str]:
        """Generate using local model (HuggingFace or vLLM backend)."""
        import torch

        # Apply chat templates and prepare prompts
        input_texts = self._prepare_prompts(conversations)

        if self.backend == "hf":
            return self._generate_hf(input_texts, temperature, max_tokens, task_type)
        elif self.backend == "vllm":
            return self._generate_vllm(input_texts, temperature, max_tokens, task_type)

    def _get_sampling_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        task_type: str = "nl"
    ) -> Dict[str, Any]:
        """
        Resolve sampling parameters with priority:
        1. User-provided instance variables (self.temperature, self.top_p, etc.)
        2. Model-specific defaults
        3. General defaults

        Args:
            temperature: Override temperature for this call
            max_tokens: Override max_tokens for this call
            task_type: Task type hint (nl or code) for model-specific tuning

        Returns:
            Dictionary with resolved sampling parameters
        """
        # Start with model-specific defaults
        if self.is_qwen3:
            if self.enable_thinking:
                defaults = {
                    'temperature': 0.6,
                    'top_p': 0.95,
                    'top_k': 20,
                    'min_p': 0,
                    'max_tokens': 32768
                }
            else:
                defaults = {
                    'temperature': 0.7,
                    'top_p': 0.8,
                    'top_k': 20,
                    'min_p': 0,
                    'max_tokens': 32768
                }
        elif self.is_phi:
            defaults = {
                'temperature': 0.8,
                'top_p': 0.95,
                'top_k': None,
                'min_p': None,
                'max_tokens': 4096
            }
        elif self.is_gpt_oss:
            gpt_oss_temp = 0.3 if task_type == "code" else 1.0
            defaults = {
                'temperature': gpt_oss_temp,
                'top_p': 1.0,
                'top_k': None,
                'min_p': None,
                'max_tokens': 131072
            }
        elif self.is_cwm:
            defaults = {
                'temperature': 0.3,
                'top_p': 1.0,
                'top_k': 0,
                'min_p': None,
                'max_tokens': 2048
            }
        else:
            # Generic defaults
            defaults = {
                'temperature': 0.7,
                'top_p': 1.0,
                'top_k': 0,
                'min_p': None,
                'max_tokens': 8192
            }

        # Resolve parameters with priority: call param > instance var > model default
        params = {}

        # Temperature: call param > instance var > model default
        params['temperature'] = temperature if temperature is not None else (
            self.temperature if self.temperature is not None else defaults['temperature']
        )

        # Max tokens: call param > instance var > model default
        params['max_tokens'] = max_tokens if max_tokens is not None else (
            self.max_tokens if self.max_tokens is not None else defaults['max_tokens']
        )

        # Top-p: instance var > model default
        params['top_p'] = self.top_p if self.top_p is not None else defaults['top_p']

        # Top-k: instance var > model default
        params['top_k'] = self.top_k if self.top_k is not None else defaults['top_k']

        # Min-p: instance var > model default
        params['min_p'] = self.min_p if self.min_p is not None else defaults['min_p']

        return params

    def _prepare_prompts(self, conversations: List[List[Dict[str, str]]]) -> List[str]:
        """Apply model-specific chat templates to conversations."""

        def _merge_system_into_first_user(conversation):
            """Merge system message into the first user message, preserving all other turns."""
            system_content = ""
            new_conv = []
            for msg in conversation:
                if msg['role'] == 'system':
                    system_content = msg['content']
                else:
                    new_conv.append(dict(msg))
            if system_content and new_conv and new_conv[0]['role'] == 'user':
                new_conv[0]['content'] = f"{system_content}\n\n{new_conv[0]['content']}"
            return new_conv

        def _is_multiturn(conversation):
            return any(msg['role'] == 'assistant' for msg in conversation)

        # Handle models that don't support system messages
        if self.is_qwen3 or self.is_deepseek:
            modified_conversations = []
            for conversation in conversations:
                if _is_multiturn(conversation):
                    # Multi-turn: merge system into first user, keep full history
                    modified_conversations.append(_merge_system_into_first_user(conversation))
                else:
                    # Single-turn: original flatten behavior
                    system_msg = next((msg for msg in conversation if msg['role'] == 'system'), None)
                    user_msg = next((msg for msg in conversation if msg['role'] == 'user'), None)
                    system_content = system_msg['content'] if system_msg else ""
                    user_content = user_msg['content'] if user_msg else ""
                    combined_content = f"{system_content}\n\n{user_content}" if system_content else user_content
                    modified_conversations.append([{"role": "user", "content": combined_content}])

            if self.is_qwen3:
                return [
                    self.tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True,
                        enable_thinking=self.enable_thinking
                    )
                    for conv in modified_conversations
                ]
            else:  # DeepSeek
                return [
                    self.tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    for conv in modified_conversations
                ]

        # GPT-OSS with reasoning
        elif self.is_gpt_oss:
            modified_conversations = []
            for conversation in conversations:
                if _is_multiturn(conversation):
                    # Multi-turn: keep full history, patch system message for reasoning if needed
                    if self.enable_thinking:
                        print(f"Using Reasoning: {self.gpt_reasoning} for GPT-OSS")
                        new_conv = []
                        for msg in conversation:
                            if msg['role'] == 'system':
                                new_conv.append({'role': 'system', 'content': f"{msg['content']}\n\nReasoning: {self.gpt_reasoning}"})
                            else:
                                new_conv.append(msg)
                        modified_conversations.append(new_conv)
                    else:
                        modified_conversations.append(conversation)
                else:
                    # Single-turn: original behavior
                    system_msg = next((msg for msg in conversation if msg['role'] == 'system'), None)
                    user_msg = next((msg for msg in conversation if msg['role'] == 'user'), None)
                    system_content = system_msg['content'] if system_msg else ""
                    user_content = user_msg['content'] if user_msg else ""
                    if self.enable_thinking:
                        print(f"Using Reasoning: {self.gpt_reasoning} for GPT-OSS")
                        modified_conversations.append([
                            {'role': 'system', 'content': f"{system_content}\n\nReasoning: {self.gpt_reasoning}"},
                            {"role": "user", "content": user_content}
                        ])
                    else:
                        modified_conversations.append([
                            {'role': 'system', 'content': system_content},
                            {"role": "user", "content": user_content}
                        ])

            return [
                self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in modified_conversations
            ]

        # CWM with thinking
        elif self.is_cwm:
            return [
                self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True,
                    enable_thinking=self.enable_thinking
                )
                for conv in conversations
            ]

        # Standard chat template
        else:
            return [
                self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                for conv in conversations
            ]

    def _generate_hf(
        self,
        input_texts: List[str],
        temperature: float,
        max_tokens: int,
        task_type: str
    ) -> List[str]:
        """Generate using HuggingFace transformers."""
        import torch

        model_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True).to(self.device)
        input_ids = model_inputs.input_ids
        prompt_token_lengths = [len(input_ids[j]) for j in range(len(input_ids))]

        # Get resolved sampling parameters (user > model defaults > general defaults)
        params = self._get_sampling_params(temperature, max_tokens, task_type)

        # Model-specific max_tokens adjustments
        max_token_scaling = 10
        max_model_tokens = self.tokenizer.model_max_length

        if self.is_qwen3:
            print("Using special settings for Qwen models")
            max_model_tokens = 32768
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling)
            )
            print(f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}")
            print(f"Using temperature={params['temperature']} ({'thinking' if self.enable_thinking else 'non-thinking'} mode)")

        elif self.is_phi:
            print("Using special settings for Phi models")
            adjusted_max_tokens = 4096
            print(f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}")

        elif self.is_gpt_oss:
            print("Using special settings for GPT-OSS models")
            max_model_tokens = 131072
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling)
            )
            print(f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}")
            print(f"Using temperature={params['temperature']} for task_type={task_type}")

        elif self.is_cwm:
            print("Using custom settings for CWM generation")
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling)
            )
            print(f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}")
            print(f"Using temperature={params['temperature']}")

        else:
            print("Using generic settings for model generation")
            adjusted_max_tokens = min(
                max_model_tokens - max(prompt_token_lengths),
                int(max(prompt_token_lengths) * max_token_scaling)
            )
            print(f"Prompt length: max {max(prompt_token_lengths)} tokens, max_new_tokens={adjusted_max_tokens}")

        # Build generation kwargs with resolved parameters
        # Only include non-None parameters
        gen_kwargs = {
            **model_inputs,
            'temperature': params['temperature'],
            'max_new_tokens': adjusted_max_tokens,
            'do_sample': True if params['temperature'] > 0 else False,
            'num_beams': 1,
            'num_return_sequences': 1
        }

        if params['top_p'] is not None:
            gen_kwargs['top_p'] = params['top_p']
        if params['top_k'] is not None:
            gen_kwargs['top_k'] = params['top_k']
        if params['min_p'] is not None:
            gen_kwargs['min_p'] = params['min_p']

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**gen_kwargs)

        # Decode outputs
        generated_ids = [output_ids[len(input_ids[i]):] for i, output_ids in enumerate(generated_ids)]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return responses

    def _generate_vllm(
        self,
        input_texts: List[str],
        temperature: float,
        max_tokens: int,
        task_type: str
    ) -> List[str]:
        """Generate using vLLM backend."""
        from vllm import SamplingParams

        # Get resolved sampling parameters (user > model defaults > general defaults)
        params = self._get_sampling_params(temperature, max_tokens, task_type)

        # Build SamplingParams with resolved values
        # Only include non-None parameters
        sampling_kwargs = {
            'temperature': params['temperature'],
            'max_tokens': params['max_tokens']
        }

        if params['top_p'] is not None:
            sampling_kwargs['top_p'] = params['top_p']
        if params['top_k'] is not None:
            sampling_kwargs['top_k'] = params['top_k']
        if params['min_p'] is not None:
            sampling_kwargs['min_p'] = params['min_p']

        sampling_params = SamplingParams(**sampling_kwargs)

        # Generate
        outputs = self.model.generate(input_texts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        return responses


# ============================================================================
# CLI for standalone testing
# ============================================================================

def add_params():
    """Argument parser for standalone testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Batched LLM Inference Engine - Standalone Testing"
    )

    # Model configuration
    parser.add_argument('--model_name', type=str, default="gpt-4",
                        help='Model name or path (e.g., "gpt-4", "meta-llama/Llama-2-7b")')
    parser.add_argument('--backend', type=str, default="gpt", choices=["hf", "vllm", "gpt"],
                        help='Inference backend: hf (HuggingFace), vllm, or gpt (OpenAI API)')
    parser.add_argument('--cache_dir', type=str, default="/data1/ss164/cache/",
                        help='Cache directory for model downloads (local models only)')

    # Generation parameters
    parser.add_argument('--temp', type=float, default=0.7,
                        help='Temperature for generation (0.0-1.0, higher=more diverse)')
    parser.add_argument('--max_tokens', type=int, default=2048,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--bsz', type=int, default=8,
                        help='Batch size for generation')

    # Hardware configuration
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs for vLLM tensor parallelism')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')

    # Model-specific flags
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Enable thinking/reasoning mode (Qwen3, GPT-OSS, CWM)')
    parser.add_argument('--gpt_reasoning', type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help='GPT-OSS reasoning level')

    # vLLM-specific flags
    parser.add_argument('--vllm_multiproc_method', type=str, default="spawn",
                        choices=["spawn", "fork"],
                        help='vLLM multiprocessing method (default: spawn, fixes CUDA errors)')
    parser.add_argument('--vllm_max_num_seqs', type=int, default=None,
                        help='Maximum number of sequences to process in a batch (vLLM only, default: 256)')
    parser.add_argument('--vllm_max_model_len', type=int, default=None,
                        help='Maximum sequence length in tokens (vLLM only, default: model context length)')
    parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=None,
                        help='Fraction of GPU memory to use (vLLM only, default: 0.9, range: 0.0-1.0)')

    # Testing
    parser.add_argument('--test', action='store_true',
                        help='Run with test conversations')

    return parser.parse_args()


if __name__ == '__main__':
    args = add_params()

    print("=" * 80)
    print("BATCHED LLM INFERENCE ENGINE - STANDALONE TEST")
    print("=" * 80)

    # Initialize inference engine
    inference = BatchedLLMInference(
        model_name=args.model_name,
        backend=args.backend,
        temperature=args.temp,
        max_tokens=args.max_tokens,
        batch_size=args.bsz,
        cache_dir=args.cache_dir,
        num_gpus=args.num_gpus,
        no_cuda=args.no_cuda,
        enable_thinking=args.enable_thinking,
        gpt_reasoning=args.gpt_reasoning,
        vllm_multiproc_method=args.vllm_multiproc_method,
        vllm_max_num_seqs=args.vllm_max_num_seqs,
        vllm_max_model_len=args.vllm_max_model_len,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization
    )

    # Test with sample conversations
    if args.test:
        print("\n" + "=" * 80)
        print("RUNNING TEST GENERATION")
        print("=" * 80)

        test_conversations = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2 + 2?"}
            ],
            [
                {"role": "user", "content": "Explain gravity in one sentence."}
            ],
            [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is the Pythagorean theorem?"}
            ]
        ]

        print(f"\nGenerating responses for {len(test_conversations)} test conversations...")
        responses = inference.generate(test_conversations)

        print("\n" + "-" * 80)
        print("RESULTS:")
        print("-" * 80)
        for i, (conv, resp) in enumerate(zip(test_conversations, responses)):
            print(f"\nConversation {i+1}:")
            print(f"  User: {conv[-1]['content']}")
            print(f"  Assistant: {resp}")

        print("\n" + "=" * 80)
        print("TEST COMPLETE")
        print("=" * 80)
    else:
        print("\nInference engine initialized successfully!")
        print("Use --test flag to run with sample conversations")
        print("Or import this module and use the BatchedLLMInference class in your code")
