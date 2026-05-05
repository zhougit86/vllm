# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.load import LoadConfig
from vllm.config.lora import LoRAConfig
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as MRV2GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker

MODEL_PATH = "Qwen/Qwen3-0.6B"
NUM_LORAS = 4

DEVICE_TYPE = current_platform.device_type

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@patch.dict(os.environ, {"RANK": "0", "VLLM_USE_V1": "1"})
def test_mrv2_lora_warmup_activates_dummy_loras():
    model_config = ModelConfig(
        MODEL_PATH,
        seed=0,
        dtype="float16",
        max_model_len=127,
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(
            download_dir=None,
            load_format="dummy", # Use dummy weights to make it fast
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
            runner_type="generate",
            max_num_batched_tokens=32,
            max_num_seqs=32,
            max_num_partial_prefills=32,
        ),
        device_config=DeviceConfig(DEVICE_TYPE),
        cache_config=CacheConfig(
            block_size=16,
            cache_dtype="auto",
        ),
        lora_config=LoRAConfig(
            max_lora_rank=8, max_cpu_loras=NUM_LORAS, max_loras=NUM_LORAS
        ),
    )

    # Note: Using Worker to properly initialize the device and distributed environment
    # The worker initializes the v1 GPUModelRunner
    worker = Worker(
        vllm_config=vllm_config,
        local_rank=0,
        rank=0,
        distributed_init_method=f"file://{tempfile.mkstemp()[1]}",
    )
    
    with set_current_vllm_config(vllm_config):
        worker.init_device()
        worker.load_model()
    
    runner = worker.model_runner
    # The worker might wrap the runner or import the MRV1 runner by default.
    # To be absolutely sure we test MRV2, we can check if it has the V2 specific attributes
    # or just trust the worker's initialization if it's using the v1/worker/gpu_worker.py
    assert hasattr(runner, 'cudagraph_manager'), "Ensure this runner supports MRV2 architecture"
    
    # 1. Test profile_run (which calls _dummy_run)
    with patch.object(runner, '_set_active_loras', wraps=runner._set_active_loras) as mock_set_active:
        worker.profile_num_available_blocks(16, 16, 16)
        
        # Verify _set_active_loras was called during memory profiling (_dummy_run)
        assert mock_set_active.called, "_set_active_loras was not called during profile_run"
        
        # Verify it used dummy LoRAs (e.g., warmup_1)
        args, kwargs = mock_set_active.call_args
        # In _set_active_loras(*lora_inputs), args are (sample_lora_mapping, token_lora_mapping, lora_requests, mapping_type)
        lora_requests = args[2]
        assert len(lora_requests) > 0, "No dummy LoRAs were activated"
        assert any("warmup_" in lr.lora_name for lr in lora_requests), "Dummy LoRA name 'warmup_' not found"
        
    # 2. Test capture_model (CUDA graph capture)
    with patch.object(runner, '_set_active_loras', wraps=runner._set_active_loras) as mock_set_active:
        runner.capture_model()
        
        assert mock_set_active.called, "_set_active_loras was not called during capture_model"
        
        args, kwargs = mock_set_active.call_args
        lora_requests = args[2]
        assert len(lora_requests) > 0, "No dummy LoRAs were activated during capture"
        assert any("warmup_" in lr.lora_name for lr in lora_requests), "Dummy LoRA name 'warmup_' not found during capture"

if __name__ == "__main__":
    test_mrv2_lora_warmup_activates_dummy_loras()
