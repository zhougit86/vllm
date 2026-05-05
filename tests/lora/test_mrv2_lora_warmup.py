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

    with set_current_vllm_config(vllm_config):
        # Instead of going through Worker, which might have complex routing
        # for MRV1 vs MRV2, we initialize the MRV2 runner directly.
        from vllm.distributed import get_tensor_model_parallel_world_size
        import vllm.distributed.parallel_state as parallel_state
        
        # We need a minimal distributed env
        if not torch.distributed.is_initialized():
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"file://{tempfile.mkstemp()[1]}"
            )
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

        runner = MRV2GPUModelRunner(vllm_config, torch.device("cuda:0"))
        runner.load_model()
        
        # 1. Test profile_run (which calls _dummy_run)
        # Note: We need to initialize kv cache after load_model and before profile_run
        # because profile_run needs to use block_tables
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.platforms import current_platform
        current_platform.update_block_size_for_backend(vllm_config)
        kv_cache_spec = runner.get_kv_cache_spec()
        available_memory = 1 * 1024**3 # 1 GB
        kv_cache_config = get_kv_cache_configs(
            vllm_config, [kv_cache_spec], [available_memory]
        )[0]
        runner.initialize_kv_cache(kv_cache_config)

    assert hasattr(runner, 'cudagraph_manager'), "Ensure this runner supports MRV2 architecture"

    with patch.object(runner, '_set_active_loras', wraps=runner._set_active_loras) as mock_set_active:
        runner.profile_run()
        
        # Print all calls to _set_active_loras to debug
        print(f"\n[DEBUG] _set_active_loras called {mock_set_active.call_count} times during profile_run")
        for i, call in enumerate(mock_set_active.call_args_list):
            print(f"[DEBUG] Call {i} args: {call.args}")
            print(f"[DEBUG] Call {i} kwargs: {call.kwargs}")
        
        # Verify _set_active_loras was called during memory profiling (_dummy_run)
        assert mock_set_active.called, "_set_active_loras was not called during profile_run"
        
        # Verify it used dummy LoRAs (e.g., warmup_1) in ANY of the calls
        found_dummy_loras = False
        for call in mock_set_active.call_args_list:
            args, kwargs = call
            lora_requests = None
            for arg in args:
                if isinstance(arg, set):
                    lora_requests = arg
                    break
            if lora_requests is None:
                lora_requests = kwargs.get('lora_requests', set())
                
            if len(lora_requests) > 0 and any("warmup_" in lr.lora_name for lr in lora_requests):
                found_dummy_loras = True
                break
                
        assert found_dummy_loras, "No dummy LoRAs were activated in any of the calls"
        
    # 2. Test capture_model (CUDA graph capture)
    # Ensure capture sizes are populated so needs_capture() is true
    if not vllm_config.compilation_config.cudagraph_capture_sizes:
        vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32]
        # Re-evaluate needs_capture since we modified the config
        runner.cudagraph_manager._init_candidates()
        
    print(f"\n[DEBUG] cudagraph_manager needs_capture: {runner.cudagraph_manager.needs_capture()}")

    with patch.object(runner, '_set_active_loras', wraps=runner._set_active_loras) as mock_set_active:
        runner.capture_model()
        
        print(f"\n[DEBUG] _set_active_loras called {mock_set_active.call_count} times during capture_model")
        for i, call in enumerate(mock_set_active.call_args_list):
            print(f"[DEBUG] Call {i} args: {call.args}")
            print(f"[DEBUG] Call {i} kwargs: {call.kwargs}")

        assert mock_set_active.called, "_set_active_loras was not called during capture_model"
        
        found_dummy_loras = False
        for call in mock_set_active.call_args_list:
            args, kwargs = call
            lora_requests = None
            for arg in args:
                if isinstance(arg, set):
                    lora_requests = arg
                    break
            if lora_requests is None:
                lora_requests = kwargs.get('lora_requests', set())
                
            if len(lora_requests) > 0 and any("warmup_" in lr.lora_name for lr in lora_requests):
                found_dummy_loras = True
                break
                
        assert found_dummy_loras, "No dummy LoRAs were activated during capture"

if __name__ == "__main__":
    test_mrv2_lora_warmup_activates_dummy_loras()
