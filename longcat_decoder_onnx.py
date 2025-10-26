"""
ONNX wrappers for LongCat Audio Codec.

This module provides drop-in replacements for the PyTorch LongCat models
using ONNX Runtime for faster inference.
"""

import os
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Tuple, Optional
from huggingface_hub import hf_hub_download

class LongCatDecoderONNX:
    """
    ONNX/TensorRT implementation of LongCat decoder for faster audio synthesis.
    
    This class provides a drop-in replacement for the PyTorch decoder with
    significant speed improvements.
    """
    
    def __init__(
        self,
        model_path: str = None,
        use_tensorrt: bool = False,
        device_id: int = 0,
        batch_size: int = 1,
        output_rate: int = 24000
    ):
        """
        Initialize ONNX LongCat decoder.
        
        Args:
            model_path: Path to ONNX decoder model
            use_tensorrt: Whether to use TensorRT for acceleration
            device_id: CUDA device ID to use
            batch_size: Batch size for TensorRT optimization
            output_rate: Output sample rate (16000 or 24000)
        """
        self.use_tensorrt = use_tensorrt
        self.device_id = device_id
        self.batch_size = batch_size
        self.output_rate = output_rate
        self.decoder_path = model_path
        
        if model_path is None:
            raise ValueError("Please provide model_path")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX decoder not found at {model_path}")
        
        print(f"Loading ONNX LongCat decoder from: {model_path}")
        
        # Check CUDA availability
        cuda_available = ort.get_device() == "GPU"
        
        # Setup ONNX Runtime providers
        if cuda_available and device_id < torch.cuda.device_count():
            if use_tensorrt:
                # TensorRT configuration with dynamic shape profiles
                # Code length can vary (16.6 Hz frame rate)
                min_code_length = 16    # ~1 second
                max_code_length = 500   # ~30 seconds
                opt_code_length = 83    # ~5 seconds optimal
                
                trt_options = {
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': f'./trt_cache_longcat_decoder_{output_rate}',
                    'trt_fp16_enable': True,
                    'trt_max_workspace_size': (1 << 30) * 8,  # 8GB
                    'device_id': device_id,
                    'trt_profile_min_shapes': f"semantic_codes:{batch_size}x{min_code_length},acoustic_codes:{batch_size}x3x{min_code_length}",
                    'trt_profile_max_shapes': f"semantic_codes:{batch_size}x{max_code_length},acoustic_codes:{batch_size}x3x{max_code_length}",
                    'trt_profile_opt_shapes': f"semantic_codes:{batch_size}x{opt_code_length},acoustic_codes:{batch_size}x3x{opt_code_length}",
                }
                providers = [
                    ('TensorrtExecutionProvider', trt_options),
                    ('CUDAExecutionProvider', {'device_id': device_id}),
                    'CPUExecutionProvider'
                ]
                print(f"Using ONNX Runtime with TensorRT for decoder on device {device_id}")
            else:
                providers = [
                    ('CUDAExecutionProvider', {'device_id': device_id}),
                    'CPUExecutionProvider'
                ]
                print(f"Using ONNX Runtime with CUDA for decoder on device {device_id}")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU provider for decoder")
        
        # Create ONNX session with fallback handling
        session_created = False
        
        if use_tensorrt and cuda_available:
            try:
                self.session = ort.InferenceSession(model_path, providers=providers)
                print(f"ONNX session created with providers: {self.session.get_providers()}")
                session_created = True
            except Exception as e:
                print(f"TensorRT provider failed: {e}")
                print("Falling back to CUDA provider...")
        
        if not session_created and cuda_available:
            try:
                cuda_providers = [
                    ('CUDAExecutionProvider', {'device_id': device_id}),
                    'CPUExecutionProvider'
                ]
                self.session = ort.InferenceSession(model_path, providers=cuda_providers)
                print(f"ONNX session created with providers: {self.session.get_providers()}")
                session_created = True
            except Exception as e:
                print(f"CUDA provider failed: {e}")
                print("Falling back to CPU provider...")
        
        if not session_created:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            print("ONNX session created with CPU provider")
        
        # Warmup if using TensorRT
        if use_tensorrt and session_created and 'TensorrtExecutionProvider' in self.session.get_providers():
            self._warmup()
    
    def _warmup(self):
        """Warmup the model with dummy input"""
        print("Warming up ONNX decoder model...")
        try:
            # Create dummy codes (~3 seconds)
            code_length = 50
            dummy_semantic = np.zeros((self.batch_size, code_length), dtype=np.int64)
            dummy_acoustic = np.zeros((self.batch_size, 3, code_length), dtype=np.int64)
            _ = self.session.run(None, {
                'semantic_codes': dummy_semantic,
                'acoustic_codes': dummy_acoustic
            })
            print("ONNX decoder warmup successful")
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    def decode(
        self,
        semantic_codes: np.ndarray,
        acoustic_codes: np.ndarray
    ) -> np.ndarray:
        """
        Decode LongCat codes to audio waveform.
        
        Args:
            semantic_codes: Semantic token indices [B, T_codes]
            acoustic_codes: Acoustic token indices [B, N_q, T_codes]
            
        Returns:
            Audio waveform [B, 1, T_audio]
        """
        # Convert to numpy if needed
        if isinstance(semantic_codes, torch.Tensor):
            semantic_codes = semantic_codes.cpu().numpy().astype(np.int64)
        if isinstance(acoustic_codes, torch.Tensor):
            acoustic_codes = acoustic_codes.cpu().numpy().astype(np.int64)
        
        # Run inference
        try:
            outputs = self.session.run(None, {
                'semantic_codes': semantic_codes,
                'acoustic_codes': acoustic_codes
            })
            audio_output = outputs[0]
            return audio_output
        except Exception as e:
            print(f"Decoder inference error: {e}")
            raise
    
    def __call__(
        self,
        semantic_codes: np.ndarray,
        acoustic_codes: np.ndarray
    ) -> np.ndarray:
        """Alias for decode()"""
        return self.decode(semantic_codes, acoustic_codes)


def create_longcat_decoder(
    decoder_path: str,
    use_tensorrt: bool = False,
    device_id: int = 0,
    batch_size: int = 1
) -> LongCatDecoderONNX:
    """
    Convenience function to create LongCat ONNX decoder.
    
    Args:
        decoder_path: Path to ONNX decoder
        use_tensorrt: Whether to use TensorRT
        device_id: CUDA device ID
        batch_size: Batch size for optimization
        
    Returns:
        LongCat decoder instance
    """
    # Determine output rate from decoder filename
    output_rate = 24000
    if '16k' in os.path.basename(decoder_path):
        output_rate = 16000
    
    decoder = LongCatDecoderONNX(
        model_path=decoder_path,
        use_tensorrt=use_tensorrt,
        device_id=device_id,
        batch_size=batch_size,
        output_rate=output_rate
    )
    
    return decoder

