"""
LongCat encoder implementation using PyTorch.

This module provides encoding functionality for dataset creation:
- PyTorch encoder: Handles complex semantic tokenizer with full compatibility
"""

import os
import torch
from typing import Tuple
from networks.semantic_codec.model_loader import load_encoder


class LongCatEncoder:
    """
    LongCat encoder using PyTorch.
    
    This provides full compatibility for encoding audio to tokens,
    which is used primarily for dataset creation.
    """
    
    def __init__(
        self,
        encoder_config: str,
        device_id: int = 0,
        n_acoustic_codebooks: int = 3
    ):
        """
        Initialize LongCat encoder.
        
        Args:
            encoder_config: Path to encoder YAML config
            device_id: CUDA device ID
            n_acoustic_codebooks: Number of acoustic codebooks to use (1-3)
        """
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.n_acoustic_codebooks = n_acoustic_codebooks
        
        # Load PyTorch encoder
        print("Loading LongCat encoder...")
        self.encoder = load_encoder(encoder_config, self.device)
        self.encoder.eval()
        
        print(f"✓ LongCat encoder ready!")
        print(f"  - Device: {self.device}")
        print(f"  - Acoustic codebooks: {n_acoustic_codebooks}")
    
    def encode(self, audio: torch.Tensor, sample_rate: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to LongCat tokens.
        
        Args:
            audio: Audio tensor [B, 1, T] or [1, T]
            sample_rate: Input sample rate (default: encoder's input_sample_rate)
            
        Returns:
            Tuple of (semantic_codes, acoustic_codes)
            - semantic_codes: [B, T_codes] 
            - acoustic_codes: [B, N_q, T_codes]
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        
        audio = audio.to(self.device)
        
        with torch.no_grad():
            semantic_codes, acoustic_codes = self.encoder(
                audio,
                sample_rate,
                n_acoustic_codebooks=self.n_acoustic_codebooks
            )
        
        return semantic_codes, acoustic_codes
    
    def __call__(
        self,
        audio: torch.Tensor,
        sample_rate: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio to LongCat tokens.
        
        Args:
            audio: Input audio tensor
            sample_rate: Input sample rate
            
        Returns:
            Tuple of (semantic_codes, acoustic_codes)
        """
        return self.encode(audio, sample_rate)


# Convenience function for easy setup
def create_longcat_encoder(
    encoder_config: str = 'configs/LongCatAudioCodec_encoder.yaml',
    device_id: int = 0,
    n_acoustic_codebooks: int = 3
) -> LongCatEncoder:
    """
    Create LongCat encoder with sensible defaults.
    
    Args:
        encoder_config: Path to encoder config
        device_id: CUDA device ID
        n_acoustic_codebooks: Number of acoustic codebooks (1-3)
        
    Returns:
        Configured LongCat encoder
    """
    return LongCatEncoder(
        encoder_config=encoder_config,
        device_id=device_id,
        n_acoustic_codebooks=n_acoustic_codebooks
    )

