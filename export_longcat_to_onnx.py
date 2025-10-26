#!/usr/bin/env python3
"""
Export LongCat Audio Codec models to ONNX format for faster inference.

This script exports both the encoder and decoder models to ONNX, with support
for dynamic shapes and TensorRT optimization.
"""

import os
import argparse
import torch
import numpy as np
import torchaudio
from pathlib import Path

from networks.semantic_codec.model_loader import load_encoder, load_decoder


def export_encoder_to_onnx(
    encoder,
    output_path: str,
    n_acoustic_codebooks: int = 3,
    opset_version: int = 17,
    simplify: bool = True
):
    """
    Export LongCat encoder to ONNX format.
    
    Args:
        encoder: The loaded LongCat encoder model
        output_path: Path to save the ONNX model
        n_acoustic_codebooks: Number of acoustic codebooks to use
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model (requires onnx-simplifier)
    """
    print("\n" + "="*70)
    print("Exporting Encoder to ONNX")
    print("="*70)
    
    encoder.eval()
    device = next(encoder.parameters()).device
    
    # Create dummy input
    # Use different lengths to test dynamic shapes
    sample_rate = encoder.input_sample_rate
    
    # Test with 3 seconds of audio
    audio_length_samples = sample_rate * 3
    dummy_audio = torch.randn(1, 1, audio_length_samples).to(device)
    
    print(f"  - Input shape: {dummy_audio.shape}")
    print(f"  - Sample rate: {sample_rate} Hz")
    print(f"  - Acoustic codebooks: {n_acoustic_codebooks}")
    
    # Test the model first
    with torch.no_grad():
        test_output = encoder(dummy_audio, sample_rate, n_acoustic_codebooks)
        semantic_codes, acoustic_codes = test_output
        print(f"  - Semantic codes shape: {semantic_codes.shape}")
        if acoustic_codes is not None:
            print(f"  - Acoustic codes shape: {acoustic_codes.shape}")
    
    # Create a wrapper module for cleaner ONNX export
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder, n_acoustic_codebooks):
            super().__init__()
            self.encoder = encoder
            self.n_acoustic_codebooks = n_acoustic_codebooks
            
        def forward(self, audio_data):
            """
            Forward pass that returns both semantic and acoustic codes.
            
            Args:
                audio_data: Input audio tensor [B, 1, T]
                
            Returns:
                semantic_codes: [B, T_codes]
                acoustic_codes: [B, N_q, T_codes]
            """
            # Preprocess
            audio_data = self.encoder.preprocess(audio_data, self.encoder.input_sample_rate)
            
            # Get semantic codes
            semantic_codes = self.encoder.get_semantic_codes(audio_data)
            
            # Get acoustic codes
            acoustic_codes = self.encoder.get_acoustic_codes(audio_data, self.n_acoustic_codebooks)
            
            return semantic_codes, acoustic_codes
    
    wrapped_encoder = EncoderWrapper(encoder, n_acoustic_codebooks).to(device)
    wrapped_encoder.eval()
    
    # Define dynamic axes for variable-length audio
    dynamic_axes = {
        'audio_data': {0: 'batch_size', 2: 'audio_length'},
        'semantic_codes': {0: 'batch_size', 1: 'code_length'},
        'acoustic_codes': {0: 'batch_size', 2: 'code_length'}
    }
    
    # Export to ONNX
    print(f"\n  - Exporting to: {output_path}")
    torch.onnx.export(
        wrapped_encoder,
        dummy_audio,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['audio_data'],
        output_names=['semantic_codes', 'acoustic_codes'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"  ✓ Encoder exported successfully!")
    
    # Simplify the model if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\n  - Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(simplified_model, simplified_path)
                print(f"  ✓ Simplified model saved to: {simplified_path}")
            else:
                print("  ⚠ Simplification check failed, using original model")
        except ImportError:
            print("  ⚠ onnx-simplifier not installed, skipping simplification")
            print("    Install with: pip install onnx-simplifier")
    
    return output_path


def export_decoder_to_onnx(
    decoder,
    output_path: str,
    n_acoustic_codebooks: int = 3,
    opset_version: int = 17,
    simplify: bool = True
):
    """
    Export LongCat decoder to ONNX format.
    
    Args:
        decoder: The loaded LongCat decoder model
        output_path: Path to save the ONNX model
        n_acoustic_codebooks: Number of acoustic codebooks
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX model
    """
    print("\n" + "="*70)
    print(f"Exporting Decoder ({decoder.decoder_type}) to ONNX")
    print("="*70)
    
    decoder.eval()
    device = next(decoder.parameters()).device
    
    # Create dummy inputs matching typical token lengths
    # For ~3 seconds at 16.6Hz frame rate: ~50 frames
    code_length = 50
    batch_size = 1
    
    dummy_semantic_codes = torch.randint(0, 8192, (batch_size, code_length)).to(device)
    dummy_acoustic_codes = torch.randint(0, 1024, (batch_size, n_acoustic_codebooks, code_length)).to(device)
    
    print(f"  - Semantic codes shape: {dummy_semantic_codes.shape}")
    print(f"  - Acoustic codes shape: {dummy_acoustic_codes.shape}")
    print(f"  - Output sample rate: {decoder.output_rate} Hz")
    
    # Test the model first
    with torch.no_grad():
        test_output = decoder(dummy_semantic_codes, dummy_acoustic_codes)
        print(f"  - Output audio shape: {test_output.shape}")
    
    # Define dynamic axes
    dynamic_axes = {
        'semantic_codes': {0: 'batch_size', 1: 'code_length'},
        'acoustic_codes': {0: 'batch_size', 2: 'code_length'},
        'audio_output': {0: 'batch_size', 2: 'audio_length'}
    }
    
    # Export to ONNX
    print(f"\n  - Exporting to: {output_path}")
    torch.onnx.export(
        decoder,
        (dummy_semantic_codes, dummy_acoustic_codes),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['semantic_codes', 'acoustic_codes'],
        output_names=['audio_output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"  ✓ Decoder exported successfully!")
    
    # Simplify the model if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\n  - Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(simplified_model, simplified_path)
                print(f"  ✓ Simplified model saved to: {simplified_path}")
            else:
                print("  ⚠ Simplification check failed, using original model")
        except ImportError:
            print("  ⚠ onnx-simplifier not installed, skipping simplification")
    
    return output_path


def validate_onnx_model(
    onnx_path: str,
    pytorch_model,
    is_encoder: bool,
    test_audio_path: str = None,
    n_acoustic_codebooks: int = 3
):
    """
    Validate that the ONNX model produces the same output as PyTorch.
    
    Args:
        onnx_path: Path to the ONNX model
        pytorch_model: The original PyTorch model
        is_encoder: Whether this is an encoder or decoder
        test_audio_path: Path to test audio file (for encoder validation)
        n_acoustic_codebooks: Number of acoustic codebooks
    """
    print("\n" + "="*70)
    print(f"Validating ONNX Model: {os.path.basename(onnx_path)}")
    print("="*70)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ⚠ onnxruntime not installed, skipping validation")
        print("    Install with: pip install onnxruntime-gpu")
        return False
    
    # Load ONNX model
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  - Loaded ONNX model with providers: {session.get_providers()}")
    
    device = next(pytorch_model.parameters()).device
    
    if is_encoder:
        # Test encoder
        if test_audio_path and os.path.exists(test_audio_path):
            print(f"  - Using test audio: {test_audio_path}")
            audio, sr = torchaudio.load(test_audio_path)
            audio = audio[:1, :pytorch_model.input_sample_rate * 3]  # First 3 seconds
        else:
            print(f"  - Using random audio")
            audio = torch.randn(1, 1, pytorch_model.input_sample_rate * 3)
        
        audio = audio.to(device)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pt_semantic, pt_acoustic = pytorch_model(audio, pytorch_model.input_sample_rate, n_acoustic_codebooks)
        
        # ONNX inference
        audio_np = audio.cpu().numpy().astype(np.float32)
        onnx_outputs = session.run(None, {'audio_data': audio_np})
        onnx_semantic = onnx_outputs[0]
        onnx_acoustic = onnx_outputs[1]
        
        # Compare outputs
        semantic_match = np.allclose(pt_semantic.cpu().numpy(), onnx_semantic, atol=1e-5)
        acoustic_match = np.allclose(pt_acoustic.cpu().numpy(), onnx_acoustic, atol=1e-5)
        
        print(f"\n  - Semantic codes match: {semantic_match}")
        print(f"  - Acoustic codes match: {acoustic_match}")
        
        if semantic_match and acoustic_match:
            print("\n  ✓ ONNX encoder validation PASSED!")
            return True
        else:
            print("\n  ✗ ONNX encoder validation FAILED!")
            max_diff_semantic = np.max(np.abs(pt_semantic.cpu().numpy() - onnx_semantic))
            max_diff_acoustic = np.max(np.abs(pt_acoustic.cpu().numpy() - onnx_acoustic))
            print(f"    Max semantic difference: {max_diff_semantic}")
            print(f"    Max acoustic difference: {max_diff_acoustic}")
            return False
    else:
        # Test decoder
        code_length = 50
        batch_size = 1
        
        semantic_codes = torch.randint(0, 8192, (batch_size, code_length)).to(device)
        acoustic_codes = torch.randint(0, 1024, (batch_size, n_acoustic_codebooks, code_length)).to(device)
        
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pt_output = pytorch_model(semantic_codes, acoustic_codes)
        
        # ONNX inference
        semantic_np = semantic_codes.cpu().numpy().astype(np.int64)
        acoustic_np = acoustic_codes.cpu().numpy().astype(np.int64)
        onnx_output = session.run(None, {
            'semantic_codes': semantic_np,
            'acoustic_codes': acoustic_np
        })[0]
        
        # Compare outputs
        match = np.allclose(pt_output.cpu().numpy(), onnx_output, atol=1e-4)
        
        print(f"\n  - Audio output match: {match}")
        
        if match:
            print("\n  ✓ ONNX decoder validation PASSED!")
            return True
        else:
            print("\n  ✗ ONNX decoder validation FAILED!")
            max_diff = np.max(np.abs(pt_output.cpu().numpy() - onnx_output))
            print(f"    Max difference: {max_diff}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Export LongCat Audio Codec to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--encoder_config',
        type=str,
        default='configs/LongCatAudioCodec_encoder.yaml',
        help='Path to encoder YAML config'
    )
    parser.add_argument(
        '--decoder16k_config',
        type=str,
        default='configs/LongCatAudioCodec_decoder_16k_4codebooks.yaml',
        help='Path to 16kHz decoder YAML config'
    )
    parser.add_argument(
        '--decoder24k_config',
        type=str,
        default='configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml',
        help='Path to 24kHz decoder YAML config'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='onnx_models',
        help='Directory to save ONNX models'
    )
    parser.add_argument(
        '--n_acoustic_codebooks',
        type=int,
        default=3,
        help='Number of acoustic codebooks to use'
    )
    parser.add_argument(
        '--opset_version',
        type=int,
        default=17,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX models (requires onnx-simplifier)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate ONNX models against PyTorch'
    )
    parser.add_argument(
        '--test_audio',
        type=str,
        default='demos/org/common.wav',
        help='Path to test audio for validation'
    )
    parser.add_argument(
        '--skip_encoder',
        action='store_true',
        help='Skip encoder export'
    )
    parser.add_argument(
        '--skip_decoder',
        action='store_true',
        help='Skip decoder export'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    exported_models = []
    
    # Export encoder
    if not args.skip_encoder:
        print("\n" + "="*70)
        print("STEP 1: Export Encoder")
        print("="*70)
        
        encoder = load_encoder(args.encoder_config, device)
        encoder_path = os.path.join(args.output_dir, 'longcat_encoder.onnx')
        
        export_encoder_to_onnx(
            encoder,
            encoder_path,
            n_acoustic_codebooks=args.n_acoustic_codebooks,
            opset_version=args.opset_version,
            simplify=args.simplify
        )
        
        exported_models.append(('encoder', encoder_path, encoder, True))
        
        # Check if simplified version exists
        simplified_path = encoder_path.replace('.onnx', '_simplified.onnx')
        if os.path.exists(simplified_path):
            exported_models.append(('encoder_simplified', simplified_path, encoder, True))
    
    # Export decoders
    if not args.skip_decoder:
        print("\n" + "="*70)
        print("STEP 2: Export Decoders")
        print("="*70)
        
        # 16kHz decoder
        print("\n--- 16kHz Decoder ---")
        decoder16k = load_decoder(args.decoder16k_config, device)
        decoder16k_path = os.path.join(args.output_dir, 'longcat_decoder_16k.onnx')
        
        export_decoder_to_onnx(
            decoder16k,
            decoder16k_path,
            n_acoustic_codebooks=args.n_acoustic_codebooks,
            opset_version=args.opset_version,
            simplify=args.simplify
        )
        
        exported_models.append(('decoder_16k', decoder16k_path, decoder16k, False))
        
        # Check if simplified version exists
        simplified_path = decoder16k_path.replace('.onnx', '_simplified.onnx')
        if os.path.exists(simplified_path):
            exported_models.append(('decoder_16k_simplified', simplified_path, decoder16k, False))
        
        # 24kHz decoder
        print("\n--- 24kHz Decoder ---")
        decoder24k = load_decoder(args.decoder24k_config, device)
        decoder24k_path = os.path.join(args.output_dir, 'longcat_decoder_24k.onnx')
        
        export_decoder_to_onnx(
            decoder24k,
            decoder24k_path,
            n_acoustic_codebooks=args.n_acoustic_codebooks,
            opset_version=args.opset_version,
            simplify=args.simplify
        )
        
        exported_models.append(('decoder_24k', decoder24k_path, decoder24k, False))
        
        # Check if simplified version exists
        simplified_path = decoder24k_path.replace('.onnx', '_simplified.onnx')
        if os.path.exists(simplified_path):
            exported_models.append(('decoder_24k_simplified', simplified_path, decoder24k, False))
    
    # Validate models
    if args.validate:
        print("\n" + "="*70)
        print("STEP 3: Validation")
        print("="*70)
        
        validation_results = []
        for name, path, model, is_encoder in exported_models:
            result = validate_onnx_model(
                path,
                model,
                is_encoder,
                args.test_audio,
                args.n_acoustic_codebooks
            )
            validation_results.append((name, result))
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        for name, result in validation_results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"  {name}: {status}")
    
    # Final summary
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nExported models saved to: {args.output_dir}/")
    for name, path, _, _ in exported_models:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {name}: {os.path.basename(path)} ({size_mb:.1f} MB)")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Test the ONNX models with the test script:")
    print(f"   python test_longcat_onnx.py --onnx_dir {args.output_dir}")
    print("\n2. Use in your pipeline:")
    print("   - For dataset creation: Update create_hindi_dataset.py")
    print("   - For TTS inference: Update tts.py")
    print("\n3. For faster inference, consider:")
    print("   - Using simplified models (*_simplified.onnx)")
    print("   - Enabling TensorRT (see integration scripts)")
    print("="*70)


if __name__ == '__main__':
    main()

