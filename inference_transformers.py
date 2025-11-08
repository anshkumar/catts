"""
LongCat TTS Inference using Transformers
"""

import os
import argparse
import torch
import torchaudio
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from longcat_decoder_onnx import LongCatDecoderONNX


# Token constants
TOKENISER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3    # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4     # 128260
START_OF_AI = TOKENISER_LENGTH + 5       # 128261
END_OF_AI = TOKENISER_LENGTH + 6         # 128262
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 128266

# LongCat configuration
SEMANTIC_CODEBOOK_SIZE = 8192
ACOUSTIC_CODEBOOK_SIZE = 8100
N_ACOUSTIC_CODEBOOKS = 3


def detect_repetition_and_truncate(tokens: List[int], window_size: int = 4, threshold: int = 10) -> List[int]:
    """
    Detect excessive repetition and truncate.
    
    Args:
        tokens: List of token IDs  
        window_size: Size of repeating pattern
        threshold: Max repetitions before truncating
        
    Returns:
        Truncated list if repetition detected
    """
    if len(tokens) < window_size * 2:
        return tokens
    
    # Look for repeating patterns at the end
    for start in range(len(tokens) - window_size * threshold, max(0, len(tokens) - window_size * 50), -window_size):
        pattern = tokens[start:start + window_size]
        
        # Count consecutive repetitions
        count = 0
        pos = start
        while pos + window_size <= len(tokens) and tokens[pos:pos + window_size] == pattern:
            count += 1
            pos += window_size
        
        if count >= threshold:
            print(f"⚠️  Detected {count} repetitions of pattern, truncating...")
            return tokens[:start + window_size]
    
    return tokens


def extract_audio_tokens(token_ids: List[int]) -> List[int]:
    """
    Extract audio tokens from generated token sequence.
    
    The model may generate either START_OF_SPEECH or START_OF_AI before audio tokens.
    
    Args:
        token_ids: List of generated token IDs
        
    Returns:
        List of audio token IDs (interleaved semantic + acoustic)
    """
    try:
        # Try to find START_OF_SPEECH first
        start_idx = None
        start_token_name = None
        
        if START_OF_SPEECH in token_ids:
            start_idx = token_ids.index(START_OF_SPEECH)
            start_token_name = "START_OF_SPEECH"
        elif START_OF_AI in token_ids:
            # Model sometimes generates START_OF_AI instead
            start_idx = token_ids.index(START_OF_AI)
            start_token_name = "START_OF_AI"
        else:
            print(f"Error: No start token found (neither START_OF_SPEECH nor START_OF_AI)")
            return []
        
        print(f"Found {start_token_name} at position {start_idx}")
        
        # Find end of speech or end of AI
        end_idx = None
        end_token_name = None
        
        # Look for end markers after the start position
        if END_OF_SPEECH in token_ids[start_idx:]:
            end_idx = start_idx + token_ids[start_idx:].index(END_OF_SPEECH)
            end_token_name = "END_OF_SPEECH"
        elif END_OF_AI in token_ids[start_idx:]:
            end_idx = start_idx + token_ids[start_idx:].index(END_OF_AI)
            end_token_name = "END_OF_AI"
        else:
            # No end token found, take everything after start
            end_idx = len(token_ids)
            end_token_name = "end of sequence"
        
        print(f"Found {end_token_name} at position {end_idx}")
        
        # Extract audio tokens (between start and end)
        audio_tokens = token_ids[start_idx + 1:end_idx]
        
        return audio_tokens
    except (ValueError, IndexError) as e:
        print(f"Error extracting audio tokens: {e}")
        return []


def deinterleave_tokens(audio_tokens: List[int]) -> tuple:
    """
    Convert interleaved token IDs to semantic and acoustic codes.
    
    LongCat format: [sem, ac0, ac1, ac2] per frame
    
    Args:
        audio_tokens: List of audio token IDs
        
    Returns:
        semantic_codes: np.ndarray [T_codes]
        acoustic_codes: np.ndarray [N_q, T_codes]
    """
    tokens_per_frame = 1 + N_ACOUSTIC_CODEBOOKS  # 1 semantic + 3 acoustic
    n_frames = len(audio_tokens) // tokens_per_frame
    
    if len(audio_tokens) % tokens_per_frame != 0:
        print(f"Warning: Audio tokens ({len(audio_tokens)}) not divisible by {tokens_per_frame}")
        # Truncate to complete frames
        audio_tokens = audio_tokens[:n_frames * tokens_per_frame]
    
    if n_frames == 0:
        print("Warning: No complete frames found in audio tokens")
        return np.array([], dtype=np.int64), np.array([[], [], []], dtype=np.int64)
    
    semantic_codes = []
    acoustic_codes = [[] for _ in range(N_ACOUSTIC_CODEBOOKS)]
    
    for i in range(n_frames):
        frame_start = i * tokens_per_frame
        
        # Semantic token (first in frame)
        sem_token = audio_tokens[frame_start]
        sem_code = sem_token - AUDIO_TOKENS_START
        
        # Validate semantic code
        if sem_code < 0 or sem_code >= SEMANTIC_CODEBOOK_SIZE:
            print(f"Warning: Invalid semantic code {sem_code} at frame {i}")
            sem_code = max(0, min(sem_code, SEMANTIC_CODEBOOK_SIZE - 1))
        
        semantic_codes.append(sem_code)
        
        # Acoustic tokens (next N_ACOUSTIC_CODEBOOKS in frame)
        for q in range(N_ACOUSTIC_CODEBOOKS):
            ac_token = audio_tokens[frame_start + 1 + q]
            offset = AUDIO_TOKENS_START + SEMANTIC_CODEBOOK_SIZE + (q * ACOUSTIC_CODEBOOK_SIZE)
            ac_code = ac_token - offset
            
            # Validate acoustic code
            if ac_code < 0 or ac_code >= ACOUSTIC_CODEBOOK_SIZE:
                print(f"Warning: Invalid acoustic code {ac_code} at frame {i}, codebook {q}")
                ac_code = max(0, min(ac_code, ACOUSTIC_CODEBOOK_SIZE - 1))
            
            acoustic_codes[q].append(ac_code)
    
    # Convert to numpy arrays
    semantic_codes = np.array(semantic_codes, dtype=np.int64)
    acoustic_codes = np.array(acoustic_codes, dtype=np.int64)  # [N_q, T]
    
    return semantic_codes, acoustic_codes


class LongCatTTS:
    """LongCat TTS using Transformers"""
    
    def __init__(
        self,
        model_path: str,
        decoder_path: str,
        output_rate: int = 24000,
        use_tensorrt: bool = True,
        device_id: int = 0,
        device: str = "cuda"
    ):
        """
        Initialize LongCat TTS system.
        
        Args:
            model_path: Path to the LongCat language model (HuggingFace format)
            decoder_path: Path to ONNX decoder model
            output_rate: Output sample rate (16000 or 24000)
            use_tensorrt: Use TensorRT for decoder acceleration
            device_id: CUDA device ID
            device: Device for language model ("cuda" or "cpu")
        """
        print(f"Initializing LongCat TTS...")
        print(f"  Model: {model_path}")
        print(f"  Decoder: {decoder_path}")
        print(f"  Device: {device}")
        
        self.device = device
        self.output_rate = output_rate
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        
        # Load language model
        print("Loading language model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Update model config to include custom EOS tokens
        # The model needs to know that END_OF_SPEECH (128258) and END_OF_AI (128262) are stopping tokens
        if hasattr(self.model.config, 'eos_token_id'):
            original_eos = self.model.config.eos_token_id
            custom_eos_tokens = [END_OF_SPEECH, END_OF_AI]
            
            if isinstance(original_eos, int):
                self.model.config.eos_token_id = [original_eos] + custom_eos_tokens
            elif isinstance(original_eos, list):
                # Add custom tokens if not already present
                eos_set = set(original_eos)
                for token in custom_eos_tokens:
                    if token not in eos_set:
                        original_eos.append(token)
                self.model.config.eos_token_id = original_eos
            
            print(f"Updated EOS tokens: {self.model.config.eos_token_id}")
        
        # Load ONNX decoder
        print("Loading ONNX decoder...")
        self.decoder = LongCatDecoderONNX(
            model_path=decoder_path,
            use_tensorrt=use_tensorrt,
            device_id=device_id,
            batch_size=1,
            output_rate=output_rate
        )
        
        print("✓ LongCat TTS initialized successfully!\n")
    
    def format_prompt(self, text: str, voice: str = "Babla") -> List[int]:
        """
        Format the input text into the expected prompt format.
        
        Training format (from create_dataset_longcat.py):
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)  # Adds BOS (128000)
        text_ids.append(END_OF_TEXT)  # 128009
        [START_OF_HUMAN] + text_ids + [END_OF_HUMAN] + [START_OF_AI] + [START_OF_SPEECH] + audio + [END_OF_SPEECH] + [END_OF_AI]
        
        Inference should match training format up to START_OF_AI, then model generates:
        [START_OF_SPEECH] + audio_tokens + [END_OF_SPEECH] + [END_OF_AI]
        
        Args:
            text: Text to convert to speech
            voice: Voice ID (e.g., "Babla", "Pallavi", etc.)
            
        Returns:
            List of token IDs
        """
        # Tokenize the text with voice (add_special_tokens=True adds BOS token like in training)
        text_with_voice = f"{voice}: {text}"
        text_ids = self.tokenizer.encode(text_with_voice, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)  # Add END_OF_TEXT like in training
        
        # Construct prompt with actual token IDs matching training format
        # [START_OF_HUMAN] + [BOS + text_tokens + END_OF_TEXT] + [END_OF_HUMAN] + [START_OF_AI]
        prompt_ids = (
            [START_OF_HUMAN]  # 128259
            + text_ids        # [BOS(128000) + "voice: text" + END_OF_TEXT(128009)]
            + [END_OF_HUMAN]  # 128260
            + [START_OF_AI]   # 128261
        )
        
        return prompt_ids
    
    def generate(
        self,
        text: str,
        voice: str = "Babla",
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        debug: bool = False
    ) -> List[int]:
        """
        Generate audio tokens from text.
        
        Args:
            text: Text to convert to speech
            voice: Voice ID
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum number of tokens to generate
            repetition_penalty: Repetition penalty
            debug: Print debug information
            
        Returns:
            List of generated token IDs
        """
        # Format prompt (returns token IDs)
        prompt_ids = self.format_prompt(text, voice)
        print(f"Generating speech for: '{text}'")
        print(f"Voice: {voice}")
        
        # Convert to tensor
        input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
        
        print(f"Input tokens: {input_ids.shape[1]}")
        
        if debug:
            print(f"\nDEBUG - Prompt token IDs: {input_ids[0].tolist()}")
            print(f"DEBUG - First 10 tokens: {input_ids[0].tolist()[:10]}")
            print(f"DEBUG - Last 10 tokens: {input_ids[0].tolist()[-10:]}")
        
        # Generate
        # Use model's configured EOS tokens (which now includes END_OF_SPEECH)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 128001,
                # eos_token_id is automatically taken from model.config.eos_token_id
            )
        
        # Get generated tokens
        generated_ids = outputs[0].cpu().tolist()
        new_tokens = generated_ids[input_ids.shape[1]:]  # Only the newly generated tokens
        
        print(f"Total output tokens: {len(generated_ids)}")
        print(f"Newly generated tokens: {len(new_tokens)}")
        
        if debug:
            print(f"\nDEBUG - First 50 generated tokens: {new_tokens[:50]}")
            print(f"DEBUG - Last 50 generated tokens: {new_tokens[-50:]}")
            
            # Check for special tokens
            special_tokens_found = []
            if START_OF_SPEECH in new_tokens:
                special_tokens_found.append(f"START_OF_SPEECH (128257) at {new_tokens.index(START_OF_SPEECH)}")
            if END_OF_SPEECH in new_tokens:
                special_tokens_found.append(f"END_OF_SPEECH (128258) at {new_tokens.index(END_OF_SPEECH)}")
            if START_OF_AI in new_tokens:
                special_tokens_found.append(f"START_OF_AI (128261) at {new_tokens.index(START_OF_AI)}")
            if END_OF_AI in new_tokens:
                special_tokens_found.append(f"END_OF_AI (128262) at {new_tokens.index(END_OF_AI)}")
            
            print(f"\nDEBUG - Special tokens found: {special_tokens_found if special_tokens_found else 'None'}")
        
        return generated_ids
    
    def synthesize(
        self,
        text: str,
        voice: str = "Babla",
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        debug: bool = False
    ) -> np.ndarray:
        """
        Synthesize speech from text (end-to-end).
        
        Args:
            text: Text to convert to speech
            voice: Voice ID
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            debug: Print debug information
            
        Returns:
            Audio array [1, T_audio]
        """
        # Generate tokens
        generated_ids = self.generate(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            debug=debug
        )
        
        # Extract audio tokens
        print("Extracting audio tokens...")
        audio_tokens = extract_audio_tokens(generated_ids)
        
        if not audio_tokens:
            print("Error: No audio tokens found in generated sequence")
            if debug:
                print(f"DEBUG - Full token sequence: {generated_ids}")
            return None
        
        print(f"Audio tokens (raw): {len(audio_tokens)}")
        
        # Remove repetitive loops (model sometimes gets stuck)
        audio_tokens = detect_repetition_and_truncate(audio_tokens, window_size=4, threshold=10)
        
        print(f"Audio tokens (cleaned): {len(audio_tokens)}")
        
        if debug:
            print(f"DEBUG - First 20 audio tokens: {audio_tokens[:20]}")
            print(f"DEBUG - Last 20 audio tokens: {audio_tokens[-20:]}")
            print(f"DEBUG - Audio token range: [{min(audio_tokens)}, {max(audio_tokens)}]")
        
        # Deinterleave tokens
        print("Deinterleaving tokens...")
        semantic_codes, acoustic_codes = deinterleave_tokens(audio_tokens)
        
        if len(semantic_codes) == 0:
            print("Error: No valid frames found")
            return None
        
        print(f"Semantic codes: {semantic_codes.shape}")
        print(f"Acoustic codes: {acoustic_codes.shape}")
        
        if debug:
            print(f"DEBUG - Semantic code range: [{semantic_codes.min()}, {semantic_codes.max()}]")
            print(f"DEBUG - Acoustic code range: [{acoustic_codes.min()}, {acoustic_codes.max()}]")
        
        # Add batch dimension
        semantic_codes = semantic_codes[np.newaxis, :]  # [1, T]
        acoustic_codes = acoustic_codes[np.newaxis, :, :]  # [1, N_q, T]
        
        # Decode to audio
        print("Decoding audio...")
        audio = self.decoder.decode(semantic_codes, acoustic_codes)
        
        duration = audio.shape[-1] / self.output_rate
        print(f"✓ Audio generated: {audio.shape}, duration: {duration:.2f}s")
        
        return audio
    
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        voice: str = "Babla",
        temperature: float = 0.4,
        top_p: float = 0.9,
        top_k: int = 40,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.1,
        debug: bool = False
    ):
        """
        Synthesize speech and save to file.
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice ID
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty
            debug: Print debug information
        """
        # Synthesize
        audio = self.synthesize(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            debug=debug
        )
        
        if audio is None:
            print("Error: Failed to synthesize audio")
            return False
        
        # Convert to tensor and save
        audio_tensor = torch.from_numpy(audio).squeeze(0)  # Remove batch dim
        torchaudio.save(output_path, audio_tensor, self.output_rate)
        print(f"✓ Saved audio to: {output_path}")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="LongCat TTS Inference using Transformers (without LMDeploy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="Path to LongCat language model (HuggingFace format)"
    )
    parser.add_argument(
        '--decoder',
        type=str,
        required=True,
        help="Path to ONNX decoder model"
    )
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help="Text to convert to speech"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help="Output audio file path"
    )
    parser.add_argument(
        '--voice',
        type=str,
        default='Babla',
        help="Voice ID (e.g., Babla, Pallavi, Anuradha, Abhishek)"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.4,
        help="Sampling temperature (0.0-1.0)"
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=40,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024, ~5-10 seconds of audio)"
    )
    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=1.2,
        help="Repetition penalty (higher = less repetition, default: 1.2)"
    )
    parser.add_argument(
        '--output_rate',
        type=int,
        default=24000,
        choices=[16000, 24000],
        help="Output sample rate"
    )
    parser.add_argument(
        '--use_tensorrt',
        action='store_true',
        help="Use TensorRT for decoder acceleration"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Device for language model"
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        help="CUDA device ID"
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug output (shows generated tokens and detailed info)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LongCat TTS - Transformers Inference")
    print("="*70 + "\n")
    
    # Initialize TTS system
    tts = LongCatTTS(
        model_path=args.model,
        decoder_path=args.decoder,
        output_rate=args.output_rate,
        use_tensorrt=args.use_tensorrt,
        device_id=args.device_id,
        device=args.device
    )
    
    # Synthesize
    print("="*70)
    print("Starting synthesis...")
    print("="*70 + "\n")
    
    success = tts.synthesize_to_file(
        text=args.text,
        output_path=args.output,
        voice=args.voice,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        debug=args.debug
    )
    
    if success:
        print("\n" + "="*70)
        print("✓ SUCCESS!")
        print("="*70)
        print(f"Audio saved to: {args.output}")
    else:
        print("\n" + "="*70)
        print("✗ FAILED")
        print("="*70)


if __name__ == '__main__':
    main()

