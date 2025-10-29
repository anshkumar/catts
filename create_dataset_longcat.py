import os
import argparse
import numpy as np
import torch
import torchaudio
from datasets import Dataset, Features, Sequence, Value
from transformers import AutoTokenizer
import json
from pathlib import Path
import locale
from longcat_encoder import create_longcat_encoder

# Set locale for UTF-8 support
locale.getpreferredencoding = lambda: "UTF-8"

# Base configuration
OUTPUT_DIR = "combined_tts_dataset_longcat"

# Token constants (similar to SNAC but for LongCat)
TOKENISER_LENGTH = 128256
START_OF_TEXT = 128000
END_OF_TEXT = 128009
START_OF_SPEECH = TOKENISER_LENGTH + 1  # 128257
END_OF_SPEECH = TOKENISER_LENGTH + 2    # 128258
START_OF_HUMAN = TOKENISER_LENGTH + 3    # 128259
END_OF_HUMAN = TOKENISER_LENGTH + 4     # 128260
START_OF_AI = TOKENISER_LENGTH + 5       # 128261
END_OF_AI = TOKENISER_LENGTH + 6         # 128262
PAD_TOKEN = TOKENISER_LENGTH + 7         # 128263
AUDIO_TOKENS_START = TOKENISER_LENGTH + 10  # 128266

# LongCat token configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LongCat uses Adaptive Grouped RVQ with product quantization:
# - Each acoustic codebook uses 2 internal codebooks of size 90
# - Effective size per acoustic codebook: 90 × 90 = 8,100
#
# For LLM training, add these tokens:
# number_add_tokens = 32502  # 8192 (semantic) + 3×8100 (acoustic) + 10 (special)
# new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
# tokenizer.add_tokens(new_tokens)
# model.resize_token_embeddings(len(tokenizer))
#
# Token ID ranges:
#   128257-128266: Special tokens (10)
#   128266-136457: Semantic tokens (8,192)
#   136458-144557: Acoustic codebook 0 (8,100)
#   144558-152657: Acoustic codebook 1 (8,100)  
#   152658-160757: Acoustic codebook 2 (8,100)
#
# Total: 32,502 new tokens (vs 28,682 for SNAC)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEMANTIC_CODEBOOK_SIZE = 8192
ACOUSTIC_CODEBOOK_SIZE = 8100  # 90 × 90 from Adaptive Grouped RVQ product quantization


def load_audio_with_longcat(
    audio_path: str,
    longcat_codec,
    normalize: bool = False
) -> list:
    """
    Load and tokenize audio using LongCat codec.
    
    Args:
        audio_path: Path to audio file
        longcat_codec: LongCat hybrid codec instance
        normalize: Whether to normalize audio
        
    Returns:
        List of LongCat audio tokens in interleaved format
    """
    try:
        # Load audio
        if normalize:
            try:
                from pydub import AudioSegment
                import tempfile
                audio_segment = AudioSegment.from_file(audio_path)
                normalized_audio = audio_segment.normalize()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    normalized_audio.export(temp_file.name, format="wav")
                    waveform, sample_rate = torchaudio.load(temp_file.name)
                    os.unlink(temp_file.name)
            except Exception as e:
                print(f"Warning: Failed to normalize {audio_path}, using original: {e}")
                waveform, sample_rate = torchaudio.load(audio_path)
        else:
            waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Encode with LongCat
        semantic_codes, acoustic_codes = longcat_codec.encode(waveform, sample_rate)
        
        # Interleave semantic and acoustic codes
        # LongCat format: 1 semantic + N acoustic codes per frame
        # For 3 acoustic codebooks: [sem, ac0, ac1, ac2] per frame
        all_codes = []
        
        semantic_np = semantic_codes.cpu().numpy()[0]  # [T]
        acoustic_np = acoustic_codes.cpu().numpy()[0]  # [N_q, T]
        
        n_frames = semantic_np.shape[0]
        n_acoustic_codebooks = acoustic_np.shape[0]
        
        for i in range(n_frames):
            # Semantic token (offset by AUDIO_TOKENS_START)
            all_codes.append(int(semantic_np[i]) + AUDIO_TOKENS_START)
            
            # Acoustic tokens (offset by semantic codebook size)
            # Each acoustic codebook has 8100 codes (90 × 90 from product quantization)
            for q in range(n_acoustic_codebooks):
                offset = AUDIO_TOKENS_START + SEMANTIC_CODEBOOK_SIZE + (q * ACOUSTIC_CODEBOOK_SIZE)
                all_codes.append(int(acoustic_np[q, i]) + offset)
        
        return all_codes
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_entry(
    file_id: str,
    text: str,
    audio_path: str,
    speaker_name: str,
    longcat_codec,
    tokenizer,
    normalize: bool = False,
    language: str = None
):
    """
    Process a single audio-text pair with LongCat.
    
    Args:
        file_id: Unique identifier
        text: Text transcription
        audio_path: Path to audio file
        speaker_name: Speaker identifier
        longcat_codec: LongCat codec instance
        tokenizer: Text tokenizer
        normalize: Whether to normalize audio
        language: Language name (optional)
    """
    try:
        # Tokenize audio with LongCat
        audio_codes = load_audio_with_longcat(audio_path, longcat_codec, normalize)
        
        if not audio_codes:
            return None
        
        # Format text with language and speaker
        if language:
            text_prompt = f"{language} {speaker_name}: {text}"
        else:
            text_prompt = f"{speaker_name}: {text}"
        
        # Tokenize text
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(END_OF_TEXT)
        
        # Create full input tokens
        input_ids = (
            [START_OF_HUMAN]
            + text_ids
            + [END_OF_HUMAN]
            + [START_OF_AI]
            + [START_OF_SPEECH]
            + audio_codes
            + [END_OF_SPEECH]
            + [END_OF_AI]
        )
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "file_id": file_id,
            "source": speaker_name,
            "input_ids": input_ids,
            "labels": input_ids,
            "attention_mask": attention_mask,
            "original_text": text,
            "audio_path": audio_path
        }
        
    except Exception as e:
        print(f"Error processing entry {file_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create TTS dataset using LongCat Audio Codec",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=OUTPUT_DIR,
        help='Directory to save the dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='datasets_config.yaml',
        help='Path to dataset configuration YAML file'
    )
    parser.add_argument(
        '--n_acoustic_codebooks',
        type=int,
        default=3,
        help='Number of acoustic codebooks (1-3 for LongCat)'
    )
    parser.add_argument(
        '--device_id',
        type=int,
        default=0,
        help='CUDA device ID'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LongCat Dataset Creation")
    print("="*70)
    print(f"Output directory: {args.output_dir}")
    print(f"Acoustic codebooks: {args.n_acoustic_codebooks}")
    print("="*70)
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

    # Add custom tokens (same as in train.py)
    # LongCat uses Adaptive Grouped RVQ with product quantization: 90 × 90 = 8,100
    SEMANTIC_CODEBOOK_SIZE = 8192
    ACOUSTIC_CODEBOOK_SIZE = 8100
    NUM_SPECIAL_TOKENS = 10
    
    total_audio_tokens = SEMANTIC_CODEBOOK_SIZE + (args.n_acoustic_codebooks * ACOUSTIC_CODEBOOK_SIZE)
    number_add_tokens = NUM_SPECIAL_TOKENS + total_audio_tokens

    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"✓ Added {num_added} custom tokens to tokenizer")
    print(f"✓ Total tokenizer vocabulary size: {len(tokenizer):,}")
    print(f"  Token breakdown:")
    print(f"    - Semantic:  {SEMANTIC_CODEBOOK_SIZE:,} tokens")
    print(f"    - Acoustic:  {args.n_acoustic_codebooks} × {ACOUSTIC_CODEBOOK_SIZE:,} = {args.n_acoustic_codebooks * ACOUSTIC_CODEBOOK_SIZE:,} tokens")
    print(f"    - Special:   {NUM_SPECIAL_TOKENS} tokens")
    print(f"    - Total new: {number_add_tokens:,} tokens")

    # Initialize LongCat encoder
    print("\nInitializing LongCat encoder...")
    longcat_codec = create_longcat_encoder(
        device_id=args.device_id,
        n_acoustic_codebooks=args.n_acoustic_codebooks
    )
    
    # Load dataset configuration
    import yaml
    print(f"\nLoading dataset configuration from: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    dataset_configs = config.get('datasets', [])
    print(f"Found {len(dataset_configs)} dataset configurations")
    
    # Process all datasets
    all_entries = []
    
    for i, ds_config in enumerate(dataset_configs):
        dataset_path = ds_config.get('dataset_path')
        speaker_name = ds_config.get('speaker_name')
        num_samples = ds_config.get('num_samples')
        normalize = ds_config.get('normalize', False)
        language = ds_config.get('language')
        
        print(f"\n{'='*70}")
        print(f"Processing dataset {i+1}/{len(dataset_configs)}: {speaker_name}")
        print(f"{'='*70}")
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue
        
        # Load dataset (CSV or JSONL)
        if dataset_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(dataset_path, on_bad_lines='skip', engine='python')
            
            # Standardize columns
            if 'transcription' in df.columns:
                df['text'] = df['transcription']
            if 'audio' in df.columns:
                df['audio_path'] = df['audio']
            if 'audio_file' in df.columns:
                df['audio_path'] = df['audio_file']
            
            # Sample if needed
            if num_samples and num_samples < len(df):
                df = df.sample(n=num_samples, random_state=42)
            
            print(f"Processing {len(df)} samples...")
            
            if speaker_name == "" or speaker_name is None:
                override_speaker_name = True
            else:
                override_speaker_name = False
            
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(df)}")
                
                # Override speaker name if needed (for multi-speaker datasets)
                if override_speaker_name:
                    current_speaker_name = row.get('speaker_name', f'speaker_{idx}')
                else:
                    current_speaker_name = speaker_name
                
                entry = process_entry(
                    file_id=row.get('file_id', f'entry_{idx}'),
                    text=row['text'],
                    audio_path=row['audio_path'],
                    speaker_name=current_speaker_name,
                    longcat_codec=longcat_codec,
                    tokenizer=tokenizer,
                    normalize=normalize,
                    language=language
                )
                
                if entry:
                    all_entries.append(entry)
        
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            # Read JSONL file line by line
            entries = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON line: {e}")
                        continue
            
            # Apply sampling if specified
            if num_samples is not None:
                import random
                random.seed(42)
                entries = random.sample(entries, min(num_samples, len(entries)))
            
            print(f"Processing {len(entries)} samples...")
            
            # Check if speaker name should be overridden from manifest
            if speaker_name == "" or speaker_name is None:
                override_speaker_name = True
            else:
                override_speaker_name = False
            
            # Process each entry
            for idx, entry_data in enumerate(entries):
                if idx % 100 == 0:
                    print(f"  Progress: {idx}/{len(entries)}")
                
                # Extract data
                if 'audio_filepath' not in entry_data or 'text' not in entry_data:
                    print(f"Skipping entry {idx}: Missing required fields")
                    continue
                
                text = entry_data['text']
                audio_path = entry_data['audio_filepath']
                
                # Override speaker name if needed (for multi-speaker datasets)
                if override_speaker_name:
                    current_speaker_name = entry_data.get('speaker', f'speaker_{idx}')
                else:
                    current_speaker_name = speaker_name
                
                # Create a file ID
                file_id = f"entry_{idx}"
                
                # Process the entry
                processed_entry = process_entry(
                    file_id=file_id,
                    text=text,
                    audio_path=audio_path,
                    speaker_name=current_speaker_name,
                    longcat_codec=longcat_codec,
                    tokenizer=tokenizer,
                    normalize=normalize,
                    language=language
                )
                
                if processed_entry:
                    all_entries.append(processed_entry)
        
        else:
            print(f"Warning: Unsupported file format: {dataset_path}")
            continue
    
    # Create dataset
    print(f"\n{'='*70}")
    print(f"Creating final dataset...")
    print(f"Total entries: {len(all_entries)}")
    print(f"{'='*70}")
    
    # Extract speakers
    speakers = list(set(entry["source"] for entry in all_entries))
    
    # Define features
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'labels': Sequence(Value('int64')),
        'attention_mask': Sequence(Value('int64'))
    })
    
    # Keep only required columns
    minimal_entries = []
    for entry in all_entries:
        minimal_entries.append({
            'input_ids': entry['input_ids'],
            'labels': entry['labels'],
            'attention_mask': entry['attention_mask']
        })
    
    combined_dataset = Dataset.from_list(minimal_entries, features=features)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "codec": "LongCat",
        "acoustic_codebooks": args.n_acoustic_codebooks,
        "speakers": speakers,
        "total_examples": len(all_entries),
        "examples_by_speaker": {
            speaker: sum(1 for entry in all_entries if entry["source"] == speaker)
            for speaker in speakers
        }
    }
    
    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Save dataset
    combined_dataset.save_to_disk(args.output_dir)
    
    print(f"\n{'='*70}")
    print("DATASET CREATION COMPLETE")
    print(f"{'='*70}")
    print(f"Dataset saved to: {args.output_dir}")
    print(f"Total examples: {len(combined_dataset)}")
    print(f"Speakers: {', '.join(speakers)}")
    print(f"\n✓ Ready for training!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

