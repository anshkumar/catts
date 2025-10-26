# Catts TTS Training

## 🛠️ Installation

### 1. Create Environment

```bash
conda create -n catts python=3.10
conda activate catts
```

### 2. Install Dependencies

```bash
pip install torch torchaudio transformers datasets accelerate
pip install onnxruntime-gpu pyyaml pandas soundfile pydub
pip install wandb  # For training metrics
```

---

## 📦 Model Download

### Required Files

Download these files from [HuggingFace](https://huggingface.co/meituan-longcat/LongCat-Audio-Codec):

```bash
# Create directories
mkdir -p ckpts onnx_models

# Download encoder checkpoint
wget https://huggingface.co/meituan-longcat/LongCat-Audio-Codec/resolve/main/ckpts/LongCatAudioCodec_encoder.pt \
     -O ckpts/LongCatAudioCodec_encoder.pt

# Download encoder CMVN (important!)
wget https://huggingface.co/meituan-longcat/LongCat-Audio-Codec/resolve/main/ckpts/LongCatAudioCodec_encoder_cmvn.npy \
     -O ckpts/LongCatAudioCodec_encoder_cmvn.npy

# Download decoder checkpoint
wget https://huggingface.co/meituan-longcat/LongCat-Audio-Codec/resolve/main/ckpts/LongCatAudioCodec_decoder_24k_4codebooks.pt \
     -O ckpts/LongCatAudioCodec_decoder_24k_4codebooks.pt
```

### Export to ONNX (for faster inference)

```bash
python export_longcat_to_onnx.py --skip_encoder --simplify
```

This creates:
- `onnx_models/longcat_decoder_24k_simplified.onnx` (576 MB)
- `onnx_models/longcat_decoder_16k_simplified.onnx` (620 MB)

---

## 📊 Dataset Preparation

### Input Format Options

Catts supports **CSV** or **JSONL** datasets with audio and text.

#### Option 1: CSV Format

Create a CSV file with these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `audio_path` | ✅ Yes | Path to audio file | `/data/audio/sample1.wav` |
| `text` | ✅ Yes | Text transcription | `नमस्ते, आप कैसे हैं?` |
| `speaker_name` | Optional | Speaker identifier | `speaker_1` |
| `file_id` | Optional | Unique identifier | `audio_001` |

**Example CSV** (`my_dataset.csv`):
```csv
audio_path,text,speaker_name
/data/audio/file1.wav,नमस्ते दोस्तों,female_speaker
/data/audio/file2.wav,आज मौसम अच्छा है,male_speaker
/data/audio/file3.wav,धन्यवाद,female_speaker
```

#### Option 2: JSONL Format

Create a JSONL file (one JSON object per line):

**Example JSONL** (`my_dataset.jsonl`):
```jsonl
{"audio_filepath": "/data/audio/file1.wav", "text": "नमस्ते दोस्तों", "speaker": "female_speaker"}
{"audio_filepath": "/data/audio/file2.wav", "text": "आज मौसम अच्छा है", "speaker": "male_speaker"}
{"audio_filepath": "/data/audio/file3.wav", "text": "धन्यवाद", "speaker": "female_speaker"}
```

**Required fields**:
- `audio_filepath` or `audio_path`
- `text` or `transcription`
- `speaker` (optional)

### Audio Requirements

- **Format**: WAV, MP3, FLAC (any format supported by `torchaudio`)
- **Sample Rate**: Any (auto-resampled to 16kHz for encoding)
- **Channels**: Mono or stereo (auto-converted to mono)
- **Duration**: Less than 30 seconds per file
- **Quality**: Clean speech without excessive background noise

---

## 🎯 Dataset Creation

### Step 1: Create Configuration File

Create `datasets_config.yaml`:

```yaml
datasets:
  - dataset_path: "/data/hindi_dataset.csv"
    speaker_name: "hindi_speaker_1"
    num_samples: 1000  # Optional: limit samples
    normalize: false   # Optional: normalize audio
    language: "Hindi"  # Optional: add language prefix
    
  - dataset_path: "/data/english_dataset.jsonl"
    speaker_name: "english_speaker_1"
    num_samples: 500
    language: "English"
```

**Parameters**:
- `dataset_path`: Path to CSV or JSONL file
- `speaker_name`: Speaker identifier (will override JSONL speaker field)
- `num_samples`: (Optional) Limit number of samples to process
- `normalize`: (Optional) Normalize audio volume (default: false)
- `language`: (Optional) Add language prefix to prompt

### Step 2: Run Dataset Creation

```bash
python create_dataset_longcat.py \
    --config datasets_config.yaml \
    --output_dir /workspace/longcat_dataset \
    --n_acoustic_codebooks 3 \
    --device_id 0
```

**Parameters**:
- `--config`: Path to datasets config YAML
- `--output_dir`: Where to save processed dataset
- `--n_acoustic_codebooks`: 1-3 codebooks (3 = best quality)
- `--device_id`: CUDA device ID (default: 0)

---

## ⚙️ Training Configuration

### Configuration File: `finetunine_llm/config.yaml`

```yaml
# Dataset
TTS_dataset: "/workspace/longcat_dataset"

# Model
model_name: "meta-llama/Llama-3.2-3B"  # Base model to start from
# model_name: "./checkpoints_longcat/checkpoint-5000"  # Or continue from checkpoint

# Training Args
epochs: 3
batch_size: 4  # Adjust based on GPU memory
number_processes: 4  # Number of GPUs
pad_token: 128263
save_steps: 1000
save_total_limit: 2  # Keep only last 2 checkpoints
learning_rate: 5.0e-5
lr_scheduler_type: "constant"

# LongCat Token Configuration
add_custom_tokens: true  # true for first training, false for fine-tuning
n_acoustic_codebooks: 3  # Must match dataset creation

# Naming and paths
save_folder: "checkpoints_longcat_v1"
project_name: "longcat-tts"
run_name: "train-longcat-v1"
resume_from_checkpoint: false
```

### Key Configuration Parameters

#### Model Configuration

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name` | Base model or checkpoint path | `"meta-llama/Llama-3.2-3B"` |
| `TTS_dataset` | Path to processed dataset | `"/workspace/longcat_dataset"` |

#### Training Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `batch_size` | Per-device batch size | 4 (adjust for GPU) |
| `number_processes` | Number of GPUs | 4 for multi-GPU |
| `epochs` | Training epochs | 3-5 for pretraining |
| `learning_rate` | Learning rate | `5.0e-5` for scratch, `1.0e-5` for fine-tune |
| `save_steps` | Save checkpoint every N steps | 1000-5000 |

#### LongCat Token Settings

| Parameter | Description | Values |
|-----------|-------------|--------|
| `add_custom_tokens` | Add LongCat tokens to model | `true` for first training, `false` for fine-tuning |
| `n_acoustic_codebooks` | Acoustic codebooks | 3 (best quality), 2 (balanced), 1 (fastest) |

---

## 🚀 Training

### Single GPU

```bash
cd finetunine_llm
python train.py
```

### Multi-GPU (4 GPUs)

#### Option 1: Using Accelerate

```bash
cd finetunine_llm

accelerate launch --config_file accelerate_config.yaml train.py
```