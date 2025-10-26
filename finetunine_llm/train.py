from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments, 
    AutoTokenizer, 
    LlamaTokenizer,
    TrainerCallback,
    integrations
)
import numpy as np
import yaml
import wandb
import os
import torch
import torch.distributed as dist
import sys
import logging
from datetime import datetime, timedelta
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set NCCL environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_TIMEOUT"] = "23"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for mixed GPU setup
os.environ["NCCL_BLOCKING_WAIT"] = "1"

# Set PyTorch memory management variables to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# class MemoryManagementCallback(TrainerCallback):
#     """Callback to manage GPU memory and prevent fragmentation."""
    
#     def __init__(self, clear_cache_steps=200):
#         self.clear_cache_steps = clear_cache_steps
#         self.step_count = 0
        
#     def on_step_end(self, args, state, control, **kwargs):
#         """Clear cache periodically to prevent memory fragmentation."""
#         self.step_count += 1
        
#         # Clear cache every N steps (balanced for speed vs memory)
#         if self.step_count % self.clear_cache_steps == 0:
#             torch.cuda.empty_cache()
            
#             # Only log memory stats occasionally (every 1000 steps) to minimize overhead
#             if self.step_count % 1000 == 0:
#                 if not dist.is_initialized() or dist.get_rank() == 0:
#                     for i in range(torch.cuda.device_count()):
#                         allocated = torch.cuda.memory_allocated(i) / 1024**3
#                         reserved = torch.cuda.memory_reserved(i) / 1024**3
#                         logger.info(f"Step {self.step_count} - GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
#         return control

def find_latest_valid_checkpoint(output_dir):
    """Finds the latest valid checkpoint in a directory."""
    # Get all checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    # Filter out paths that are not directories
    checkpoint_dirs = [d for d in checkpoint_dirs if os.path.isdir(d)]

    if not checkpoint_dirs:
        return None

    # Sort by step number
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)

    for checkpoint in checkpoint_dirs:
        if os.path.exists(os.path.join(checkpoint, "trainer_state.json")):
            return checkpoint
    return None

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

dsn = config["TTS_dataset"]

model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
save_total_limit = config["save_total_limit"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
lr_scheduler_type = config["lr_scheduler_type"]
resume_from_checkpoint = config.get("resume_from_checkpoint", False)

# LongCat token configuration
add_custom_tokens = config.get("add_custom_tokens", False)
n_acoustic_codebooks = config.get("n_acoustic_codebooks", 3)
base_llama_tokenizer = config.get("base_llama_tokenizer", "meta-llama/Llama-3.2-3B")

# Check if running in distributed mode
is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
local_rank = int(os.environ.get("LOCAL_RANK", 0))

print(f"Distributed training: {is_distributed}")
print(f"Local rank: {local_rank}")
print(f"CUDA devices available: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")

# Initialize process group for distributed training
if is_distributed:
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=180))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Loading tokenizer for model: {model_name}")

# If we need to replace custom tokens, load from base LLAMA tokenizer
if add_custom_tokens:
    print(f"  ⚠️ Loading clean tokenizer from {base_llama_tokenizer}")
    print(f"  This will discard any custom tokens from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_llama_tokenizer)
    print(f"  Base tokenizer loaded with {len(tokenizer):,} tokens")
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"  Loaded tokenizer with {len(tokenizer):,} tokens")

# Set up padding token for data collation
if tokenizer.pad_token is None:
    tokenizer.pad_token_id = pad_token
    print(f"Set pad_token_id to {pad_token}")
else:
    print(f"Using existing pad_token_id: {tokenizer.pad_token_id}")

# Add LongCat custom tokens if enabled
if add_custom_tokens:
    print(f"\n{'='*70}")
    print(f"Checking LongCat tokens in tokenizer and model")
    print(f"{'='*70}")
    
    # LongCat token configuration
    SEMANTIC_CODEBOOK_SIZE = 8192
    ACOUSTIC_CODEBOOK_SIZE = 1024
    NUM_SPECIAL_TOKENS = 10
    
    total_audio_tokens = SEMANTIC_CODEBOOK_SIZE + (n_acoustic_codebooks * ACOUSTIC_CODEBOOK_SIZE)
    number_add_tokens = NUM_SPECIAL_TOKENS + total_audio_tokens
    
    print(f"  Semantic codebook: {SEMANTIC_CODEBOOK_SIZE:,} tokens")
    print(f"  Acoustic codebooks: {n_acoustic_codebooks} × {ACOUSTIC_CODEBOOK_SIZE:,} = {n_acoustic_codebooks * ACOUSTIC_CODEBOOK_SIZE:,} tokens")
    print(f"  Special tokens: {NUM_SPECIAL_TOKENS}")
    print(f"  Total new tokens needed: {number_add_tokens:,}")
    
    # If replacing tokens, we already have a clean tokenizer, so just add
    print(f"\n  🔄 REPLACING CUSTOM TOKENS mode:")
    print(f"     Using clean base tokenizer + adding fresh custom tokens")
    new_tokens = [f"<custom_token_{i}>" for i in range(0, number_add_tokens + 1)]
    num_added = tokenizer.add_tokens(new_tokens)
    print(f"  ✓ Added {num_added} fresh tokens to clean tokenizer")
    print(f"  ✓ New tokenizer length: {len(tokenizer):,}")
    
    print(f"{'='*70}\n")
else:
    print(f"\n⚠ Skipping LongCat token addition (add_custom_tokens=False)")
    print(f"  Current tokenizer length: {len(tokenizer):,}")
    print(f"  Use this mode when fine-tuning on a model that already has LongCat tokens\n")

print(f"Loading model from {model_name}")

# Clear any cached memory before loading model
torch.cuda.empty_cache()

if is_distributed:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to(device)
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

# Resize model embeddings if needed
if add_custom_tokens:
    old_vocab_size = model.config.vocab_size
    tokenizer_size = len(tokenizer)
    
    if old_vocab_size != tokenizer_size:
        print(f"\n{'='*70}")
        print(f"Resizing model embeddings to match tokenizer...")
        print(f"{'='*70}")
        
        print(f"  🔄 REPLACING CUSTOM TOKENS:")
        print(f"     Old model vocab: {old_vocab_size:,} (includes {old_vocab_size - 128256:,} old custom tokens)")
        print(f"     New tokenizer:   {tokenizer_size:,} (includes {tokenizer_size - 128256:,} new custom tokens)")
        print(f"     → Keeping base {128256:,} embeddings from pretrained model")
        print(f"     → Discarding {old_vocab_size - 128256:,} old custom token embeddings")
        print(f"     → Adding {tokenizer_size - 128256:,} new random-initialized custom token embeddings")
        
        model.resize_token_embeddings(tokenizer_size)
        new_vocab_size = model.config.vocab_size
        print(f"  ✓ Resized model embeddings: {old_vocab_size:,} → {new_vocab_size:,}")
        print(f"{'='*70}\n")
        
        # Clear cache after resizing
        torch.cuda.empty_cache()
    else:
        print(f"✓ Model embeddings already match tokenizer size: {old_vocab_size:,}")
        print(f"  No resizing needed.")

# Enable gradient checkpointing with memory-efficient settings
model.gradient_checkpointing_enable()

# Clear cache again after model loading
torch.cuda.empty_cache()

# Log initial memory usage
if not is_distributed or local_rank == 0:
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        logger.info(f"Initial GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

print(f"Loading dataset from {dsn}")
dataset = load_from_disk(dsn)
ds = dataset["train"] if "train" in dataset else dataset

if not is_distributed or local_rank == 0:
    wandb.init(project=project_name, name=run_name)

save_dir = f"./{base_repo_id}"
os.makedirs(save_dir, exist_ok=True)

training_args = TrainingArguments(
    overwrite_output_dir=False,  # Allow resuming from checkpoints
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size, 
    logging_steps=10,  # Reduced logging frequency for speed (was 1)
    bf16=True,
    gradient_checkpointing=True,
    gradient_accumulation_steps=4,  # Balanced for throughput and memory
    output_dir=save_dir,
    report_to="wandb" if (not is_distributed or local_rank == 0) else None,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    remove_unused_columns=True, 
    learning_rate=learning_rate,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False
)

# Create custom data collator for TTS dataset (matches pretraining approach)
def data_collator(features):
    input_ids = [f["input_ids"] for f in features]

    if any("attention_mask" not in f for f in features):
        attention_mask = [[1]*len(ids) for ids in input_ids]
    else:
        attention_mask = [f["attention_mask"] for f in features]

    if any("labels" not in f for f in features):
        labels = input_ids
    else:
        labels = [f["labels"] for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        i, dtype=torch.long) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        m, dtype=torch.long) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        l, dtype=torch.long) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

print("Setting up custom data collator for TTS dataset...")

print("\n🔍 Initializing trainer...")
try:
    # Initialize memory management callback (clear cache every 200 steps for better speed)
    # memory_callback = MemoryManagementCallback(clear_cache_steps=200)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        # callbacks=[memory_callback],
    )
    print(f"✓ Trainer initialized successfully on rank {local_rank}")
    if is_distributed:
        dist.barrier()  # Synchronize before training starts
except Exception as e:
    print(f"✗ Trainer initialization failed on rank {local_rank}: {e}")
    raise

print(f"\n🔍 Starting training loop on rank {local_rank}...")
if local_rank == 0:
    print("This might take several minutes for the first step...")

# Clear cache before training starts
torch.cuda.empty_cache()

resume_from_checkpoint_path = False
if resume_from_checkpoint:
    latest_valid_checkpoint = find_latest_valid_checkpoint(save_dir)
    if latest_valid_checkpoint:
        print(f"Resuming training from {latest_valid_checkpoint}")
        resume_from_checkpoint_path = latest_valid_checkpoint
    else:
        print("No valid checkpoint found, starting training from scratch.")

try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint_path)
    print("\n✓ Training completed successfully!")
except Exception as e:
    print(f"\n✗ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

# Save the model (only on rank 0)
if not is_distributed or local_rank == 0:
    print(f"Saving model to {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save in float16 format for VLLM and GGUF/llama.cpp compatibility
    fp16_dir = os.path.join(save_dir, "fp16")
    os.makedirs(fp16_dir, exist_ok=True)

    print(f"Converting model to float16 and saving to {fp16_dir}")
    model_fp16 = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.float16)
    model_fp16.save_pretrained(fp16_dir)
    tokenizer.save_pretrained(fp16_dir)

    # Save in bfloat16 format
    bf16_dir = os.path.join(save_dir, "bf16")
    os.makedirs(bf16_dir, exist_ok=True)

    print(f"Converting model to bfloat16 and saving to {bf16_dir}")
    model_bf16 = AutoModelForCausalLM.from_pretrained(save_dir, torch_dtype=torch.bfloat16)
    model_bf16.save_pretrained(bf16_dir)
    tokenizer.save_pretrained(bf16_dir)

    print("Saving GGUF-compatible model")
    gguf_dir = os.path.join(save_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    try:
        from transformers.utils.quantization_config import QuantizationMethod
        
        # Save config for GGUF conversion
        with open(os.path.join(gguf_dir, "config.json"), "w") as f:
            f.write(model_fp16.config.to_json_string())
        
        print("Model and tokenizer saved successfully in multiple formats")
        print(f"Standard model: {save_dir}")
        print(f"Float16 model (VLLM): {fp16_dir}")
        print(f"Bfloat16 model: {bf16_dir}")
        print(f"GGUF-ready model: {gguf_dir}")
        print("To convert to GGUF format, use the llama.cpp tools")
        
    except ImportError:
        print("QuantizationMethod not found, skipping GGUF config preparation")
        print("Model saved in standard, float16, and bfloat16 formats")
