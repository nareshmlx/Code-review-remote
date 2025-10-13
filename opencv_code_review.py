#!/usr/bin/env python3
"""
OpenCV Code Reviewer Fine-tuning Script
Fine-tunes Qwen2.5-Coder-7B-Instruct for OpenCV code review tasks
"""
import unsloth  # Ensure unsloth is imported first
import os
import re
import torch
from datasets import load_dataset
from transformers import TextStreamer, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MAX_SEQ_LENGTH = 16384  # Maximum sequence length for context
DTYPE = None  # Auto-detect (Float16 for Tesla T4/V100, Bfloat16 for Ampere+)
LOAD_IN_4BIT = True  # Use 4-bit quantization to reduce memory
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"

# LoRA Configuration
LORA_R = 16  # LoRA rank (higher = more parameters, better fit, more memory)
LORA_ALPHA = 16  # LoRA scaling factor
LORA_DROPOUT = 0  # Dropout for LoRA layers (0 is optimized)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Configuration
PER_DEVICE_BATCH_SIZE = 1  # Keep at 1 for 16k context
GRADIENT_ACCUMULATION_STEPS = 16  # Effective batch size = 16
WARMUP_STEPS = 50
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
LOGGING_STEPS = 5
OPTIMIZER = "paged_adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 250
SAVE_TOTAL_LIMIT = 2

# Dataset Configuration
DATASET_NAME = "nareshmlx/16k_opencvpr"

# HuggingFace Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_NAME = "nareshmlx/code-reviewer-opencv-16k"

# Output Configuration
OUTPUT_DIR = "outputs"
RANDOM_SEED = 3407

# ============================================================================
# SETUP AND INSTALLATION
# ============================================================================

def setup_environment():
    """Install required packages based on environment"""
    print("Setting up environment...")
    
    # Check if running in Colab
    is_colab = "COLAB_" in "".join(os.environ.keys())
    
    if not is_colab:
        os.system("pip install -q unsloth")
    else:
        # Colab-specific installation
        v = re.match(r"[0-9\.]{3,}", str(torch.__version__)).group(0)
        xformers = "xformers==" + ("0.0.32.post2" if v == "2.8.0" else "0.0.29.post3")
        os.system(f"pip install -q --no-deps bitsandbytes accelerate {xformers} peft trl triton cut_cross_entropy unsloth_zoo")
        os.system('pip install -q sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer')
        os.system("pip install -q --no-deps unsloth")
    
    # Install specific versions
    os.system("pip install -q transformers==4.55.4")
    os.system("pip install -q --no-deps trl==0.22.2")
    
    # Upgrade unsloth for CUDA 12.1 and Ampere GPUs
    os.system("pip uninstall -y unsloth unsloth-zoo torchao")
    os.system('pip install -q --upgrade --no-cache-dir "unsloth[cu121-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"')
    
    # Install specific torchao version
    os.system("pip uninstall -y torchao")
    os.system("pip install -q torchao==0.12.0")
    
    print("✓ Environment setup complete!")

def check_gpu():
    """Check GPU availability and display information"""
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU detected! Training will be very slow.")
    print("="*70 + "\n")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def alpaca_to_conversations(batch):
    """Convert Alpaca format to conversation format"""
    conversations = []
    for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
        if inp.strip():
            user_msg = f"{instr}\n\nHere is the code:\n{inp}"
        else:
            user_msg = instr

        conversations.append([
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": out}
        ])
    return {"conversations": conversations}

def formatting_prompts_func(examples, tokenizer):
    """Format conversations using chat template"""
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        )
        for convo in examples["conversations"]
    ]
    return {"text": texts}

def prepare_dataset(tokenizer):
    """Load and prepare the dataset"""
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    # Load dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Standardize and convert format
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(
        alpaca_to_conversations,
        batched=True,
        remove_columns=["instruction", "input", "output"]
    )
    
    # Apply chat template
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, tokenizer),
        batched=True
    )
    
    print(f"✓ Dataset prepared with {len(dataset)} examples")
    print("="*70 + "\n")
    
    return dataset

# ============================================================================
# MODEL SETUP
# ============================================================================

def load_model():
    """Load the base model with 4-bit quantization"""
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    print("✓ Model loaded successfully!")
    print("="*70 + "\n")
    
    return model, tokenizer

def add_lora_adapters(model):
    """Add LoRA adapters to the model"""
    print("\n" + "="*70)
    print("ADDING LoRA ADAPTERS")
    print("="*70)
    print(f"LoRA rank (r): {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Target modules: {TARGET_MODULES}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ LoRA adapters added successfully!")
    print("="*70 + "\n")
    
    return model

def setup_tokenizer(tokenizer):
    """Setup chat template for tokenizer"""
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )
    return tokenizer

# ============================================================================
# TRAINING
# ============================================================================

def create_trainer(model, tokenizer, dataset):
    """Create and configure the trainer"""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Optimizer: {OPTIMIZER}")
    print(f"LR Scheduler: {LR_SCHEDULER_TYPE}")
    print("="*70 + "\n")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=True,  # Can make training 5x faster for short sequences
        args=SFTConfig(
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=RANDOM_SEED,
            output_dir=OUTPUT_DIR,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=SAVE_TOTAL_LIMIT,
            fp16=False,
            bf16=True,  # Use bfloat16 for Ampere+ GPUs
            gradient_checkpointing=True,
            max_grad_norm=MAX_GRAD_NORM,
            report_to="none",
        ),
    )
    
    return trainer

def train_model(trainer):
    """Train the model and display statistics"""
    # Show memory before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved memory before training: {start_gpu_memory} GB")
    print("="*70 + "\n")
    
    # Train
    trainer_stats = trainer.train()
    
    # Show memory after training
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak reserved memory: {used_memory} GB")
    print(f"Peak reserved memory for training: {used_memory_for_lora} GB")
    print(f"Peak reserved memory % of max memory: {used_percentage}%")
    print(f"Peak reserved memory for training % of max memory: {lora_percentage}%")
    print("="*70 + "\n")
    
    return trainer_stats

# ============================================================================
# INFERENCE AND TESTING
# ============================================================================

def test_model(model, tokenizer):
    """Test the trained model with a sample code review"""
    print("\n" + "="*70)
    print("TESTING MODEL")
    print("="*70)
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    messages = [
        {"role": "user", "content":
         "You are an expert OpenCV code reviewer. Review this change:\n\n"
         "File: modules/imgproc/src/resize.cpp\n"
         "@@ -100,7 +100,7 @@\n"
         " cv::Mat src, dst;\n"
         "-cv::resize(src, dst, cv::Size(100,100));\n"
         "+cv::resize(src, dst, cv::Size(100,100), CV_INTER_LINEAR);\n"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("\nModel Response:")
    print("-" * 70)
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.3,  # Lower temperature for code accuracy
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    print("-" * 70)
    print("="*70 + "\n")

# ============================================================================
# MODEL SAVING
# ============================================================================

def save_model(model, tokenizer):
    """Save the model to HuggingFace Hub"""
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    print(f"Pushing to: {HF_REPO_NAME}")
    
    model.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
    
    print("✓ Model saved successfully!")
    print("="*70 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow"""
    print("\n" + "="*70)
    print("OPENCV CODE REVIEWER TRAINING SCRIPT")
    print("="*70 + "\n")
    
    # Setup
    setup_environment()
    check_gpu()
    
    # Load model
    model, tokenizer = load_model()
    tokenizer = setup_tokenizer(tokenizer)
    
    # Add LoRA adapters
    model = add_lora_adapters(model)
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, dataset)
    
    # Train
    trainer_stats = train_model(trainer)
    
    # Test
    test_model(model, tokenizer)
    
    # Save
    save_model(model, tokenizer)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()