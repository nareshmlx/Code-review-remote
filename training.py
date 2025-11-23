#!/usr/bin/env python3
"""
OpenCV Code Reviewer Fine-tuning Script
Fine-tunes Qwen2.5-Coder-7B-Instruct for OpenCV code review tasks
Optimized for NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM)
"""
import unsloth  # must be first for patches
import os
import sys
import torch
from datasets import load_dataset
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from dotenv import load_dotenv

load_dotenv(override=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME            = "nareshmlx/code-reviewer-opencv-16k"
MAX_SEQ_LENGTH        = 16384
DTYPE                 = None        # Auto-detect (bfloat16 on Blackwell)
LOAD_IN_4BIT          = True

LORA_R                = 16
LORA_ALPHA            = 16
LORA_DROPOUT          = 0.0
TARGET_MODULES        = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"
]

PER_DEVICE_BATCH_SIZE = 1    # Fit 3×16k contexts on 96 GB
GRAD_ACC_STEPS        = 16   # Effective batch = 24
NUM_TRAIN_EPOCHS      = 2
LEARNING_RATE         = 2.5e-4
WEIGHT_DECAY          = 0.01
LR_SCHEDULER_TYPE     = "cosine"
WARMUP_STEPS          = 50
MAX_GRAD_NORM         = 1.0
LOGGING_STEPS         = 10
SAVE_STEPS            = 250
SAVE_TOTAL_LIMIT      = 2

DATASET_NAME          = "nareshmlx/16k_opencvpr"

HF_TOKEN              = os.getenv("HF_TOKEN")
HF_REPO_NAME          = "nareshmlx/code-reviewer-opencv-16k"

OUTPUT_DIR            = "outputs"
RANDOM_SEED           = 3407

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def prepare_dataset(tokenizer):
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(ds)} examples")
    
    from unsloth.chat_templates import standardize_sharegpt
    ds = standardize_sharegpt(ds)
    
    def to_convs(batch):
        convs = []
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            user = f"{instr}\n\nHere is the code:\n{inp}" if inp.strip() else instr
            convs.append([{"role":"user","content":user},{"role":"assistant","content":out}])
        return {"conversations": convs}
    
    ds = ds.map(to_convs, batched=True, remove_columns=["instruction","input","output"])
    
    def fmt(ex):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in ex["conversations"]
        ]
        return {"text": texts}
    
    ds = ds.map(fmt, batched=True)
    print(f"✓ Dataset prepared with {len(ds)} examples")
    print("="*70 + "\n")
    
    return ds

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("OPENCV CODE REVIEWER TRAINING")
    print("="*70 + "\n")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.2f} GB\n")
    
    # Load model & tokenizer
    print("="*70)
    print("LOADING MODEL")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"4-bit quantization: {LOAD_IN_4BIT}")
    
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "92GB"}  # Use 92GB of 96GB VRAM
    )
    
    print("✓ Model loaded successfully!")
    print("="*70 + "\n")
    
    # Setup tokenizer
    tokenizer = unsloth.chat_templates.get_chat_template(tokenizer, "qwen-2.5")
    
    # Add LoRA adapters
    print("="*70)
    print("ADDING LoRA ADAPTERS")
    print("="*70)
    print(f"LoRA rank (r): {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"Target modules: {TARGET_MODULES}")
    
    model = unsloth.FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
        use_rslora=False,
        loftq_config=None
    )
    
    print("✓ LoRA adapters added successfully!")
    print("="*70 + "\n")
    
    # Prepare dataset
    dataset = prepare_dataset(tokenizer)
    
    # Trainer configuration
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRAD_ACC_STEPS}")
    print(f"Effective batch size: {PER_DEVICE_BATCH_SIZE * GRAD_ACC_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_TRAIN_EPOCHS}")
    print(f"Optimizer: paged_adamw_8bit")
    print(f"LR Scheduler: {LR_SCHEDULER_TYPE}")
    print("="*70 + "\n")
    
    config = SFTConfig(
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        max_grad_norm=MAX_GRAD_NORM,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        output_dir=OUTPUT_DIR,
        seed=RANDOM_SEED,
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        args=config
    )
    
    # Show memory before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    print("="*70)
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
    
    # Inference test
    print("="*70)
    print("TESTING MODEL")
    print("="*70)
    
    unsloth.FastLanguageModel.for_inference(model)
    
    messages = [
        {"role":"user", "content":
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
        return_tensors="pt"
    ).to("cuda")
    
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("\nModel Response:")
    print("-" * 70)
    model.generate(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id
    )
    print("-" * 70)
    print("="*70 + "\n")
    
# Save model - FIXED VERSION
    print("="*70)
    print("SAVING MODEL")
    print("="*70)
    
    # Option 1: Save merged model (best for deployment)
    print("Merging LoRA adapters with base model...")
    merged_model = model.merge_and_unload()
    
    # Save locally first (safer)
    print("\nSaving merged model locally...")
    merged_model.save_pretrained("./local_model_merged")
    tokenizer.save_pretrained("./local_model_merged")
    print("✓ Merged model saved locally to ./local_model_merged")
    
    # Save LoRA adapters separately (smaller, reusable)
    print("\nSaving LoRA adapters...")
    model.save_pretrained("./local_model_lora")
    tokenizer.save_pretrained("./local_model_lora")
    print("✓ LoRA adapters saved locally to ./local_model_lora")
    
    # Push to HuggingFace Hub
    if HF_TOKEN:
        try:
            print(f"\nPushing to HuggingFace Hub: {HF_REPO_NAME}")
            
            # Push merged model
            merged_model.push_to_hub(
                HF_REPO_NAME,
                token=HF_TOKEN,
                commit_message="Fine-tuned OpenCV code reviewer"
            )
            tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
            print("✓ Merged model pushed to HuggingFace Hub!")
            
            # Optionally push LoRA adapters to separate repo
            lora_repo = f"{HF_REPO_NAME}-lora"
            print(f"\nPushing LoRA adapters to: {lora_repo}")
            model.push_to_hub(lora_repo, token=HF_TOKEN)
            tokenizer.push_to_hub(lora_repo, token=HF_TOKEN)
            print("✓ LoRA adapters pushed to HuggingFace Hub!")
            
        except Exception as e:
            print(f"❌ Error uploading to Hub: {e}")
            print("Model is still saved locally in ./local_model_merged and ./local_model_lora")
    else:
        print("⚠️  HF_TOKEN not found. Skipping upload.")
        print("Models saved locally only.")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)