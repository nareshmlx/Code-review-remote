#!/usr/bin/env python3
"""
Continue OpenCV Code Reviewer Training with Enhanced System Prompt
Loads checkpoint and trains with OpenCV-specific knowledge
"""
import unsloth
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

CHECKPOINT_PATH       = "outputs/checkpoint-1570"  # Your last checkpoint
MODEL_NAME            = "unsloth/qwen2.5-coder-7b-instruct-bnb-4bit"
MAX_SEQ_LENGTH        = 16384  # Increased for more context
DTYPE                 = None
LOAD_IN_4BIT          = True

LORA_R                = 16
LORA_ALPHA            = 16
LORA_DROPOUT          = 0.0
TARGET_MODULES        = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj"
]

PER_DEVICE_BATCH_SIZE = 1
GRAD_ACC_STEPS        = 16
NUM_TRAIN_EPOCHS      = 5      # Train to epoch 5 (3 more epochs)
LEARNING_RATE         = 1.2e-4  # Slightly lower for stability
WEIGHT_DECAY          = 0.01
LR_SCHEDULER_TYPE     = "cosine"
WARMUP_STEPS          = 20
MAX_GRAD_NORM         = 1.0
LOGGING_STEPS         = 10
SAVE_STEPS            = 250
SAVE_TOTAL_LIMIT      = 3

DATASET_NAME          = "harxsan/opencv-pr-reviews"

HF_TOKEN              = os.getenv("HF_TOKEN")
HF_REPO_NAME          = "nareshmlx/code-reviewer-opencv-harxsan-v2"

OUTPUT_DIR            = "outputs_continued"
RANDOM_SEED           = 3407

# Enhanced system prompt with OpenCV-specific knowledge
ENHANCED_SYSTEM_PROMPT = """You are an expert OpenCV code reviewer with deep knowledge of the OpenCV library.

KEY OPENCV API DEFAULTS (Do NOT flag these as issues if unchanged):
- cv::resize() → defaults to INTER_LINEAR interpolation
- cv::imread() → defaults to IMREAD_COLOR (3-channel BGR)
- cv::cvtColor() → no default, type must be specified
- cv::threshold() → no default, type must be specified
- cv::warpAffine()/warpPerspective() → default to INTER_LINEAR
- cv::dilate()/erode() → no default kernel, must be specified

MEMORY SAFETY RULES:
- cv::Mat uses reference counting (shallow copy by default)
- .clone() creates deep copy, .copyTo() copies data
- In-place operations (src == dst) are DANGEROUS unless documented as safe
- Check for proper AutoBuffer usage in performance-critical code
- Verify proper exception handling and RAII patterns

CROSS-PLATFORM ISSUES:
- Check for platform-specific code (#ifdef WIN32, etc.)
- Verify endianness handling for file I/O
- Check for proper SIMD intrinsics usage (SSE, AVX, NEON, RVV)
- Ensure thread safety (cv::parallel_for_, OpenMP, etc.)

REVIEW GUIDELINES:
- Focus: API design, memory safety, cross-platform compatibility, performance, documentation
- Output JSON format: {
    "assessment": "APPROVE|REQUEST_CHANGES|COMMENT",
    "priority": "LOW|MEDIUM|HIGH",
    "focus": ["area1", "area2"],
    "issues": [{"type": "issue_type", "message": "description"}],
    "suggestions": [{"type": "suggestion|question|nitpick", "message": "improvement"}]
  }
- Max 75 words per message
- Be concise and technically accurate
- If a change only makes default parameters explicit → APPROVE or COMMENT (not REQUEST_CHANGES)
"""

# =============================================================================
# DATA PREPROCESSING WITH ENHANCED PROMPT
# =============================================================================

def prepare_dataset(tokenizer):
    print("\n" + "="*70)
    print("LOADING DATASET WITH ENHANCED PROMPTING")
    print("="*70)
    
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Loaded {len(ds)} examples")
    
    from unsloth.chat_templates import standardize_sharegpt
    ds = standardize_sharegpt(ds)
    
    def to_convs(batch):
        convs = []
        for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
            # Use enhanced system prompt instead of original instruction
            # Keep the code review context from input
            user_message = f"{ENHANCED_SYSTEM_PROMPT}\n\n{inp}"
            
            convs.append([
                {"role":"user", "content": user_message},
                {"role":"assistant", "content": out}
            ])
        return {"conversations": convs}
    
    ds = ds.map(to_convs, batched=True, remove_columns=["instruction","input","output"])
    
    def fmt(ex):
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in ex["conversations"]
        ]
        return {"text": texts}
    
    ds = ds.map(fmt, batched=True)
    print(f"✓ Dataset prepared with enhanced prompts: {len(ds)} examples")
    print(f"✓ System prompt length: {len(ENHANCED_SYSTEM_PROMPT)} chars")
    print("="*70 + "\n")
    
    return ds

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("CONTINUING OPENCV CODE REVIEWER TRAINING (ENHANCED)")
    print("="*70 + "\n")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_memory:.2f} GB\n")
    
    # Check checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        if os.path.exists("outputs"):
            print("Available checkpoints:")
            for item in sorted(os.listdir("outputs")):
                if item.startswith("checkpoint-"):
                    print(f"  - outputs/{item}")
        sys.exit(1)
    
    # Load model
    print("="*70)
    print("LOADING MODEL FROM CHECKPOINT")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: "92GB"}
    )
    
    tokenizer = unsloth.chat_templates.get_chat_template(tokenizer, "qwen-2.5")
    
    # Add LoRA
    print("\nAdding LoRA adapters...")
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
    
    # Load checkpoint weights
    print(f"\nLoading checkpoint weights...")
    adapter_path = os.path.join(CHECKPOINT_PATH, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        import safetensors.torch as st
        state_dict = st.load_file(adapter_path)
        model.load_state_dict(state_dict, strict=False)
        print("✓ Checkpoint loaded successfully!")
    else:
        print("⚠️  Starting fresh (checkpoint not found)")
    
    print("="*70 + "\n")
    
    # Prepare dataset with enhanced prompt
    dataset = prepare_dataset(tokenizer)
    
    # Training config
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Strategy: Continue from epoch 2 → epoch {NUM_TRAIN_EPOCHS}")
    print(f"Enhancement: OpenCV-specific system prompt")
    print(f"Batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"Gradient accumulation: {GRAD_ACC_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Context length: {MAX_SEQ_LENGTH}")
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
    
    # Memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    print("="*70)
    print("STARTING CONTINUED TRAINING")
    print("="*70)
    print(f"Reserved memory: {start_gpu_memory} GB")
    print("="*70 + "\n")
    
    # Train
    trainer_stats = trainer.train()
    
    # Stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory: {used_memory} GB")
    print("="*70 + "\n")
    
    # Comprehensive testing
    print("="*70)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*70)
    
    unsloth.FastLanguageModel.for_inference(model)
    
    test_cases = [
        {
            "name": "Test 1: Making default explicit (should APPROVE/COMMENT)",
            "code": """Review this code change:

File: modules/imgproc/src/resize.cpp
Language: cpp
Change: Make default interpolation explicit

Code:
@@ -100,7 +100,7 @@
 cv::Mat src, dst;
-cv::resize(src, dst, cv::Size(100,100));
+cv::resize(src, dst, cv::Size(100,100), INTER_LINEAR);
"""
        },
        {
            "name": "Test 2: Actual behavior change (should REQUEST_CHANGES)",
            "code": """Review this code change:

File: modules/imgproc/src/resize.cpp
Language: cpp
Change: Change interpolation method

Code:
@@ -100,7 +100,7 @@
 cv::Mat src, dst;
-cv::resize(src, dst, cv::Size(100,100));
+cv::resize(src, dst, cv::Size(100,100), INTER_CUBIC);
"""
        },
        {
            "name": "Test 3: Memory safety issue (should REQUEST_CHANGES)",
            "code": """Review this code change:

File: modules/core/src/matrix.cpp
Language: cpp
Change: Optimize memory allocation

Code:
@@ -50,7 +50,7 @@
-Mat img = imread(path);
+Mat* img = new Mat(imread(path));
 // ... use img
-// automatic cleanup
+delete img;
"""
        },
        {
            "name": "Test 4: In-place operation concern (should flag if unsafe)",
            "code": """Review this code change:

File: modules/imgproc/src/filter.cpp
Language: cpp
Change: Add in-place filtering

Code:
@@ -200,7 +200,7 @@
-Mat temp;
-GaussianBlur(src, temp, Size(5,5), 1.0);
-dst = temp;
+GaussianBlur(src, src, Size(5,5), 1.0);
"""
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"🧪 {test['name']}")
        print(f"{'='*70}")
        
        messages = [{"role":"user", "content": f"{ENHANCED_SYSTEM_PROMPT}\n\n{test['code']}"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        
        print("\nModel Response:")
        print("-" * 70)
        
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        print("-" * 70)
    
    print("\n" + "="*70 + "\n")
    
    # Save models
    print("="*70)
    print("SAVING ENHANCED MODEL")
    print("="*70)
    
    print("\nMerging LoRA adapters...")
    merged_model = model.merge_and_unload()
    
    print("\nSaving locally...")
    save_path_merged = "./local_model_v2_merged"
    save_path_lora = "./local_model_v2_lora"
    
    merged_model.save_pretrained(save_path_merged)
    tokenizer.save_pretrained(save_path_merged)
    print(f"✓ Merged model: {save_path_merged}")
    
    model.save_pretrained(save_path_lora)
    tokenizer.save_pretrained(save_path_lora)
    print(f"✓ LoRA adapters: {save_path_lora}")
    
    # Push to HuggingFace
    if HF_TOKEN:
        try:
            print(f"\n📤 Pushing to HuggingFace Hub...")
            print(f"   Merged model → {HF_REPO_NAME}")
            
            merged_model.push_to_hub(
                HF_REPO_NAME,
                token=HF_TOKEN,
                commit_message=f"Enhanced training: {NUM_TRAIN_EPOCHS} epochs with OpenCV-specific knowledge"
            )
            tokenizer.push_to_hub(HF_REPO_NAME, token=HF_TOKEN)
            print("   ✓ Merged model uploaded!")
            
            lora_repo = f"{HF_REPO_NAME}-lora"
            print(f"   LoRA adapters → {lora_repo}")
            model.push_to_hub(lora_repo, token=HF_TOKEN)
            tokenizer.push_to_hub(lora_repo, token=HF_TOKEN)
            print("   ✓ LoRA adapters uploaded!")
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
            print("Models saved locally only.")
    else:
        print("\n⚠️  HF_TOKEN not set - skipping upload")
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nLocal models:")
    print(f"  - {save_path_merged}")
    print(f"  - {save_path_lora}")
    if HF_TOKEN:
        print(f"\nHuggingFace repos:")
        print(f"  - {HF_REPO_NAME}")
        print(f"  - {HF_REPO_NAME}-lora")
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