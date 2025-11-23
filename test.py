#!/usr/bin/env python3
"""
Evaluation script using real samples from the training dataset
Compares model output with expected output from the dataset
"""
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "nareshmlx/code-reviewer-opencv-16k"
DATASET_NAME = "nareshmlx/16k_opencvpr"
MAX_SEQ_LENGTH = 16384
LOAD_IN_4BIT = True
NUM_SAMPLES = 5  # Number of random samples to test

# =============================================================================
# LOAD MODEL
# =============================================================================

print("="*80)
print("LOADING MODEL")
print("="*80)
print(f"Model: {MODEL_NAME}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"4-bit quantization: {LOAD_IN_4BIT}\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
    trust_remote_code=True,
    device_map="auto",
)

# Enable fast inference mode
FastLanguageModel.for_inference(model)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✓ Model loaded successfully!\n")

# =============================================================================
# LOAD DATASET
# =============================================================================

print("="*80)
print("LOADING DATASET")
print("="*80)

dataset = load_dataset(DATASET_NAME, split="train")
print(f"Dataset size: {len(dataset)} examples")

# Filter for reasonable-sized examples (< 8000 tokens)
print("Filtering for reasonably-sized examples...")
filtered_indices = []
for i in range(len(dataset)):
    input_text = dataset[i].get('input', '')
    instruction = dataset[i].get('instruction', '')
    total_len = len(input_text) + len(instruction)
    if total_len < 32000:  # ~8000 tokens
        filtered_indices.append(i)

print(f"Filtered dataset size: {len(filtered_indices)} examples")
print(f"Testing on: {NUM_SAMPLES} random samples\n")

# Sample random examples from filtered dataset
random.seed(42)
if len(filtered_indices) < NUM_SAMPLES:
    sample_indices = filtered_indices
else:
    sample_indices = random.sample(filtered_indices, NUM_SAMPLES)
samples = [dataset[i] for i in sample_indices]

print("✓ Dataset loaded successfully!\n")

# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

def generate_review(instruction, code, max_tokens=1024):
    """Generate code review for given prompt"""
    # Format prompt similar to training data
    if code.strip():
        prompt = f"{instruction}\n\nHere is the code:\n{code}"
    else:
        prompt = instruction
    
    messages = [{"role": "user", "content": prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=0.5,  # Lower for more consistent output
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    return response

# =============================================================================
# EVALUATION
# =============================================================================

def truncate_text(text, max_lines=15, max_chars=2000):
    """Truncate text to max_lines or max_chars for display"""
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated, {len(text) - max_chars} more chars)"
    
    lines = text.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
    return text

def main():
    print("="*80)
    print("RUNNING EVALUATION ON DATASET SAMPLES")
    print("="*80 + "\n")
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}/{len(samples)} (Index: {sample_indices[i-1]})")
        print(f"{'='*80}\n")
        
        # Extract fields
        instruction = sample.get('instruction', '')
        code_input = sample.get('input', '')
        expected_output = sample.get('output', '')
        
        # Display instruction
        print(f"INSTRUCTION:")
        print(f"{'-'*80}")
        print(truncate_text(instruction, max_lines=5))
        print(f"{'-'*80}\n")
        
        # Display code input
        print(f"CODE INPUT:")
        print(f"{'-'*80}")
        # print(truncate_text(code_input, max_lines=20))
        print(code_input)
        print(f"{'-'*80}\n")
        
        # Display expected output
        print(f"EXPECTED OUTPUT (from dataset):")
        print(f"{'-'*80}")
        print(truncate_text(expected_output, max_lines=15))
        print(f"{'-'*80}\n")
        
        # Generate model output
        print(f"MODEL OUTPUT:")
        print(f"{'-'*80}")
        model_output = generate_review(instruction, code_input, max_tokens=1024)
        print(truncate_text(model_output, max_lines=15))
        print(f"{'-'*80}\n")
        
        # Simple comparison
        print(f"COMPARISON:")
        print(f"{'-'*80}")
        
        # Check if key phrases match
        expected_words = set(expected_output.lower().split())
        model_words = set(model_output.lower().split())
        overlap = len(expected_words & model_words)
        total = len(expected_words)
        similarity = (overlap / total * 100) if total > 0 else 0
        
        print(f"Word overlap: {overlap}/{total} ({similarity:.1f}%)")
        
        # Length comparison
        print(f"Expected length: {len(expected_output)} chars")
        print(f"Model length: {len(model_output)} chars")
        
        print(f"{'-'*80}\n")
        
        if i < len(samples):
            choice = input("Press Enter for next sample, 'q' to quit, or 'f' for full output: ").strip().lower()
            if choice == 'q':
                break
            elif choice == 'f':
                print("\n" + "="*80)
                print("FULL OUTPUTS")
                print("="*80 + "\n")
                print("EXPECTED OUTPUT (FULL):")
                print("-"*80)
                print(expected_output)
                print("-"*80 + "\n")
                print("MODEL OUTPUT (FULL):")
                print("-"*80)
                print(model_output)
                print("-"*80 + "\n")
                input("Press Enter to continue...")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tested {len(samples)} samples from dataset")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()