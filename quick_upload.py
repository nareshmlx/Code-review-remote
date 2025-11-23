from unsloth import FastLanguageModel
import os
from dotenv import load_dotenv
# Load from your checkpoint
load_dotenv()  # Load environment variables from .env file

# Load from your checkpoint
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/workspace/opencv_code_review/outputs/checkpoint-1286",
    max_seq_length=16384,
    dtype=None,
    load_in_4bit=True,
)

# Update with your NEW token here
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_NAME = "nareshmlx/code-reviewer-opencv-16k"

print(f"Pushing to {HF_REPO_NAME}...")
if True: model.push_to_hub_merged(HF_REPO_NAME, tokenizer, save_method = "merged_16bit", token = HF_TOKEN)

print("✓ Done!")
