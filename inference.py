# Pure HuggingFace inference - bypasses Unsloth inference engine entirely
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

BASE_MODEL = "/home/civil/.unsloth/studio/cache/huggingface/hub/models--unsloth--qwen2.5-3b-instruct-unsloth-bnb-4bit/snapshots/21d7790b1332196a5cf65ba72f2b46b659f74ddf"
ADAPTER    = "outputs/clinical-snomed-adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
print("Ready.")

test_cases = [
    "Patient admitted with acute myocardial infarction. Started on aspirin.",
    "Mrs Fatima with type 2 diabetes and atrial fibrillation and heart failure.",
    "Elderly patient with hypertension and COPD. Prescribed warfarin and metformin.",
]

for text in test_cases:
    prompt = f"""<|im_start|>system
You are a clinical NLP system. Extract all clinical concepts from NHS discharge summaries and return their SNOMED-CT codes as JSON.
<|im_end|>
<|im_start|>user
Extract SNOMED-CT concepts from this discharge summary:

{text}
<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    print(f"\nTEXT: {text}")
    print(f"OUTPUT:\n{response.strip()}")
    print("-" * 60)
