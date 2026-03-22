from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import json
import torch

# ── Model config ──────────────────────────────────────────────
MODEL_NAME    = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LEN   = 2048
OUTPUT_DIR    = "outputs/clinical-snomed-adapter"

# ── Load model ────────────────────────────────────────────────
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = torch.bfloat16,
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                   = 16,
    lora_alpha          = 32,
    target_modules      = ["q_proj","k_proj","v_proj","up_proj","down_proj"],
    lora_dropout        = 0.05,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",
)

# ── Format training data ───────────────────────────────────────
def format_example(summary):
    entities_json = json.dumps(summary["entities"], indent=2)
    return {
        "text": f"""<|im_start|>system
You are a clinical NLP system. Extract all clinical concepts from NHS discharge summaries and return their SNOMED-CT codes as JSON.
<|im_end|>
<|im_start|>user
Extract SNOMED-CT concepts from this discharge summary:

{summary["text"]}
<|im_end|>
<|im_start|>assistant
{entities_json}
<|im_end|>"""
    }

print("Loading training data...")
summaries = []
with open("data/discharge_summaries.jsonl") as f:
    for line in f:
        summaries.append(json.loads(line))

formatted = [format_example(s) for s in summaries]
dataset   = Dataset.from_list(formatted)
print(f"Training on {len(dataset)} examples")

# ── Train ──────────────────────────────────────────────────────
print("Starting fine-tuning on RTX 5080...")
trainer = SFTTrainer(
    model      = model,
    tokenizer  = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 10,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps            = 5,
        learning_rate           = 2e-4,
        bf16                    = True,
        logging_steps           = 1,
        save_strategy           = "epoch",
        report_to               = "none",
        dataset_text_field      = "text",
        max_seq_length          = MAX_SEQ_LEN,
    ),
)

trainer_stats = trainer.train()
print(f"Training complete!")
print(f"Final loss: {trainer_stats.training_loss:.4f}")

# ── Save adapter ───────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")
