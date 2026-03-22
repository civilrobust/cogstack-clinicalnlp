from flask import Flask, request, jsonify, render_template_string
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from medcat.vocab import Vocab
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

app = Flask(__name__)

# ── Load MedCAT ────────────────────────────────────────────────
print("Loading MedCAT...")
cdb = CDB.load('/home/civil/clinical-cdb')
config = Config()
vocab = Vocab()
cat = CAT(cdb=cdb, config=config, vocab=vocab)
print("MedCAT ready!")

# ── Load Fine-tuned LLM ────────────────────────────────────────
print("Loading fine-tuned LLM...")
BASE_MODEL = "/home/civil/.unsloth/studio/cache/huggingface/hub/models--unsloth--qwen2.5-3b-instruct-unsloth-bnb-4bit/snapshots/21d7790b1332196a5cf65ba72f2b46b659f74ddf"
ADAPTER    = "/home/civil/cogstack-project/outputs/clinical-snomed-adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
llm = PeftModel.from_pretrained(llm, ADAPTER)
llm.eval()
print("LLM ready!")

# ── HTML Template ──────────────────────────────────────────────
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>ClinicalNLP-KCH Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f0f4f8; }
        header { background: #003087; color: white; padding: 20px 40px; }
        header h1 { font-size: 22px; font-weight: 600; }
        header p { font-size: 13px; opacity: 0.8; margin-top: 4px; }
        .container { max-width: 1100px; margin: 30px auto; padding: 0 20px; }
        .input-card { background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .input-card h2 { font-size: 16px; color: #003087; margin-bottom: 12px; }
        textarea { width: 100%; height: 120px; border: 1px solid #ddd; border-radius: 8px; padding: 12px; font-size: 14px; resize: vertical; font-family: inherit; }
        .samples { margin: 10px 0; }
        .sample-btn { background: #e8f0fe; color: #003087; border: none; border-radius: 6px; padding: 6px 12px; margin-right: 8px; cursor: pointer; font-size: 12px; }
        .sample-btn:hover { background: #c5d8fd; }
        button.extract { background: #003087; color: white; border: none; border-radius: 8px; padding: 12px 32px; font-size: 15px; cursor: pointer; margin-top: 12px; width: 100%; }
        button.extract:hover { background: #00509e; }
        button.extract:disabled { background: #aaa; }
        .results { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .result-card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .result-card h2 { font-size: 15px; margin-bottom: 6px; }
        .result-card .subtitle { font-size: 12px; color: #666; margin-bottom: 16px; }
        .medcat-title { color: #003087; }
        .llm-title { color: #7b2d8b; }
        .entity { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; }
        .medcat-entity { background: #e8f0fe; }
        .llm-entity { background: #f3e8fd; }
        .entity-term { font-weight: 500; font-size: 14px; color: #1a1a1a; }
        .entity-code { font-size: 12px; color: #555; font-family: monospace; background: rgba(0,0,0,0.08); padding: 2px 8px; border-radius: 4px; }
        .empty { color: #999; font-size: 13px; text-align: center; padding: 20px; }
        .loading { text-align: center; color: #003087; padding: 20px; font-size: 14px; }
        .badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 8px; }
        .badge-medcat { background: #003087; color: white; }
        .badge-llm { background: #7b2d8b; color: white; }
        .stats { font-size: 12px; color: #666; margin-top: 12px; padding-top: 12px; border-top: 1px solid #eee; }
    </style>
</head>
<body>
<header>
    <h1>ClinicalNLP-KCH &nbsp;|&nbsp; SNOMED-CT Concept Extraction</h1>
    <p>MedCAT rule-based vs Unsloth QLoRA fine-tuned LLM &nbsp;•&nbsp; Built on RTX 5080, WSL2, CUDA 12.8</p>
</header>
<div class="container">
    <div class="input-card">
        <h2>NHS Discharge Summary</h2>
        <div class="samples">
            <span style="font-size:12px;color:#666;margin-right:8px;">Examples:</span>
            <button class="sample-btn" onclick="setSample(0)">Cardiology</button>
            <button class="sample-btn" onclick="setSample(1)">Respiratory</button>
            <button class="sample-btn" onclick="setSample(2)">Neurology</button>
            <button class="sample-btn" onclick="setSample(3)">Diabetes</button>
        </div>
        <textarea id="text" placeholder="Paste an NHS discharge summary here..."></textarea>
        <button class="extract" onclick="extract()">Extract SNOMED Concepts</button>
    </div>
    <div class="results">
        <div class="result-card">
            <h2 class="medcat-title">MedCAT <span class="badge badge-medcat">Rule-based</span></h2>
            <div class="subtitle">High precision • Misses multi-word terms</div>
            <div id="medcat-results"><div class="empty">Results will appear here</div></div>
            <div class="stats" id="medcat-stats"></div>
        </div>
        <div class="result-card">
            <h2 class="llm-title">Fine-tuned LLM <span class="badge badge-llm">Unsloth QLoRA</span></h2>
            <div class="subtitle">Captures complex multi-word clinical terms</div>
            <div id="llm-results"><div class="empty">Results will appear here</div></div>
            <div class="stats" id="llm-stats"></div>
        </div>
    </div>
</div>
<script>
const samples = [
    "Patient admitted with acute myocardial infarction. ECG confirmed STEMI. Started on aspirin and atorvastatin. Background hypertension and type 2 diabetes managed with metformin.",
    "Known COPD patient presenting with exacerbation of breathlessness. Background of atrial fibrillation on warfarin. Amoxicillin commenced for chest infection.",
    "Elderly patient admitted following stroke. Background of hypertension and dementia. Paracetamol prescribed for pain. Depression managed with sertraline.",
    "Mrs Fatima, 63, type 2 diabetes with poor control. Admitted with DKA. Metformin restarted. Heart failure clinic follow-up arranged. Warfarin continued for atrial fibrillation."
];
function setSample(i) { document.getElementById("text").value = samples[i]; }
async function extract() {
    const text = document.getElementById("text").value.trim();
    if (!text) return;
    document.getElementById("medcat-results").innerHTML = "<div class='loading'>Extracting...</div>";
    document.getElementById("llm-results").innerHTML = "<div class='loading'>Extracting...</div>";
    document.querySelector(".extract").disabled = true;
    try {
        const r = await fetch("/extract", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text})});
        const data = await r.json();
        renderMedcat(data.medcat);
        renderLLM(data.llm);
    } catch(e) {
        document.getElementById("medcat-results").innerHTML = "<div class='empty'>Error</div>";
        document.getElementById("llm-results").innerHTML = "<div class='empty'>Error</div>";
    }
    document.querySelector(".extract").disabled = false;
}
function renderMedcat(entities) {
    if (!entities.length) { document.getElementById("medcat-results").innerHTML = "<div class='empty'>No entities found</div>"; return; }
    document.getElementById("medcat-results").innerHTML = entities.map(e =>
        `<div class="entity medcat-entity"><span class="entity-term">${e.term}</span><span class="entity-code">${e.snomed}</span></div>`
    ).join("");
    document.getElementById("medcat-stats").innerHTML = `${entities.length} concepts extracted`;
}
function renderLLM(entities) {
    if (!entities.length) { document.getElementById("llm-results").innerHTML = "<div class='empty'>No entities found</div>"; return; }
    document.getElementById("llm-results").innerHTML = entities.map(e =>
        `<div class="entity llm-entity"><span class="entity-term">${e.term}</span><span class="entity-code">${e.snomed}</span></div>`
    ).join("");
    document.getElementById("llm-stats").innerHTML = `${entities.length} concepts extracted`;
}
</script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/extract', methods=['POST'])
def extract():
    text = request.json.get('text', '')

    # MedCAT extraction
    medcat_entities = []
    result = cat.get_entities(text)
    for ent in result['entities'].values():
        medcat_entities.append({
            'term': ent['detected_name'],
            'snomed': ent['cui']
        })

    # LLM extraction
    llm_entities = []
    prompt = f"""<|im_start|>system
You are a clinical NLP system. Extract all clinical concepts from NHS discharge summaries and return their SNOMED-CT codes as JSON.
<|im_end|>
<|im_start|>user
Extract SNOMED-CT concepts from this discharge summary:

{text}
<|im_end|>
<|im_start|>assistant
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        parsed = json.loads(response)
        if isinstance(parsed, list):
            llm_entities = [{'term': e.get('term',''), 'snomed': e.get('snomed','')} for e in parsed]
    except Exception as e:
        print(f"LLM error: {e}")
        llm_entities = [{'term': 'See terminal for details', 'snomed': str(e)[:50]}]

    return jsonify({'medcat': medcat_entities, 'llm': llm_entities})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("ClinicalNLP-KCH Demo running!")
    print("Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
