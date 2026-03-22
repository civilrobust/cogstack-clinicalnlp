# ClinicalNLP-KCH

**Clinical NLP pipeline for SNOMED-CT concept extraction from NHS discharge summaries**  
*Built to demonstrate production-readiness for deployment within NHS Secure Data Environments*

---

## Overview

This project replicates and extends the CogStack/MedCAT clinical NLP pipeline, combining:

- **MedCAT 2.6.0** — rule-based SNOMED-CT concept extraction (baseline)
- **Qwen2.5-3B fine-tuned with Unsloth QLoRA** — LLM-based extraction for complex multi-word terms
- **R Markdown validation report** — F1, precision, recall, Cohen's Kappa analysis

Built on local hardware: NVIDIA RTX 5080 (16GB), WSL2, CUDA 12.8

---

## Background

This project was built by someone with 25 years NHS IT experience (Active Directory, DNS, DHCP, 
domain infrastructure, Microsoft stack) now transitioning into AI Engineering. The NHS infrastructure 
knowledge directly informs the deployment context — CogStack runs inside hospital networks and 
Secure Data Environments, not in the cloud.

---

## Project Structure

\`\`\`
cogstack-project/
├── build_cdb.py                  # Build MedCAT SNOMED concept database
├── generate_summaries.py         # Generate synthetic NHS discharge summaries
├── evaluate_medcat.py            # Baseline evaluation (F1, precision, recall)
├── finetune.py                   # Unsloth QLoRA fine-tuning script
├── inference.py                  # Fine-tuned model inference
├── data/
│   ├── discharge_summaries.jsonl # 8 labelled NHS-style discharge summaries
│   └── baseline_results.json    # MedCAT baseline scores
└── outputs/
    └── clinical-snomed-adapter/  # QLoRA adapter weights
\`\`\`

---

## Results

### MedCAT Baseline

| Metric    | Score |
|-----------|-------|
| Precision | 0.923 |
| Recall    | 0.818 |
| F1        | 0.867 |

### Key Finding

MedCAT (rule-based) consistently missed **multi-word clinical terms**:

| Missed Term                  | SNOMED      | Reason |
|------------------------------|-------------|--------|
| acute myocardial infarction  | 57054005    | 3-word span |
| type 2 diabetes              | 44054006    | 3-word span |
| atrial fibrillation          | 49436004    | 2-word span |
| heart failure                | 84114007    | 2-word span |

### Fine-tuned LLM Output

The QLoRA fine-tuned Qwen2.5-3B model trained in **28 seconds** on 8 examples:

- Loss: 2.036 → 1.049
- Successfully extracts multi-word terms MedCAT misses
- Outputs structured JSON with SNOMED codes
- SNOMED code accuracy improves with more training data

### Architecture Conclusion

| Approach | Precision | Multi-word Recall | Code Accuracy |
|----------|-----------|-------------------|---------------|
| MedCAT alone | High | Low | Exact |
| LLM alone | Medium | High | Approximate |
| MedCAT + LLM (hybrid) | High | High | Exact |

**The hybrid approach is the production architecture** — which is exactly what CogStack deploys at King's College Hospital.

---

## Setup

\`\`\`bash
# MedCAT environment
python3 -m venv medcat-env
source medcat-env/bin/activate
pip install medcat==2.6.0 torch transformers

# Unsloth environment  
source unsloth_studio/bin/activate  # Unsloth 2026.3.8

# Build clinical CDB
python3 build_cdb.py

# Generate synthetic data
python3 generate_summaries.py

# Run baseline evaluation
python3 evaluate_medcat.py

# Fine-tune
python3 finetune.py

# Run inference
python3 inference.py
\`\`\`

---

## Clinical Domains Covered

- Cardiology (AMI, AF, heart failure)
- Respiratory (COPD, asthma)
- Diabetes (T2DM, metformin)
- Neurology (stroke, dementia)
- Mental health (depression)
- Medications (aspirin, warfarin, atorvastatin)

---

## Relation to CogStack

This project independently replicates the core CogStack/MedCAT pipeline:

| CogStack Production | This Project |
|---------------------|--------------|
| MedCAT NER + linking | MedCAT 2.6.0 CDB |
| SNOMED-CT ontology | 19 SNOMED concepts |
| LLM fine-tuning | Unsloth QLoRA Qwen2.5-3B |
| NHS discharge summaries | 8 synthetic KCH-style summaries |
| F1 evaluation | precision/recall/F1 baseline |

---

## Previous Work

This builds on an earlier NHS HR RAG system (`nhs-hr-rag`) using ChromaDB + LLaMA 3.1 8B,
demonstrating progression from general RAG to domain-specific clinical NLP.

---

## Author

25 years NHS IT infrastructure → AI Engineering transition  
Targeting clinical NLP roles at CogStack / King's College Hospital / King's Health Partners
