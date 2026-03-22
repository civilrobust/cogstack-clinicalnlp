import json
from medcat.cat import CAT
from medcat.cdb import CDB
from medcat.config import Config
from medcat.vocab import Vocab

# Load model
cdb = CDB.load('/home/civil/clinical-cdb')
config = Config()
vocab = Vocab()
cat = CAT(cdb=cdb, config=config, vocab=vocab)

# Load summaries
summaries = []
with open('data/discharge_summaries.jsonl') as f:
    for line in f:
        summaries.append(json.loads(line))

# Evaluate
results = []
total_true = 0
total_predicted = 0
true_positives = 0

print("=" * 60)
print("MedCAT Baseline Evaluation")
print("=" * 60)

for summary in summaries:
    text = summary['text']
    ground_truth = {e['snomed'] for e in summary['entities']}
    
    entities = cat.get_entities(text)
    predicted = {e['cui'] for e in entities['entities'].values()}
    
    tp = len(ground_truth & predicted)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    total_true      += len(ground_truth)
    total_predicted += len(predicted)
    true_positives  += tp
    
    print(f"\n{summary['id']}: {text[:60]}...")
    print(f"  Ground truth : {ground_truth}")
    print(f"  Predicted    : {predicted}")
    print(f"  TP={tp} FP={fp} FN={fn} | P={precision:.2f} R={recall:.2f} F1={f1:.2f}")
    
    # Log missed entities
    missed = ground_truth - predicted
    if missed:
        missed_terms = [e['term'] for e in summary['entities'] if e['snomed'] in missed]
        print(f"  MISSED: {missed_terms}")

    results.append({
        'id': summary['id'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'missed': list(ground_truth - predicted)
    })

# Overall scores
overall_precision = true_positives / total_predicted if total_predicted > 0 else 0
overall_recall    = true_positives / total_true if total_true > 0 else 0
overall_f1        = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

print("\n" + "=" * 60)
print("OVERALL BASELINE SCORES")
print("=" * 60)
print(f"  Precision : {overall_precision:.3f}")
print(f"  Recall    : {overall_recall:.3f}")
print(f"  F1        : {overall_f1:.3f}")
print(f"  Total ground truth entities : {total_true}")
print(f"  Total predicted             : {total_predicted}")
print(f"  True positives              : {true_positives}")

# Save results for R analysis later
with open('data/baseline_results.json', 'w') as f:
    json.dump({
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'per_summary': results
    }, f, indent=2)

print("\nResults saved to data/baseline_results.json")
print("This is your BEFORE score. Unsloth fine-tuning is the AFTER.")
