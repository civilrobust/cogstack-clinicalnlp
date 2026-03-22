import json
import random

# Synthetic NHS discharge summaries with ground truth SNOMED labels
# Generated to mirror King's College Hospital discharge summary style

summaries = [
    {
        "id": "DS001",
        "text": "Mr James Henderson, 67 year old male, admitted via A&E with acute myocardial infarction. ECG confirmed STEMI. Started on aspirin, atorvastatin and metformin for background type 2 diabetes. Hypertension managed with amlodipine. Discharged to cardiology follow-up.",
        "entities": [
            {"term": "acute myocardial infarction", "snomed": "57054005"},
            {"term": "aspirin", "snomed": "387458008"},
            {"term": "atorvastatin", "snomed": "373444002"},
            {"term": "metformin", "snomed": "386013000"},
            {"term": "type 2 diabetes", "snomed": "44054006"},
            {"term": "hypertension", "snomed": "59621000"},
        ]
    },
    {
        "id": "DS002",
        "text": "Mrs Patricia Okafor, 72 year old female, known COPD and heart failure. Admitted with exacerbation of breathlessness. Background of atrial fibrillation on warfarin. Depression managed with sertraline. Chest X-ray and ECG performed.",
        "entities": [
            {"term": "COPD", "snomed": "13645005"},
            {"term": "heart failure", "snomed": "84114007"},
            {"term": "atrial fibrillation", "snomed": "49436004"},
            {"term": "warfarin", "snomed": "372756006"},
            {"term": "depression", "snomed": "35489007"},
            {"term": "electrocardiogram", "snomed": "309911002"},
        ]
    },
    {
        "id": "DS003",
        "text": "Mr Tariq Ahmed, 58 year old male, admitted following stroke. Background of hypertension and diabetes mellitus. Started on aspirin and atorvastatin. Amoxicillin prescribed for concurrent chest infection. Physiotherapy referral made.",
        "entities": [
            {"term": "stroke", "snomed": "230690007"},
            {"term": "hypertension", "snomed": "59621000"},
            {"term": "diabetes mellitus", "snomed": "73211009"},
            {"term": "aspirin", "snomed": "387458008"},
            {"term": "atorvastatin", "snomed": "373444002"},
            {"term": "amoxicillin", "snomed": "372687004"},
        ]
    },
    {
        "id": "DS004",
        "text": "Mrs Edith Clarke, 81 year old female, known Alzheimer's disease and depression. Admitted with fall and confusion. Paracetamol prescribed for pain management. Warfarin held perioperatively. Dementia care team review requested.",
        "entities": [
            {"term": "alzheimers disease", "snomed": "26929004"},
            {"term": "depression", "snomed": "35489007"},
            {"term": "paracetamol", "snomed": "371068009"},
            {"term": "warfarin", "snomed": "372756006"},
            {"term": "dementia", "snomed": "26929004"},
        ]
    },
    {
        "id": "DS005",
        "text": "Mr David Nwosu, 45 year old male, admitted with severe asthma exacerbation. Background of hypertension. Amoxicillin commenced for pneumonia. Morphine given for pain. ECG unremarkable. Discharged with respiratory follow-up.",
        "entities": [
            {"term": "asthma", "snomed": "195967001"},
            {"term": "hypertension", "snomed": "59621000"},
            {"term": "amoxicillin", "snomed": "372687004"},
            {"term": "morphine", "snomed": "108537001"},
            {"term": "electrocardiogram", "snomed": "309911002"},
        ]
    },
    {
        "id": "DS006",
        "text": "Mrs Fatima Al-Hassan, 63 year old female, type 2 diabetes with poor glycaemic control. Admitted with diabetic ketoacidosis. Metformin restarted on discharge. Background atrial fibrillation, warfarin continued. Heart failure clinic follow-up arranged.",
        "entities": [
            {"term": "type 2 diabetes", "snomed": "44054006"},
            {"term": "metformin", "snomed": "386013000"},
            {"term": "atrial fibrillation", "snomed": "49436004"},
            {"term": "warfarin", "snomed": "372756006"},
            {"term": "heart failure", "snomed": "84114007"},
        ]
    },
    {
        "id": "DS007",
        "text": "Mr George Williams, 76 year old male, COPD and depression. Admitted with fall. Paracetamol and morphine for analgesia. Known stroke last year, aspirin continued. Hypertension medications reviewed.",
        "entities": [
            {"term": "COPD", "snomed": "13645005"},
            {"term": "depression", "snomed": "35489007"},
            {"term": "paracetamol", "snomed": "371068009"},
            {"term": "morphine", "snomed": "108537001"},
            {"term": "stroke", "snomed": "230690007"},
            {"term": "aspirin", "snomed": "387458008"},
            {"term": "hypertension", "snomed": "59621000"},
        ]
    },
    {
        "id": "DS008",
        "text": "Mrs Amara Diallo, 55 year old female, admitted with chest pain. Acute MI ruled out. Background diabetes mellitus and hypertension. Atorvastatin commenced. ECG and bloods unremarkable. Discharged with GP follow-up.",
        "entities": [
            {"term": "acute myocardial infarction", "snomed": "57054005"},
            {"term": "diabetes mellitus", "snomed": "73211009"},
            {"term": "hypertension", "snomed": "59621000"},
            {"term": "atorvastatin", "snomed": "373444002"},
            {"term": "electrocardiogram", "snomed": "309911002"},
        ]
    },
]

# Save as JSONL - standard ML training format
with open('data/discharge_summaries.jsonl', 'w') as f:
    for summary in summaries:
        f.write(json.dumps(summary) + '\n')

print(f'Generated {len(summaries)} discharge summaries')
print(f'Total entities: {sum(len(s["entities"]) for s in summaries)}')
print(f'Saved to data/discharge_summaries.jsonl')

# Show a sample
print('\nSample summary:')
print(json.dumps(summaries[0], indent=2))
