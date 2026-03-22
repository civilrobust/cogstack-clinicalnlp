from medcat.cdb import CDB
from medcat.config import Config
from medcat.preprocessors.cleaners import NameDescriptor

config = Config()
cdb = CDB(config=config)

concepts = [
    ('57054005', ['acute myocardial infarction', 'acute mi', 'heart attack', 'ami']),
    ('44054006', ['diabetes mellitus type 2', 'type 2 diabetes', 't2dm']),
    ('195967001', ['asthma', 'bronchial asthma']),
    ('387458008', ['aspirin', 'acetylsalicylic acid']),
    ('373444002', ['atorvastatin', 'lipitor']),
    ('386013000', ['metformin']),
    ('49436004',  ['atrial fibrillation', 'af', 'afib']),
    ('230690007', ['stroke', 'cerebrovascular accident', 'cva']),
    ('13645005',  ['chronic obstructive pulmonary disease', 'copd']),
    ('59621000',  ['hypertension', 'high blood pressure', 'htn']),
    ('84114007',  ['heart failure', 'cardiac failure']),
    ('73211009',  ['diabetes mellitus', 'diabetes']),
    ('35489007',  ['depression', 'depressive disorder']),
    ('26929004',  ['alzheimers disease', 'alzheimer', 'dementia']),
    ('371068009', ['paracetamol', 'acetaminophen']),
    ('372756006', ['warfarin', 'coumadin']),
    ('108537001', ['morphine']),
    ('372687004', ['amoxicillin', 'amoxil']),
    ('309911002', ['electrocardiogram', 'ecg']),
]

for cui, names in concepts:
    name_dict = {}
    for name in names:
        nd = NameDescriptor(
            tokens=name.split(),
            snames=set([name]),
            raw_name=name,
            is_upper=name.isupper()
        )
        name_dict[name] = nd
    cdb.add_names(cui=cui, names=name_dict, name_status='A')

cdb.save('/home/civil/clinical-cdb')
print(f'CDB saved: {len(list(cdb.cui2info.keys()))} concepts, {len(list(cdb.name2info.keys()))} names')
