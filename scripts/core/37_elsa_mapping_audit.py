import os
"""Read-only ELSA feasibility audit; aggregate outputs only, no model fitting."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(os.environ["ANALYSIS_ROOT"])
DATA = Path(os.environ["ELSA_DATA_DIR"])
OUT = ROOT / 'qa_logs' / 'elsa_mapping_audit.json'
SUFFIXES = ['diabe','hchole','hibpe','hearte','stroke','arthre','lunge','asthmae']
BLOOD = ['chol','hdl','trig','hscrp','hgb','wbc','fglu','hba1c']

def path(name):
    matches=list(DATA.rglob(name))
    assert len(matches)==1, (name,len(matches))
    return matches[0]

harm=path('gh_elsa_h.dta')
labels=pd.read_stata(harm,iterator=True).variable_labels()
cols=['idauniq']+[f'r{w}{s}' for w in range(4,10) for s in SUFFIXES+['agey'] if f'r{w}{s}' in labels]
h=pd.read_stata(harm,columns=cols,convert_categoricals=False)
assert h.idauniq.is_unique
result={'scope':'aggregate feasibility; no predictions or fitted models', 'files':{},'outcomes':[]}
cohorts={}
for w,name in [(4,'wave_4_nurse_data.dta'),(6,'wave_6_elsa_nurse_data_v2.dta')]:
    f=path(name); lab=pd.read_stata(f,iterator=True).variable_labels()
    b=pd.read_stata(f,convert_categoricals=False)
    assert b.idauniq.is_unique
    result['files'][name]={'n':len(b),'blood':{v:{'label':lab.get(v),'positive_n':int(b[v].gt(0).sum()),'positive_median':float(b.loc[b[v].gt(0),v].median()),'negative_codes':b.loc[b[v].lt(0),v].value_counts().to_dict()} for v in BLOOD},'protocol_fields':{k:v for k,v in lab.items() if any(t in (k+' '+v).lower() for t in ['fastelig','fasteli','blood sample obtained','date of nurse','nurse visit date'])}}
    linked=h.merge(b[['idauniq']+BLOOD],on='idauniq',validate='one_to_one')
    linked=linked.loc[linked[f'r{w}agey'].ge(50)&linked[BLOOD[:-1]].gt(0).any(axis=1)]
    cohorts[w]=linked
    for s in SUFFIXES:
        q=linked.loc[linked[f'r{w}{s}'].eq(0)]
        follow=q[[f'r{k}{s}' for k in range(w+1,w+4)]]
        event=follow.eq(1).any(axis=1)
        valid=follow.isin([0,1]).any(axis=1)
        evaluable=event|q[f'r{w+3}{s}'].eq(0)
        result['outcomes'].append({'baseline_wave':w,'outcome':s,'baseline_negative':len(q),'any_followup':int(valid.sum()),'evaluable':int(evaluable.sum()),'events':int(event.sum())})
dev_union=set()
for s in SUFFIXES:
    d=cohorts[4]; mask=d[f'r4{s}'].eq(0)&d[[f'r{k}{s}' for k in [5,6,7]]].isin([0,1]).any(axis=1)
    dev_union.update(d.loc[mask,'idauniq'])
result['development_union_n']=len(dev_union)
result['nonoverlap_feasibility']=[]
for s in SUFFIXES:
    q=cohorts[6]; q=q.loc[q.r6agey.ge(50)&q['r6'+s].eq(0)&~q.idauniq.isin(dev_union)]
    event=q[[f'r{k}{s}' for k in [7,8,9]]].eq(1).any(axis=1)
    result['nonoverlap_feasibility'].append({'outcome':s,'baseline_negative':len(q),'evaluable':int((event|q['r9'+s].eq(0)).sum()),'events':int(event.sum())})
result['reference_candidates']={k:v for k,v in labels.items() if k.startswith('r6') and any(t in (k+' '+v).lower() for t in ['agey','gender','mstat','shlt','smoke','drink','cesd','mbmi','systo','diasto','gripsum','adla','iadl'])}
result['common_metadata']={k:v for k,v in labels.items() if any(t in k for t in ['ragender','raeduc','radyear','radmonth'])}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['outcomes','development_union_n','nonoverlap_feasibility','reference_candidates','common_metadata']},ensure_ascii=False,indent=2))
