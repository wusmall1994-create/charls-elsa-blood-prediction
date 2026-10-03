import os
"""Recompute aggregate results and audit saved participant-level folds."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(os.environ["ANALYSIS_ROOT"])
QA=ROOT/'qa_logs'
PRIVATE=ROOT/'data_private'/'elsa'

def main():
    status=json.loads((QA/'elsa_training_status.json').read_text())
    assert status['status']=='completed'
    results=json.loads((QA/'elsa_nested_results.json').read_text())
    tuning=json.loads((QA/'elsa_nested_tuning.json').read_text())
    people=pd.read_pickle(PRIVATE/'checked_people.pkl')
    assert len(results)==len(status['outcomes'])*len(status['blocks'])==14
    assert len(tuning)==140 and not any(t['nonconvergence'] for t in tuning)
    for name,digest in status['cohort_hashes'].items():
        assert hashlib.sha256((PRIVATE/name).read_bytes()).hexdigest()==digest
    checks=[]
    for row in results:
        outcome,block=row['outcome'],row['block']
        p=people.loc[people.outcome.eq(outcome)]
        oof=pd.read_pickle(PRIVATE/f'{outcome}_{block}_oof.pkl')
        manifest=pd.read_pickle(PRIVATE/f'{outcome}_outer_folds.pkl')
        assert oof.person_id.is_unique and len(oof)==len(p)
        assert set(oof.person_id)==set(p.person_id)
        assert oof[['person_id','fold']].equals(manifest[['person_id','fold']])
        assert sorted(oof.fold.unique())==list(range(5))
        assert np.isfinite(oof[['reference','blood']]).all().all()
        e=oof.loc[oof.evaluable]
        assert len(e)==row['n'] and int(e.event.sum())==row['events']
        for model in ['reference','blood']:
            assert abs(roc_auc_score(e.event,e[model])-row[model]['auroc'])<1e-12
            assert abs(brier_score_loss(e.event,e[model])-row[model]['brier'])<1e-12
            for fold in range(5):
                assert (PRIVATE/f'{outcome}_{block}_{model}_fold{fold}.joblib').exists()
        assert abs(row['blood']['auroc']-row['reference']['auroc']-row['delta_auroc'])<1e-12
        checks.append({'outcome':outcome,'block':block,'unique_complete_oof':True,'shared_folds':True,'metrics_recomputed':True})
    audit={'status':'passed','comparisons':len(checks),'saved_fold_models':140,'checks':checks,'scope':'Artifact and metric audit; not an independent clinical adjudication or an estimate of full retraining uncertainty.'}
    (QA/'elsa_training_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
