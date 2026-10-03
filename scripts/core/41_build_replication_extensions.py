import os
"""Build W4 temporal and W6 response denominators from original ELSA files."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path(os.environ["ANALYSIS_ROOT"])
spec=importlib.util.spec_from_file_location('elsa_builder',ROOT/'scripts/core/38_build_elsa_checked_cohorts.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
OUT=ROOT/'data_private/replication_extensions';OUT.mkdir(exist_ok=True)
def build(w,h):
    n=B.read('wave_4_nurse_data.dta' if w==4 else 'wave_6_elsa_nurse_data_v2.dta').set_index('idauniq')
    edu=B.read(f'wave_{w}_ifs_derived_variables.dta').set_index('idauniq')
    statuses={}; qa={'wave':w,'date_skips':{},'flow':{},'unit':'mg/dL lipids/glucose; HbA1c percent'}
    for k in range(w,w+4):
        raw=B.read(f'wave_{k}_elsa_data_eul*.dta').set_index('idauniq');assert raw.index.is_unique
        for s in B.FLAGS:statuses[k,s]=B.raw_status(raw,s,k).reindex(h.index)
    hm=h.reindex(n.index);d=pd.DataFrame(index=n.index)
    mapping={'age':('agey',None),'sex':('ragender',[1,2]),'self_rated_health':('shlt',[1,2,3,4,5]),'current_smoker':('smoken',[0,1]),'bmi':('mbmi',None),'systolic_bp':('systo',None),'diastolic_bp':('diasto',None),'cesd8':('cesd',list(range(9))),'grip_strength':('gripsum',None),'adl_count':('adla',list(range(6))),'marital_status':('mstath',[1,2,3,4,5,7,8]),'alcohol_ever':('drink',[0,1]),'baseline_proxy':('proxy',[0,1])}
    for target,(src,allowed) in mapping.items():d[target]=B.clean(hm[src if src.startswith('ra') else f'r{w}{src}'],allowed)
    d['education']=B.clean(edu.edqual.reindex(d.index),list(range(1,8)))
    d['bmi']=d.bmi.where(d.bmi.between(10,80))
    for s in B.FLAGS:d[s]=statuses[w,s].reindex(d.index)
    for src,target in B.BLOOD_MAP.items():d[target]=n[src].where(n[src].gt(0)).astype(float)
    for c in ['total_cholesterol','hdl_cholesterol']:d[c]*=38.67
    d['triglycerides']=(d.triglycerides*88.57).clip(upper=500);d['glucose']*=18.018
    if w==6:d['hba1c']=d.hba1c/10.929+2.15
    d['fasting_sample']=n.fastelig.map({1:1.,2:0.})
    d.loc[~d.fasting_sample.eq(1),'glucose']=np.nan
    nurse=B.month(n.visyear,n.vismon);main=B.month(hm[f'r{w}iwy'],hm[f'r{w}iwm'])
    d['origin_month']=pd.concat([nurse,main],axis=1).max(axis=1)
    bs='bsoutc' if w==4 else 'BSOUTC'
    d=d.loc[d.age.ge(50)&n[bs].isin([1,2])&d[B.BLOOD].notna().any(axis=1)&nurse.notna()&main.notna()].copy();d.index.name='person_id'
    candidates=[];people=[];intervals=[]
    for s in B.OUTCOMES:
        q=d.loc[d[s].eq(0)].copy()
        for k in range(w+1,w+4):
            q[f'status{k}']=statuses[k,s].reindex(q.index)
            q[f'time{k}']=(B.month(h[f'r{k}iwy'],h[f'r{k}iwm']).reindex(q.index)-q.origin_month)/12
        q['outcome']=s;candidates.append(q.reset_index());ps=[];its=[];skips=0
        for pid,r in q.iterrows():
            start=0.;event=0;terminal=False;pr=[]
            for k in range(w+1,w+4):
                st=r[f'status{k}'];t=r[f'time{k}']
                if pd.isna(st):continue
                if pd.isna(t) or t<=start:skips+=1;continue
                pr.append({'person_id':pid,'outcome':s,'start':start,'duration':t-start,'label':int(st),'wave':k});start=t
                if st==1:event=1;break
                if k==w+3:terminal=True
            if pr:
                rr={c:r[c] for c in d.columns};rr.update(person_id=pid,outcome=s,event=event,evaluable=bool(event or terminal),last_time=start)
                ps.append(rr);its.extend(pr)
        p=pd.DataFrame(ps);it=pd.DataFrame(its);assert p.person_id.is_unique and it.duration.gt(0).all()
        qa['date_skips'][s]=skips;qa['flow'][s]={'baseline_n':len(q),'followup_n':len(p),'evaluable_n':int(p.evaluable.sum()),'events':int(p.event.sum())}
        people.append(p);intervals.append(it)
    p=pd.concat(people,ignore_index=True);it=pd.concat(intervals,ignore_index=True)
    if w==6:
        old=pd.read_pickle(ROOT/'data_private/elsa/checked_people.pkl')
        cols=['person_id','outcome','event','evaluable','last_time']+B.CORE_CONT+B.CORE_CAT+B.RICH_CONT+B.RICH_CAT+B.BLOOD+['hba1c']
        pd.testing.assert_frame_equal(p[cols],old[cols],check_dtype=False)
        pd.testing.assert_frame_equal(it,pd.read_pickle(ROOT/'data_private/elsa/checked_intervals.pkl'),check_dtype=False)
        qa['reproduces_frozen_w6']=True
    p.to_pickle(OUT/f'elsa_w{w}_people.pkl');it.to_pickle(OUT/f'elsa_w{w}_intervals.pkl');pd.concat(candidates,ignore_index=True).to_pickle(OUT/f'elsa_w{w}_candidates.pkl')
    return qa
if __name__=='__main__':
    h=B.read('gh_elsa_h.dta').set_index('idauniq')
    qa=[build(w,h) for w in [6,4]]
    (ROOT/'qa_logs/replication_extension_cohorts.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa,indent=2))
