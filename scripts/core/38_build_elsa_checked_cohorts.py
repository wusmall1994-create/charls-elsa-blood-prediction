import os
"""Build audited W6 ELSA cohorts without using model performance.

Source files read only. Only aggregate QA is public; all rows remain data_private.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(os.environ["ANALYSIS_ROOT"])
DATA=Path(os.environ["ELSA_DATA_DIR"])
PRIVATE=ROOT/'data_private'/'elsa'; PRIVATE.mkdir(exist_ok=True)
QA=ROOT/'qa_logs'
FLAGS={'diabe':'diabf','hchole':'hcholf','hibpe':'hibpf','hearte':'heartf','stroke':'strokf','arthre':'arthrf','lunge':'lungf','asthmae':'asthmaf','cancre':'cancrf'}
OUTCOMES=list(FLAGS)[:-1]
RAW={'diabe':(['hedimdi'],['hedacdi']), 'hchole':(['hedimch'],['hedacch']), 'hibpe':(['hedimbp'],['hedacbp']), 'stroke':(['hedimst'],['hedacst']), 'hearte':(['hediman','hedimmi','hedimhf','hedimhm','hedimar','hedim85'],['hedacan','hedacmi','hedachf','hedachm','hedacar','hedacot','hedac95']), 'arthre':(['hedibar'],['hedbdar']), 'lunge':(['hediblu'],['hedbdlu']), 'asthmae':(['hedibas'],['hedbdas'])}
BLOOD_MAP={'chol':'total_cholesterol','hdl':'hdl_cholesterol','trig':'triglycerides','hscrp':'crp','hgb':'hemoglobin','wbc':'wbc','fglu':'glucose','hba1c':'hba1c'}
CORE_CONT=['age','bmi','systolic_bp','diastolic_bp']
CORE_CAT=['sex','self_rated_health','current_smoker']+[s for s in FLAGS if s!='hchole']
RICH_CONT=['cesd8','grip_strength','adl_count']
RICH_CAT=['education','marital_status','alcohol_ever']
BLOOD=list(BLOOD_MAP.values())[:-1]

def path(name):
    x=list(DATA.rglob(name)); assert len(x)==1,(name,len(x)); return x[0]
def clean(s,allowed=None):
    x=pd.to_numeric(s,errors='coerce').astype(float)
    return x.where(x.isin(allowed) if allowed is not None else x.ge(0))
def read(name):
    f=path(name); return pd.read_stata(f,convert_categoricals=False)
def month(y,m):
    return (12*y+m-1).where(y.between(2000,2030)&m.between(1,12))

def raw_status(raw,s,w):
    """Current-wave evidence only; no retrospective harmonized disease value."""
    components={'diabe':[('di','di',7)],'hchole':[('ch','ch',9)],'hibpe':[('bp','bp',1)],'stroke':[('st','st',8)],'hearte':[('an','an',2),('mi','mi',3),('hf','hf',4),('hm','hm',5),('ar','ar',6),('85','ot' if w<8 else '95',95)],'arthre':[('ar','ar',3)],'lunge':[('lu','lu',1)],'asthmae':[('as','as',2)],'cancre':[('ca','ca',5)]}[s]
    chronic=s in ['arthre','lunge','asthmae','cancre']; parts=[]
    for new,old,code in components:
        newcol=('hedib' if chronic else 'hedim')+new
        confcol=('hedbd' if chronic else 'hedac')+old
        ffcol=('hedbw' if chronic else 'hedaw')+old
        newval=raw[newcol] if newcol in raw else pd.Series(np.nan,index=raw.index)
        ff=raw[ffcol] if ffcol in raw else pd.Series(np.nan,index=raw.index)
        conf=raw[confcol] if confcol in raw else pd.Series(np.nan,index=raw.index)
        # Wave 9 angina/MI confirmation fields are not released; retain the
        # contemporaneous preloaded history, explicitly documenting this limit.
        positive=newval.eq(1)|conf.eq(1)
        if w==9 and s=='hearte' and old in ['an','mi']:
            positive=positive|ff.eq(code)
        negative=newval.eq(0)&(ff.eq(-1)|conf.eq(2))
        x=pd.Series(np.nan,index=raw.index); x.loc[negative]=0.; x.loc[positive]=1.
        parts.append(x)
    v=pd.concat(parts,axis=1)
    x=pd.Series(np.nan,index=raw.index)
    x.loc[v.eq(0).all(axis=1)]=0.; x.loc[v.eq(1).any(axis=1)]=1.
    return x

def main():
    h=read('gh_elsa_h.dta'); n=read('wave_6_elsa_nurse_data_v2.dta')
    assert h.idauniq.is_unique and n.idauniq.is_unique
    h=h.set_index('idauniq'); n=n.set_index('idauniq')
    education=read('wave_6_ifs_derived_variables.dta').set_index('idauniq')
    assert education.index.is_unique
    qa={'source':'ELSA local original Stata + Harmonized H documentation','performance_examined':False,'restored_future_disputes':{},'raw_checks':[],'flow':{},'outcomes':{},'tests':{}}
    statuses={}
    for w in [6,7,8,9]:
        raw=read(f'wave_{w}_elsa_data_eul*.dta').set_index('idauniq')
        assert raw.index.is_unique
        for s,flag in FLAGS.items():
            x=clean(h[f'r{w}{s}'],[0,1]); f=h[f'r{w}{flag}']
            qa['restored_future_disputes'][f'r{w}{s}']=int(f.eq(6).sum())
            x=x.mask(f.eq(6),1)  # Undo only later-wave rewrites, never current-wave dispute.
            reconstructed=raw_status(raw,s,w).reindex(h.index)
            statuses[w,s]=reconstructed
            if s not in RAW: continue
            a,b=RAW[s]; cols=[v for v in a+b if v in raw]
            positive=raw[cols].eq(1).any(axis=1)
            xx=x.reindex(raw.index)
            conflict=positive & ~xx.eq(1)
            linked=raw.index.isin(h.index)
            qa['raw_checks'].append({'wave':w,'outcome':s,'raw_positive':int(positive.sum()),'harmonized_restoration_disagreement':int(conflict.sum()),'raw_positive_without_harmonized_id':int((positive&~linked).sum()),'positive_disagreement':int((positive&linked&~reconstructed.reindex(raw.index).eq(1)).sum()),'fields':cols,'raw_valid_status':int(reconstructed.notna().sum())})
    d=pd.DataFrame(index=n.index)
    hm=h.reindex(n.index)
    d['age']=clean(hm.r6agey)
    for target,src,allowed in [('sex','ragender',[1,2]),('self_rated_health','r6shlt',[1,2,3,4,5]),('current_smoker','r6smoken',[0,1]),('bmi','r6mbmi',None),('systolic_bp','r6systo',None),('diastolic_bp','r6diasto',None),('cesd8','r6cesd',list(range(9))),('grip_strength','r6gripsum',None),('adl_count','r6adla',list(range(6))),('education','raeducl',[1,2,3]),('marital_status','r6mstat',[1,2,3,4,5,7,8]),('alcohol_ever','r6drink',[0,1])]:
        d[target]=clean(hm[src],allowed)
    d['bmi']=d.bmi.where(d.bmi.between(10,80))
    # Avoid an all-wave first-nonmissing education value being sourced after W6.
    d['education']=clean(education.edqual.reindex(d.index),list(range(1,8)))
    d['marital_status']=clean(hm.r6mstath,[1,2,3,4,5,7,8])
    d['baseline_proxy']=clean(hm.r6proxy,[0,1])
    for s in FLAGS: d[s]=statuses[6,s].reindex(d.index)
    for src,target in BLOOD_MAP.items(): d[target]=n[src].where(n[src].gt(0)).astype(float)
    # Convert molar SI measurements to the units already used in CHARLS.
    for c in ['total_cholesterol','hdl_cholesterol']: d[c]*=38.67
    d['triglycerides']=(d.triglycerides*88.57).clip(upper=500)
    d['glucose']*=18.018
    d['hba1c']=d.hba1c/10.929+2.15
    d['fasting_sample']=n.fastelig.map({1:1.,2:0.})
    fasting_inconsistent=d.glucose.notna() & ~d.fasting_sample.eq(1)
    qa['fasting_glucose_without_verified_fasting']=int(fasting_inconsistent.sum())
    d.loc[fasting_inconsistent,'glucose']=np.nan
    nurse_month=month(n.visyear,n.vismon)
    harmonized_nurse=month(hm.r6iwnrsy,hm.r6iwnrsm)
    qa['nurse_date_disagreement']=int((nurse_month.notna()&harmonized_nurse.notna()&nurse_month.ne(harmonized_nurse)).sum())
    main_month=month(hm.r6iwy,hm.r6iwm)
    d['origin_month']=pd.concat([nurse_month,main_month],axis=1).max(axis=1)
    date_ok=nurse_month.notna()&main_month.notna()
    blood_ok=n.BSOUTC.isin([1,2])&d[BLOOD].notna().any(axis=1)
    qa['flow']={'nurse_rows':len(n),'age50':int(d.age.ge(50).sum()),'age50_blood':int((d.age.ge(50)&blood_ok).sum()),'age50_blood_dates':int((d.age.ge(50)&blood_ok&date_ok).sum()),'nurse_main_gap_months_quantiles':(nurse_month-main_month).quantile([0,.5,.95,1]).to_dict()}
    d=d.loc[d.age.ge(50)&blood_ok&date_ok].copy()
    d.index.name='person_id'
    qa['blood_missing_pct']=(d[BLOOD+['hba1c']].isna().mean()*100).to_dict()
    qa['reference_missing_pct']=(d[CORE_CONT+CORE_CAT+RICH_CONT+RICH_CAT].isna().mean()*100).to_dict()
    all_intervals=[]; all_people=[]
    followmonths={w:month(h[f'r{w}iwy'],h[f'r{w}iwm']) for w in [7,8,9]}
    for s in OUTCOMES:
        q=d.loc[d[s].eq(0)].copy(); rows=[]; people=[]; ambiguous=0; nodate=0
        for pid,r in q.iterrows():
            start=0.; event=0; terminal_negative=False; latest=0.; personrows=[]; proxy_follow=False
            for w in [7,8,9]:
                status=statuses[w,s].get(pid,np.nan)
                if pd.isna(status): continue
                m=followmonths[w].get(pid,np.nan)
                if pd.isna(m): nodate+=1; continue
                t=(m-r.origin_month)/12
                if t<=start: ambiguous+=1; continue
                personrows.append({'person_id':pid,'outcome':s,'start':start,'duration':t-start,'label':int(status),'wave':w})
                proxy_follow=proxy_follow or h.at[pid,f'r{w}proxy']==1
                latest=t; start=t
                if status==1: event=1; break
                if w==9: terminal_negative=True
            if not personrows: continue
            dead=np.nan  # Harmonized H does not ascertain post-W6 deaths.
            p=r.to_dict(); p.update(person_id=pid,outcome=s,event=event,evaluable=bool(event or terminal_negative),last_time=latest,known_dead=dead,follow_proxy=proxy_follow)
            rows.extend(personrows); people.append(p)
        people=pd.DataFrame(people); intervals=pd.DataFrame(rows)
        assert people.person_id.is_unique
        assert intervals.duration.gt(0).all()
        assert intervals.groupby('person_id').label.sum().le(1).all()
        qa['outcomes'][s]={'baseline_negative':len(q),'with_followup':len(people),'events':int(people.event.sum()),'evaluable':int(people.evaluable.sum()),'known_dead':None,'mortality_status':'not ascertained by Harmonized H post-W6','followup_proxy':int(people.follow_proxy.sum()),'nonincreasing_or_prebaseline_interviews_skipped':ambiguous,'valid_status_missing_date':nodate,'negative_terminal_time_quantiles':people.loc[people.evaluable&people.event.eq(0),'last_time'].quantile([0,.5,1]).to_dict()}
        all_people.append(people); all_intervals.append(intervals)
    cohort=pd.concat(all_people,ignore_index=True); intervals=pd.concat(all_intervals,ignore_index=True)
    qa['tests']={'unique_ids':True,'positive_intervals':True,'single_first_event':True,'unit_hba1c_48':bool(abs((48/10.929+2.15)-6.542)<.002),'unit_glucose_7':bool(abs(7*18.018-126.126)<1e-9),'no_model_fitted':True}
    fixture=pd.DataFrame({'hedimdi':[0,0,1,0,0],'hedacdi':[-1,1,2,-8,2],'hedawdi':[-1,7,7,7,7]})
    got=raw_status(fixture,'diabe',6)
    assert got.iloc[[0,1,2,4]].tolist()==[0,1,1,0] and pd.isna(got.iloc[3])
    qa['tests']['raw_unknown_not_negative_and_new_positive_priority']=True
    assert np.isfinite(cohort[CORE_CONT+CORE_CAT+RICH_CONT+RICH_CAT+BLOOD+['hba1c']].fillna(0).to_numpy()).all()
    assert not any(v['nonincreasing_or_prebaseline_interviews_skipped'] or v['valid_status_missing_date'] for v in qa['outcomes'].values())
    assert qa['nurse_date_disagreement']==0
    cohort.to_pickle(PRIVATE/'checked_people.pkl'); intervals.to_pickle(PRIVATE/'checked_intervals.pkl')
    qa['private_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [PRIVATE/'checked_people.pkl',PRIVATE/'checked_intervals.pkl']}
    # Raw discrepancies must be adjudicated before enabling training.
    qa['raw_disagreement_total']=sum(x['positive_disagreement'] for x in qa['raw_checks'])
    qa['training_gate']='pending_raw_discrepancy_review' if qa['raw_disagreement_total'] else 'passed'
    (QA/'elsa_cohort_quality.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(qa,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
