"""Locked CHARLS/ELSA common-panel and sensitivity analyses with cached artifacts.

All preprocessing/tuning uses development or training-fold participants only.
Bootstrap uncertainty is conditional on fitted predictions (and response weights).
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import argparse,importlib.util,json,warnings,time
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
ROOT=Path(os.environ["ANALYSIS_ROOT"])
spec=importlib.util.spec_from_file_location('elsa_train',ROOT/'scripts/core/39_train_elsa_nested_models.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T);B=T.B
OUT=ROOT/'data_private/replication_extensions';QA=ROOT/'qa_logs';SEED=20260913
MAP={'diabe':'diabetes','hchole':'dyslipidemia','hibpe':'hypertension','hearte':'heart_disease','arthre':'arthritis_or_rheumatism','stroke':'stroke','lunge':'chronic_lung_disease'}
RENAME={'hypertension':'hibpe','diabetes':'diabe','heart_disease':'hearte','stroke':'stroke','lung_disease':'lunge','arthritis':'arthre','asthma':'asthmae','cancer':'cancre'}
PANELS={'panel7':B.BLOOD,'panel6':[c for c in B.BLOOD if c!='glucose'],'panel8':B.BLOOD+['hba1c'],'fasting7':B.BLOOD}
def write_json(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def features(cohort,block,s,model,panel):
    cont=T.TIME+B.CORE_CONT;cat=[c for c in B.CORE_CAT if c!=s]
    if block=='rich':
        if cohort=='ELSA':cont+=B.RICH_CONT;cat+=B.RICH_CAT
        else:cont+=['cesd10','grip_strength'];cat+=['education','rural_hukou','marital_status','alcohol_last_year','adl_limitation','liver_disease','kidney_disease']
    if model=='blood':cont+=PANELS[panel];cat+=['fasting_sample']
    return cont,cat
def fit_predict(train,it,test,cohort,block,s,model,panel,key,schedule):
    dst=OUT/(key+'.joblib');meta=OUT/(key+'_fit.json')
    cont,cat=features(cohort,block,s,model,panel)
    grid=T.terms(pd.concat([test.assign(start=a,duration=b) for a,b in schedule],ignore_index=True))
    cached=False
    if dst.exists() and meta.exists():
        try:
            est=joblib.load(dst);saved=json.loads(meta.read_text());assert saved['features']==cont+cat and saved['train_ids']==sorted(train.person_id.astype(str).tolist());cached=True
        except Exception as error:
            print('Recover interrupted cache',key,type(error).__name__,flush=True)
            for artifact in [dst,meta]:artifact.replace(artifact.with_suffix(artifact.suffix+'.incomplete'))
    if not cached:
        rows=T.terms(it.loc[it.person_id.isin(train.person_id)].merge(train.drop(columns=['outcome']),on='person_id',validate='many_to_one').reset_index(drop=True))
        splits=[]
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED).split(train,train.event):
            a=set(train.iloc[tr].person_id);b=set(train.iloc[te].person_id);assert not a&b
            splits.append((np.flatnonzero(rows.person_id.isin(a)),np.flatnonzero(rows.person_id.isin(b))))
        search=GridSearchCV(T.pipeline(cont,cat),{'model__C':[.01,.1,1.,10.,100.]},scoring='neg_log_loss',cv=splits,n_jobs=1,error_score='raise')
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            warnings.simplefilter('always');search.fit(rows[cont+cat],rows.label.astype(int))
        assert not any(issubclass(x.category,ConvergenceWarning) for x in caught),key
        est=search.best_estimator_;tmp=dst.with_suffix('.joblib.tmp');joblib.dump(est,tmp,compress=3);tmp.replace(dst)
        write_json(meta,{'features':cont+cat,'C':search.best_params_['model__C'],'inner_log_loss':float(-search.best_score_),'train_ids':sorted(train.person_id.astype(str).tolist()),'train_n':len(train),'nonconvergence':False})
    h=est.predict_proba(grid[cont+cat])[:,1]
    risk=1-pd.DataFrame({'person_id':grid.person_id,'s':1-h}).groupby('person_id').s.prod()
    return test.person_id.map(risk).to_numpy()
def fast_metric(y,p):
    order=np.argsort(p,kind='stable');ys=y[order];ps=p[order];ends=np.r_[np.flatnonzero(np.diff(ps)),len(ps)-1]
    starts=np.r_[0,ends[:-1]+1]
    def f(w):
        ww=w[order];pos=np.add.reduceat(ww*ys,starts);neg=np.add.reduceat(ww*(1-ys),starts)
        auc=np.sum(pos*(np.cumsum(neg)-neg/2))/(pos.sum()*neg.sum())
        rp=pos[::-1];rn=neg[::-1];den=np.cumsum(rp+rn);precision=np.divide(np.cumsum(rp),den,out=np.zeros_like(den),where=den>0);ap=np.sum(rp*precision)/pos.sum()
        brier=np.average((p-y)**2,weights=w)
        return np.array([auc,brier,ap])
    return f
def summarize(pred,cohort,s,block,analysis,extra=None):
    e=pred.loc[pred.evaluable].copy();assert e.person_id.is_unique
    y=e.event.to_numpy(dtype=int);r=e.reference.to_numpy();b=e.blood.to_numpy();w=e.weight.to_numpy() if 'weight' in e else np.ones(len(e))
    assert np.isfinite(np.c_[r,b,w]).all() and np.all(w>0)
    f=fast_metric(y,r);g=fast_metric(y,b);a=f(w);z=g(w)
    assert np.allclose(a,[roc_auc_score(y,r,sample_weight=w),brier_score_loss(y,r,sample_weight=w),average_precision_score(y,r,sample_weight=w)],atol=1e-12)
    rng=np.random.default_rng(SEED);pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0);draw=[]
    for _ in range(2000):
        ii=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)];bw=np.bincount(ii,minlength=len(y))*w
        draw.append(g(bw)-f(bw))
    ci=np.quantile(draw,[.025,.975],axis=0);prev=float(np.average(y,weights=w))
    row={'cohort':cohort,'outcome':s,'block':block,'analysis':analysis,'n':len(e),'events':int(y.sum()),'observed_risk':prev,'effective_n':float(w.sum()**2/np.square(w).sum())}
    for k,j in [('auroc',0),('brier',1),('average_precision',2)]:row.update({f'reference_{k}':float(a[j]),f'blood_{k}':float(z[j]),f'delta_{k}':float(z[j]-a[j]),f'delta_{k}_low':float(ci[0,j]),f'delta_{k}_high':float(ci[1,j])})
    row.update(reference_scaled_brier=float(1-a[1]/(prev*(1-prev))),blood_scaled_brier=float(1-z[1]/(prev*(1-prev))),reference_mean_risk=float(np.average(r,weights=w)),blood_mean_risk=float(np.average(b,weights=w)))
    if 'weight' not in e:
        for name,prob in [('reference',r),('blood',b)]:
            m=T.metrics(y,prob);row[name+'_calibration_intercept']=m['calibration_intercept'];row[name+'_calibration_slope']=m['calibration_slope']
    if extra:row.update(extra)
    return row
def save_result(pred,cohort,s,block,analysis,extra=None):
    key=f'{cohort}_{s}_{block}_{analysis}';target=QA/(key+'.json')
    if target.exists():
        try:return json.loads(target.read_text())
        except json.JSONDecodeError:target.replace(target.with_suffix('.json.incomplete'))
    pred.to_pickle(OUT/(key+'_predictions.pkl'));row=summarize(pred,cohort,s,block,analysis,extra);write_json(target,row)
    print(key,'n',row['n'],'events',row['events'],'delta',round(row['delta_auroc'],5),flush=True);return row
def charls():
    rows=[]
    for s,outcome in MAP.items():
        pth=ROOT/'data_private/multoutcome'/outcome
        def read(name):return pd.read_csv(pth/(name+'.csv.gz'),dtype={'person_id':str})
        train=read('development').rename(columns=RENAME);test=read('validation').rename(columns=RENAME)
        it=read('development_intervals').rename(columns={'interval_start_years':'start','interval_duration_years':'duration','event_this_interval':'label'})
        it=it[['person_id','start','duration','label']];vit=read('validation_intervals');terminal=set(vit.loc[vit.source_wave.eq(5),'person_id'])
        test['evaluable']=test.event.eq(1)|test.person_id.isin(terminal)
        assert not set(train.person_id)&set(test.person_id)
        for block in ['core','rich']:
            pred=test[['person_id','event','evaluable','age']].copy()
            pred['reference']=fit_predict(train,it,test,'CHARLS',block,s,'reference','panel7',f'CHARLS_{s}_{block}_reference',[(0.,3.),(3.,2.)])
            for panel in ['panel7','panel6','panel8']:
                pred['blood']=fit_predict(train,it,test,'CHARLS',block,s,'blood',panel,f'CHARLS_{s}_{block}_{panel}',[(0.,3.),(3.,2.)])
                rows.append(save_result(pred,'CHARLS',s,block,panel,{'development_n':len(train),'development_events':int(train.event.sum())}))
                if panel=='panel7':rows.append(save_result(pred.loc[pred.age.ge(50)],'CHARLS',s,block,'panel7_age50_evaluation'))
    return rows
def elsa_nested():
    people=pd.read_pickle(OUT/'elsa_w6_people.pkl');intervals=pd.read_pickle(OUT/'elsa_w6_intervals.pkl');rows=[]
    for s in MAP:
        allp=people.loc[people.outcome.eq(s)].reset_index(drop=True);it=intervals.loc[intervals.outcome.eq(s)]
        for block in ['core','rich']:
            old=pd.read_pickle(ROOT/f'data_private/elsa/{s}_{block}_oof.pkl');rows.append(save_result(old,'ELSA',s,block,'panel7'))
            for panel in ['panel6','panel8','fasting7']:
                p=allp.loc[allp.fasting_sample.eq(1)&allp.glucose.notna()].reset_index(drop=True) if panel=='fasting7' else allp
                pred=p[['person_id','event','evaluable']].merge(old[['person_id','fold','reference']],on='person_id',validate='one_to_one');pred['blood']=np.nan
                for fold in range(5):
                    train=p.loc[~p.person_id.isin(pred.loc[pred.fold.eq(fold),'person_id'])].reset_index(drop=True);test=p.loc[p.person_id.isin(pred.loc[pred.fold.eq(fold),'person_id'])].reset_index(drop=True)
                    for model in (['reference','blood'] if panel=='fasting7' else ['blood']):
                        v=fit_predict(train,it,test,'ELSA',block,s,model,panel,f'ELSA_{s}_{block}_{panel}_{model}_fold{fold}',[(0.,2.),(2.,2.),(4.,2.)])
                        pred.loc[pred.fold.eq(fold),model]=pred.loc[pred.fold.eq(fold),'person_id'].map(dict(zip(test.person_id,v)))
                rows.append(save_result(pred,'ELSA',s,block,panel,{'followup_train_population_n':len(p)}))
    return rows
def temporal():
    dev=pd.read_pickle(OUT/'elsa_w4_people.pkl');testall=pd.read_pickle(OUT/'elsa_w6_people.pkl');itall=pd.read_pickle(OUT/'elsa_w4_intervals.pkl');union=set(dev.person_id);rows=[]
    for s in MAP:
        train=dev.loc[dev.outcome.eq(s)].reset_index(drop=True);test=testall.loc[testall.outcome.eq(s)&~testall.person_id.isin(union)].reset_index(drop=True);it=itall.loc[itall.outcome.eq(s)]
        assert not set(train.person_id)&set(test.person_id)
        for block in ['core','rich']:
            pred=test[['person_id','event','evaluable']].copy()
            for model in ['reference','blood']:pred[model]=fit_predict(train,it,test,'ELSA',block,s,model,'panel7',f'ELSA_temporal_{s}_{block}_{model}',[(0.,2.),(2.,2.),(4.,2.)])
            rows.append(save_result(pred,'ELSA',s,block,'temporal_panel7',{'development_n':len(train),'excluded_development_union_n':len(union)}))
    return rows
def response():
    cand=pd.read_pickle(OUT/'elsa_w6_candidates.pkl');rows=[];diagnostics=[]
    for s in MAP:
        q=cand.loc[cand.outcome.eq(s)].reset_index(drop=True);q['weight']=1.;q['retained']=True;q['event']=0;q['evaluable']=False
        cont=B.CORE_CONT+B.RICH_CONT+B.BLOOD+['hba1c'];cat=[c for c in B.CORE_CAT if c!=s]+B.RICH_CAT+['fasting_sample','baseline_proxy']
        q['prior_time']=0.
        for wv in [7,8,9]:
            mask=q.retained&q.event.eq(0);risk=q.loc[mask].copy();observed=risk[f'status{wv}'].isin([0,1])&risk[f'time{wv}'].gt(risk.prior_time)
            probability=np.full(len(risk),np.nan)
            for a,b in StratifiedKFold(5,shuffle=True,random_state=SEED+wv).split(risk,observed):
                model=T.pipeline(cont,cat)
                with threadpool_limits(limits=1):model.fit(risk.iloc[a][cont+cat],observed.iloc[a].astype(int))
                probability[b]=model.predict_proba(risk.iloc[b][cont+cat])[:,1]
            prop=np.clip(probability,.05,.99);q.loc[risk.index,'weight']/=prop
            q.loc[risk.index[~observed],'retained']=False
            valid=risk.index[observed];q.loc[valid,'prior_time']=q.loc[valid,f'time{wv}']
            eventids=valid[q.loc[valid,f'status{wv}'].eq(1)];q.loc[eventids,['event','evaluable']]=[1,True]
            if wv==9:q.loc[valid,'evaluable']=True
            diagnostics.append({'outcome':s,'wave':wv,'at_risk':len(risk),'observed':int(observed.sum()),'propensity_min':float(probability.min()),'propensity_max':float(probability.max()),'below_005':int((probability<.05).sum())})
        e=q.loc[q.retained&q.evaluable,['person_id','event','evaluable','weight']].copy();lo,hi=e.weight.quantile([.01,.99]);e.weight=e.weight.clip(lo,hi)
        for block in ['core','rich']:
            old=pd.read_pickle(ROOT/f'data_private/elsa/{s}_{block}_oof.pkl');pred=e.merge(old[['person_id','event','reference','blood']],on=['person_id','event'],validate='one_to_one');assert len(pred)==len(e)
            rows.append(save_result(pred,'ELSA',s,block,'sequential_response_weighted',{'baseline_eligible_n':len(q),'weight_min':float(pred.weight.min()),'weight_max':float(pred.weight.max())}))
            unweighted=pred.drop(columns='weight');rows.append(save_result(unweighted,'ELSA',s,block,'sequential_complete_unweighted'))
    write_json(QA/'elsa_response_diagnostics.json',diagnostics);return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['charls','elsa_nested','temporal','response']);a=ap.parse_args();start=time.time()
    write_json(QA/f'replication_{a.task}_status.json',{'status':'running','started':pd.Timestamp.now().isoformat()})
    results=globals()[a.task]();write_json(QA/f'replication_{a.task}_results.json',results)
    write_json(QA/f'replication_{a.task}_status.json',{'status':'completed','comparisons':len(results),'elapsed_seconds':time.time()-start})
