import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import importlib.util,json,warnings
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold
from sklearn.metrics import roc_auc_score,brier_score_loss
W=Path(__file__).resolve().parent;P=Path(os.environ["ANALYSIS_ROOT"]);D=P/'data_private/multoutcome';O=P/'data_private/revision_competing';Q=W/'competing_results';Q.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('model',P/'scripts/04_discrete_time_models.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
death=pd.read_pickle(O/'charls_deaths.pkl');outcomes=['hypertension','dyslipidemia','diabetes','chronic_lung_disease','heart_disease','stroke','kidney_disease','digestive_disease','arthritis_or_rheumatism']
wave={2013:2,2015:3,2018:4,2020:5};nominal={2:2013,3:2015,4:2018,5:2020}

def build(s,cohort,order):
 people=pd.read_csv(D/s/(cohort+'.csv.gz'),dtype={'person_id':str});intervals=pd.read_csv(D/s/(cohort+'_intervals.csv.gz'),dtype={'person_id':str});base=2011 if cohort=='development' else 2015;records=[];ends=[];repairs=0;ties=0;after=0
 for pid,g in intervals.groupby('person_id',sort=False):
  g=g.sort_values('source_wave');dy=death.death_wave_year.get(pid,np.nan);dw=wave.get(dy,99) if dy>base else 99
  eventrows=g.loc[g.event_this_interval.eq(1)];ew=int(eventrows.source_wave.iloc[0]) if len(eventrows) else 99
  if ew==dw and dw<99:ties+=1
  if ew>dw and ew<99:after+=1
  reportfirst=ew<dw or (ew==dw and order=='report_first')
  if reportfirst and ew<99:
   kept=g.loc[g.source_wave.le(ew)].copy();kept['state']=kept.event_this_interval;state=1;endwave=ew
  elif dw<99:
   kept=g.loc[g.source_wave.lt(dw)&g.event_this_interval.eq(0)].copy();kept['state']=0
   start=float(kept.interval_end_years.max()) if len(kept) else 0.;end=float(nominal[dw]-base)
   if end<=start:end=start+1/12;repairs+=1
   new={c:np.nan for c in g.columns};new.update(person_id=pid,source_wave=dw,interval_number=len(kept)+1,interval_start_years=start,interval_end_years=end,interval_duration_years=end-start,event_this_interval=0,state=2)
   kept=pd.concat([kept,pd.DataFrame([new])],ignore_index=True);state=2;endwave=dw
  else:
   kept=g.copy();kept['state']=0;state=0;endwave=int(g.source_wave.max())
  prior=kept.loc[kept.state.eq(0),'source_wave']
  entry=nominal[int(prior.max())]-base if len(prior) else 0
  records.append(kept);ends.append({'person_id':pid,'state':state,'endwave':endwave,'time':nominal[endwave]-base,'death_interval_entry':entry if state==2 else np.nan,'censored':state==0 and endwave<5})
 return people,pd.concat(records,ignore_index=True),pd.DataFrame(ends),{'cohort':cohort,'n':len(people),'deaths_first':sum(r['state']==2 for r in ends),'same_wave_diagnosis_death':ties,'diagnosis_after_death_flag':after,'nominal_time_repairs':repairs}

def weights(end,kind):
 # Discrete wave-time censoring distribution. Events use G(t-); horizon controls use G(5).
 t=end.time.to_numpy(float);state=end.state.to_numpy(int);c=end.censored.to_numpy(bool)
 if kind=='net':
  c=c|(state==2);t=t.copy();t[state==2]=end.death_interval_entry.to_numpy()[state==2]
 surv=1.;left={};right={}
 for u in sorted(set(t)):
  left[u]=surv;at=(t>=u).sum();surv*=1-((t==u)&c).sum()/at;right[u]=surv
 gh=max(right.get(5.,surv),.01);w=np.zeros(len(t));y=np.zeros(len(t),int)
 event=((state==1) if kind in ['net','diagnosis'] else (state==2) if kind=='death' else (state>0))&(t<=5)
 competing=((state==2) if kind=='diagnosis' else (state==1) if kind=='death' else np.zeros(len(t),bool))&(t<=5)
 for i in np.flatnonzero(event|competing):w[i]=1/max(left[t[i]],.01)
 controls=(t>=5)&~event&~competing&~c;w[controls]=1/gh;y[event]=1
 return y,w,gh

def summarize(end,pr,kind,s,analysis):
 y,w,gh=weights(end,kind);ix=w>0;y=y[ix];w=w[ix];a=pr['reference'][ix];b=pr['blood'][ix]
 ra=roc_auc_score(y,a,sample_weight=w);rb=roc_auc_score(y,b,sample_weight=w);ba=brier_score_loss(y,a,sample_weight=w);bb=brier_score_loss(y,b,sample_weight=w)
 # rank-sorted weighted AUROC for fast paired bootstrap
 def fast(p,weights):
  order=np.argsort(p);pp=p[order];yy=y[order];ww=weights[order];ends=np.r_[np.where(np.diff(pp)!=0)[0],len(pp)-1];pw=np.add.reduceat(ww*yy,np.r_[0,ends[:-1]+1]);nw=np.add.reduceat(ww*(1-yy),np.r_[0,ends[:-1]+1]);return float((pw*(np.cumsum(nw)-nw/2)).sum()/(pw.sum()*nw.sum()))
 assert abs(fast(a,w)-ra)<1e-12
 rng=np.random.default_rng(20260915);draw=[]
 for _ in range(2000):
  counts=rng.multinomial(len(y),np.full(len(y),1/len(y)));ww=w*counts
  if (ww*y).sum()==0 or (ww*(1-y)).sum()==0:continue
  draw.append([fast(b,ww)-fast(a,ww),float(np.average((y-b)**2-(y-a)**2,weights=ww))])
 lo,hi=np.quantile(draw,[.025,.975],axis=0)
 return dict(outcome=s,analysis=analysis,estimand=kind,n_total=len(end),n_weighted=int(ix.sum()),events=int(y.sum()),deaths_first=int(end.state.eq(2).sum()),censor_survival5=gh,reference_auroc=ra,blood_auroc=rb,delta_auroc=rb-ra,low=lo[0],high=hi[0],reference_brier=ba,blood_brier=bb,delta_brier=bb-ba,brier_low=lo[1],brier_high=hi[1])

def main(s,order):
 dev,di,de,da=build(s,'development',order);val,vi,ve,va=build(s,'validation',order);aud=[da,va];results=[];tuning=[];predictions={}
 for family in ['cause_specific','multinomial','composite']:
  rows=di.loc[di.state.ne(2)].copy() if family=='cause_specific' else di.copy();rows['target']=(rows.state>0).astype(int) if family=='composite' else rows.state.astype(int)
  train=M.merge_intervals(dev,rows);grid=M.nominal_grid(val,((0.,3.),(3.,2.)));pred={};deathpred={}
  for block in ['reference','blood']:
   cont=M.BASE_CONTINUOUS+(M.PRIMARY_BLOOD_CONTINUOUS if block=='blood' else []);cat=M.BASE_CATEGORICAL+(M.PRIMARY_BLOOD_CATEGORICAL if block=='blood' else []);features=M.TIME_CONTINUOUS+cont+cat
   pipe=Pipeline([('preprocess',M.preprocessing(M.TIME_CONTINUOUS+cont,cat)),('model',LogisticRegression(penalty='l2',solver='lbfgs',max_iter=3000))]);cv=StratifiedGroupKFold(5,shuffle=True,random_state=20260802)
   search=GridSearchCV(pipe,{'model__C':[.01,.1,1,10,100]},scoring='neg_log_loss',cv=cv,n_jobs=2)
   with warnings.catch_warnings(record=True) as warns:
    warnings.simplefilter('always');search.fit(train[features],train.target,groups=train.person_id)
   conv=[str(x.message) for x in warns if 'converg' in str(x.message).lower()];assert not conv,conv
   prob=search.predict_proba(grid[features]);cl=list(search.classes_);n=len(val)
   if family=='multinomial':
    a=prob[:n,cl.index(1)]+prob[:n,cl.index(0)]*prob[n:,cl.index(1)];dp=prob[:n,cl.index(2)]+prob[:n,cl.index(0)]*prob[n:,cl.index(2)];assert np.all(a+dp<=1+1e-10);deathpred[block]=ve.person_id.map(pd.Series(dp,index=val.person_id)).to_numpy()
   else:a=1-(1-prob[:n,cl.index(1)])*(1-prob[n:,cl.index(1)])
   pred[block]=ve.person_id.map(pd.Series(a,index=val.person_id)).to_numpy();tuning.append(dict(outcome=s,order=order,family=family,block=block,C=search.best_params_['model__C'],cv_loss=-search.best_score_,n_intervals=len(train),warnings=len(warns)))
  kind={'cause_specific':'net','multinomial':'diagnosis','composite':'composite'}[family];results.append(summarize(ve,pred,kind,s,family+'_'+order));predictions[family]=pred
  if family=='multinomial':results.append(summarize(ve,deathpred,'death',s,'multinomial_death_'+order))
  print(s,order,family,results[-1]['delta_auroc'],flush=True)
 results.append(summarize(ve,predictions['cause_specific'],'diagnosis',s,'cause_specific_competing_controls_'+order))
 for family,pr in predictions.items():
  q=ve.copy();q['reference']=pr['reference'];q['blood']=pr['blood'];q.to_pickle(O/f'{s}_{family}_{order}.pkl')
 (Q/f'{s}_{order}.json').write_text(json.dumps({'results':results,'tuning':tuning,'audit':aud},indent=2),encoding='utf-8')
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--outcome',default='all');a.add_argument('--order',default='report_first',choices=['report_first','death_first']);args=a.parse_args()
 for s in outcomes if args.outcome=='all' else [args.outcome]:
  if args.outcome=='all' and ((Q/f'{s}_{args.order}.json').exists() or s in ['stroke','chronic_lung_disease']):continue
  main(s,args.order)
