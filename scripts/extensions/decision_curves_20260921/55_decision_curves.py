import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(os.environ["ANALYSIS_ROOT"])/"extension_work";W.mkdir(parents=True,exist_ok=True);Q=W/'decision_curve_results';Q.mkdir(exist_ok=True)
P=Path(os.environ["ANALYSIS_ROOT"]) / "data_private"
T=np.round(np.arange(.01,.30001,.005),6);odds=T/(1-T);rows=[];summ=[];audit=[]
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
datasets=[]
for c in ['CHARLS','ELSA']:
 for o in ['diabe','hchole']:datasets.append(('common',c,o,P/f'replication_extensions/{c}_{o}_rich_panel7_predictions.pkl'))
for c in ['CHARLS_to_ELSA','ELSA_to_CHARLS']:
 for o in ['diabe','hchole']:datasets.append(('transport',c,o,P/f'revision_competing/{c}_{o}_recalibrated.pkl'))
for family,c,o,path in datasets:
 d=pd.read_pickle(path);d=d.loc[d.evaluable].reset_index(drop=True);y=d.event.to_numpy(int);n=len(d)
 assert d.person_id.is_unique and set(y)=={0,1}
 methods=['unchanged'] if family=='common' else ['unchanged','intercept','intercept_slope']
 keys=[(m,b) for m in methods for b in ['reference','blood']]
 probs=np.stack([d[b if family=='common' else b+'_'+m].to_numpy() for m,b in keys]);assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
 # Binned risks permit exact threshold counts, including ties at the threshold.
 bins=[np.searchsorted(T,p,side='right') for p in probs]
 def scores(weights):
  result=[]
  for ix in bins:
   tp=np.cumsum(np.bincount(ix,weights=weights*y,minlength=len(T)+1)[::-1])[::-1][1:]
   fp=np.cumsum(np.bincount(ix,weights=weights*(1-y),minlength=len(T)+1)[::-1])[::-1][1:]
   result.append((tp-fp*odds)/weights.sum())
  prev=np.dot(weights,y)/weights.sum();return np.array(result),prev-(1-prev)*odds
 point,allpoint=scores(np.ones(n));rng=np.random.default_rng(20260921);boot=[];ba=[]
 for _ in range(2000):
  weights=np.bincount(rng.integers(0,n,n),minlength=n);v,a=scores(weights);boot.append(v);ba.append(a)
 boot=np.array(boot);ba=np.array(ba)
 for j,(m,b) in enumerate(keys):
  direct=np.array([np.mean((probs[j]>=t)*(y-(1-y)*t/(1-t))) for t in T]);assert np.allclose(point[j],direct,atol=1e-14)
  for i,t in enumerate(T):rows.append(dict(family=family,cohort=c,outcome=o,method=m,block=b,n=n,events=int(y.sum()),threshold=float(t),net_benefit=float(point[j,i]),low=float(np.quantile(boot[:,j,i],.025)),high=float(np.quantile(boot[:,j,i],.975)),assess_all=float(allpoint[i])))
 for m in methods:
  ir=keys.index((m,'reference'));ib=keys.index((m,'blood'));ib0=keys.index(('unchanged','blood'))
  for t in [.05,.1,.2]:
   i=int(np.flatnonzero(np.isclose(T,t))[0]);r=dict(family=family,cohort=c,outcome=o,method=m,n=n,events=int(y.sum()),threshold=t,reference=float(point[ir,i]),blood=float(point[ib,i]),assess_all=float(allpoint[i]),assess_none=0.)
   for name,value,bs in [('delta',point[ib,i]-point[ir,i],boot[:,ib,i]-boot[:,ir,i]),('versus_default',point[ib,i]-max(allpoint[i],0),boot[:,ib,i]-np.maximum(ba[:,i],0)),('update_delta',point[ib,i]-point[ib0,i],boot[:,ib,i]-boot[:,ib0,i])]:
    r[name]=float(value);r[name+'_low']=float(np.quantile(bs,.025));r[name+'_high']=float(np.quantile(bs,.975))
   summ.append(r)
 audit.append(dict(family=family,cohort=c,outcome=o,n=n,events=int(y.sum()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),independent_count_check=True))
 print(family,c,o,n,int(y.sum()),flush=True)
pd.DataFrame(rows).to_csv(Q/'decision_curves.csv',index=False)
(Q/'summary.json').write_text(json.dumps(summ,indent=2));(Q/'audit.json').write_text(json.dumps(audit,indent=2))
df=pd.DataFrame(rows)
for family,method,tag in [('common','unchanged','S1'),('transport','unchanged','S2'),('transport','intercept','S3'),('transport','intercept_slope','S4')]:
 fig,axs=plt.subplots(2,2,figsize=(7.2,6.5),layout='constrained')
 for ax,(c,o),letter in zip(axs.flat,[(c,o) for c in (['CHARLS','ELSA'] if family=='common' else ['CHARLS_to_ELSA','ELSA_to_CHARLS']) for o in ['diabe','hchole']],'abcd'):
  sub=df.loc[(df.family==family)&(df.method==method)&(df.cohort==c)&(df.outcome==o)]
  for b,color,ls in [('reference','#315b7b','--'),('blood','#bd642c','-')]:
   z=sub.loc[sub.block==b];ax.plot(z.threshold*100,z.net_benefit*100,color=color,ls=ls,lw=1.7,label='Reference' if b=='reference' else 'With blood')
  ax.plot(z.threshold*100,z.assess_all*100,color='#777777',ls=':',lw=1.1,label='Assess all');ax.axhline(0,color='#222222',lw=.8,label='Assess none')
  title=c.replace('_to_',' → ')+' | '+('Diabetes' if o=='diabe' else 'Lipid-related report')
  ax.set_title(letter+'  '+title,fontsize=9,loc='left',pad=18)
  horizon=5 if c in ['CHARLS','ELSA_to_CHARLS'] else 6
  ax.text(0,1.015,f'n = {int(z.n.iloc[0]):,}; events = {int(z.events.iloc[0])}; {horizon}-year schedule',transform=ax.transAxes,fontsize=7)
  ax.set(xlabel='Illustrative threshold probability (%)',ylabel='Net benefit per 100 participants',xlim=(1,30))
  # Keep all strategies visible; no favorable-range cropping.
  ax.grid(axis='y',alpha=.15);ax.legend(fontsize=7,loc='best')
 fig.savefig(Q/f'figure_{tag}.png',dpi=300);fig.savefig(Q/f'figure_{tag}.svg');plt.close(fig)
print(json.dumps([r for r in summ if r['threshold']==.1],indent=2))
