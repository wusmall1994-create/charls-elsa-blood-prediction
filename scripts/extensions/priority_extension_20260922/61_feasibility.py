import os
from pathlib import Path
import pandas as pd,json
R=Path(os.environ["ANALYSIS_ROOT"]);P=R/'data_private';out=[]
for cohort in ['CHARLS','ELSA']:
 for outcome,folder in [('diabe','diabetes'),('hchole','dyslipidemia')]:
  for phase in ['development','validation']:
   if cohort=='CHARLS':
    d=pd.read_csv(P/'multoutcome'/folder/(phase+'.csv.gz'),dtype={'person_id':str});it=pd.read_csv(P/'multoutcome'/folder/(phase+'_intervals.csv.gz'),dtype={'person_id':str});wv='source_wave'
   else:
    w=4 if phase=='development' else 6
    d=pd.read_pickle(P/f'replication_extensions/elsa_w{w}_people.pkl');d=d.loc[d.outcome.eq(outcome)]
    it=pd.read_pickle(P/f'replication_extensions/elsa_w{w}_intervals.pkl');it=it.loc[it.outcome.eq(outcome)];wv='wave'
   if len(out)==0:print('CHARLS fields',d.columns.tolist(),'intervals',it.columns.tolist())
   if cohort=='ELSA' and outcome=='diabe' and phase=='validation':print('ELSA fields',d.columns.tolist(),'intervals',it.columns.tolist())
   fast=d.fasting_sample.eq(1);obs=fast&d.glucose.notna()&d.hba1c.notna()
   lip=d[['total_cholesterol','hdl_cholesterol','triglycerides']].notna().all(axis=1)&d.fasting_sample.notna()
   norm=obs&d.glucose.lt(126)&d.hba1c.lt(6.5)
   strict=obs&d.glucose.lt(100)&d.hba1c.lt(5.7)
   ln=lip&d.total_cholesterol.lt(190)&d.hdl_cholesterol.gt(40)&d.triglycerides.lt(d.fasting_sample.map({1:150,0:175,2:175}))
   row=dict(cohort=cohort,outcome=outcome,phase=phase,n=len(d),events=int(d.event.sum()),waves=sorted(it[wv].unique().tolist()))
   for key,mask in [('measured_glycemia',obs),('below_diabetes',norm),('normal_glycemia',strict),('measured_lipids',lip),('normal_lipids',ln)]:row[key]={'n':int(mask.sum()),'events':int(d.loc[mask,'event'].sum())}
   first=it[wv].min();lab='label' if cohort=='ELSA' else 'event_this_interval';eligible=it.loc[it[wv].eq(first)&it[lab].eq(0),'person_id'];later=it.loc[it[wv].gt(first)&it.person_id.isin(eligible)]
   row['landmark']={'n':int(later.person_id.nunique()),'events':int(later[lab].sum())};out.append(row)
print(json.dumps(out,indent=2));Path(__file__).with_name('priority_feasibility.json').write_text(json.dumps(out,indent=2))
