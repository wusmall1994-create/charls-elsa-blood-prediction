import os
from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(os.environ["CHARLS_DATA_DIR"]);P=Path(os.environ["ANALYSIS_ROOT"]);W=Path(os.environ["ANALYSIS_ROOT"])/"extension_work";W.mkdir(parents=True,exist_ok=True)
out=P/'data_private/revision_competing';out.mkdir(exist_ok=True)
rows=[]
for year in [2013,2015,2018,2020]:
 f=R/'2013/CHARLS2013_Dataset/Exit_Interview.dta' if year==2013 else R/f'{year}/CHARLS{year}'+Path('') if False else None
 if year==2013:f=R/'2013/CHARLS2013_Dataset/Exit_Interview.dta'
 else:f=R/f'{year}/CHARLS{year}r/Sample_Infor.dta'
 d=pd.read_stata(f,convert_categoricals=False);d=d if year==2013 else d.loc[d.died.eq(1)]
 for pid in d.ID.astype(str):rows.append({'person_id':pid,'death_wave_year':year})
known=pd.DataFrame(rows).groupby('person_id').death_wave_year.min().to_frame()
dates=[]
for year,f in [(2013,R/'2013/CHARLS2013_Dataset/Exit_Interview.dta'),(2020,R/'2020/CHARLS2020r/Exit_Module.dta')]:
 d=pd.read_stata(f,columns=['ID','exb001_1','exb001_2'],convert_categoricals=False);d['death_year']=d.exb001_1.where(d.exb001_1.between(1990,2021));d['death_month']=d.exb001_2.where(d.exb001_2.between(1,12));d['person_id']=d.ID.astype(str);d['source']=year;dates.append(d[['person_id','death_year','death_month','source']])
dt=pd.concat(dates).sort_values('source').drop_duplicates('person_id',keep='last').set_index('person_id');known=known.join(dt,how='outer');known.to_pickle(out/'charls_deaths.pkl')
summary=[]
for path in (P/'data_private/multoutcome').glob('*/validation.csv.gz'):
 for cohort in ['development','validation']:
  d=pd.read_csv(path.parent/f'{cohort}.csv.gz',dtype={'person_id':str});m=d.merge(known,left_on='person_id',right_index=True,how='left');base=2011 if cohort=='development' else 2015
  death=m.death_wave_year.gt(base)|m.death_year.ge(base)
  summary.append({'outcome':path.parent.name,'cohort':cohort,'n':len(d),'known_deaths':int(death.sum()),'with_year':int((death&m.death_year.notna()).sum()),'with_year_month':int((death&m.death_year.notna()&m.death_month.notna()).sum()),'year_missing':int((death&m.death_year.isna()).sum()),'death_and_diagnosis':int((death&m.event.eq(1)).sum())})
(W/'competing_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps([r for r in summary if r['outcome'] in ['diabetes','stroke','chronic_lung_disease']],indent=2))
