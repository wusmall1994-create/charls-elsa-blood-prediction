import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
Q=Path(__file__).resolve().parent/'panel_performance_results'
plt.rcParams.update({'font.family':'Arial','font.size':9,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
def save(fig,name):
 fig.savefig(Q/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(Q/(name+'.svg'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,5.2));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,title,body):
 ax.add_patch(Rectangle((x,y),w,h,facecolor='#eef3f6',edgecolor='#577185',lw=.8))
 ax.text(x+w/2,y+h-.025,title,ha='center',va='top',weight='bold',fontsize=9)
 ax.text(x+w/2,y+h-.065,body,ha='center',va='top',fontsize=8,linespacing=1.3)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#577185','lw':1})
box(.03,.81,.94,.17,'Seven shared diagnosis-report outcomes','Common seven-measure panel added to non-laboratory information\nCHARLS: 2011 development / 2015 evaluation; ELSA: wave 6 nested evaluation')
box(.03,.54,.45,.20,'Simplified versus complete panels','Diabetes: glucose; lipid reports: three lipids\nSame participants and paired folds\nAUROC, Brier, calibration and net benefit')
box(.52,.54,.45,.20,'Baseline status and later reports','Require observed biochemical measurements\nRestrict to below-threshold values\nNegative first follow-up → two-year landmark')
arrow(.26,.81,.26,.74);arrow(.74,.81,.74,.74)
box(.03,.29,.94,.18,'Checks on interpretation','Glucose availability strata; repeated fitting; follow-up and mortality sensitivities\nFrozen transfer in both directions; local recalibration\nOriginal 13-outcome CHARLS benchmark provides supporting evidence')
arrow(.26,.54,.26,.47);arrow(.74,.54,.74,.47)
box(.03,.03,.94,.19,'Potential use of existing blood results','Consider further follow-up or repeat assessment\nIllustrative action thresholds, not a validated clinical policy\nSubsequent diagnosis reports do not establish biological disease onset')
arrow(.5,.29,.5,.22);save(fig,'analysis_flow')

fig,ax=plt.subplots(figsize=(7.2,5.0));fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.99);ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
xs=[.26,.59,.88]
for x,txt in zip(xs,['Baseline blood\nand predictors','First negative follow-up\nReset prediction origin','Next scheduled\ndiagnosis report']):ax.text(x,.965,txt,ha='center',va='top',fontsize=8.5,weight='bold')
rows=[(.73,'a  CHARLS\ndevelopment',['2011','2013\nt = 0','2015\nt ≈ 2 years']),(.49,'b  CHARLS\nevaluation',['2015','2018\nt = 0','2020\nt ≈ 2 years']),(.25,'c  ELSA\nnested evaluation',['Wave 6\n2012–2013','Wave 7\n2014–2015\nt = 0','Wave 8\n2016–2017\nt ≈ 2 years'])]
for y,label,dates in rows:
 ax.text(.015,y,label,ha='left',va='center',fontsize=9,weight='bold')
 ax.plot(xs[:2],[y,y],color='#777777',ls='--',lw=1.2)
 ax.annotate('',xy=(xs[2],y),xytext=(xs[1],y),arrowprops={'arrowstyle':'->','lw':2,'color':'#315b7b'})
 ax.scatter(xs,[y]*3,s=[20,35,20],c=['#777777','#315b7b','#315b7b'],zorder=3)
 for x,txt in zip(xs,dates):ax.text(x,y-.025,txt,ha='center',va='top',fontsize=8)
 ax.text((xs[1]+xs[2])/2,y+.035,'Nominal two-year window',ha='center',fontsize=7.5,color='#315b7b')
ax.text(.5,.015,'Schematic spacing. Predictors remain at baseline; actual interview intervals vary.',ha='center',va='bottom',fontsize=8)
save(fig,'landmark_timeline')

df=pd.read_csv(Q/'curves.csv')
# Main three-block curves; other populations are fully tabulated and archived.
for scenario,tag in [('all','S5'),('measured','measured'),('below_threshold','below_threshold'),('landmark','landmark')]:
 fig,axs=plt.subplots(2,2,figsize=(7.2,6.4),layout='constrained')
 for ax,(c,o),letter in zip(axs.flat,[(c,o) for c in ['CHARLS','ELSA'] for o in ['diabe','hchole']],'abcd'):
  sub=df.loc[(df.cohort==c)&(df.outcome==o)&(df.scenario==scenario)]
  for m,color,ls,label in [('reference','#666666','--','Reference'),('simple','#315b7b','-.','Simplified'),('full','#bd642c','-','Complete')]:
   z=sub.loc[sub.model==m];ax.plot(z.threshold*100,z.net_benefit*100,color=color,ls=ls,lw=1.5,label=label)
  ax.plot(z.threshold*100,z.assess_all*100,color='#999999',ls=':',lw=1.1,label='Assess all');ax.axhline(0,color='black',lw=.7,label='Assess none')
  ax.set_title(letter+'  '+c+' | '+('Diabetes' if o=='diabe' else 'Lipid-related report'),fontsize=9,loc='left',pad=18)
  horizon=2 if scenario=='landmark' else (5 if c=='CHARLS' else 6)
  ax.text(0,1.015,f'n = {int(z.n.iloc[0]):,}; events = {int(z.events.iloc[0])}; {horizon}-year schedule',transform=ax.transAxes,fontsize=7)
  ax.set(xlabel='Illustrative threshold probability (%)',ylabel='Net benefit per 100 participants',xlim=(1,30));ax.grid(axis='y',alpha=.15);ax.legend(fontsize=7,loc='best')
 save(fig,'decision_curves_'+tag)
print('Created timeline, revised flow and three-block decision curves')
