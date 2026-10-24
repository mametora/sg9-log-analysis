from pathlib import Path
from .paths import LOG_DIR, OUTPUT_DIR
import numpy as np
import pandas as pd
from src.loader import load_logs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=OUTPUT_DIR
ROOT.mkdir(parents=True, exist_ok=True)
logs=load_logs(LOG_DIR.glob('*20260912*.csv'))
parts=[]; summary=[]
for l in logs:
 d=l.data.copy(); d['log']=l.metadata.path.stem[-6:];d['phase']='after' if d.log.iloc[0]=='160818' else 'before';parts.append(d)
 dt=d.time_s.diff(); valid=dt.between(.001,.2)
 summary.append(dict(log=d.log.iloc[0],rows=len(d),elapsed_s=l.metadata.duration_s,recorded_s=dt[valid].sum(),iam_min=d.iam.min(),iam_max=d.iam.max(),fbkc_min=d.fbkc.min(),fbkc_negative_pct=100*(d.fbkc<0).mean(),flkc_median=d.flkc.median(),knock_increases=int(((d.knock_sum.diff()>0)&valid).sum())))
a=pd.concat(parts,ignore_index=True)
# Shared operating cells; minimum 20 samples per phase, equal cell weights.
x=a[(a.rpm>=700)&(a.throttle>=3)].copy()
for c,w in [('rpm',200),('throttle',2),('boost_rel',5)]:x[c+'_bin']=np.floor(x[c]/w)
keys=['rpm_bin','throttle_bin','boost_rel_bin'];cols=['maf','af_correction_1','lambda_1','flkc','fbkc']
g=x.groupby(keys+['phase'])[cols].median().unstack('phase');n=x.groupby(keys+['phase']).size().unstack('phase').fillna(0)
g=g.loc[(n.before>=20)&(n.after>=20)];g.to_csv(ROOT/'matched_cells.csv')
print(pd.DataFrame(summary).to_string(index=False));pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
print('shared cells',len(g),'samples',n.loc[g.index].sum().to_dict())
for c in cols:
 diff=g[c]['after']-g[c]['before'];print(c,'before median',g[c]['before'].median(),'after',g[c]['after'].median(),'paired delta',diff.median(),'q25/75',diff.quantile([.25,.75]).tolist())
pct=100*(g.maf.after/g.maf.before-1);print('maf paired percent median',pct.median(),'q25/75',pct.quantile([.25,.75]).tolist())
fig,ax=plt.subplots(1,2,figsize=(11,4.5))
for p,color in [('before','tab:orange'),('after','tab:blue')]:
 ax[0].scatter(g.index.get_level_values('rpm_bin')*200+100,g.maf[p],s=16,alpha=.6,label=p,color=color)
ax[0].set(xlabel='RPM (cell center)',ylabel='MAF median (g/s)',title='Shared RPM / throttle / boost cells');ax[0].legend()
ax[1].scatter(g.maf.before,g.maf.after,s=22,alpha=.6);lim=max(g.maf.before.max(),g.maf.after.max())*1.05;ax[1].plot([0,lim],[0,lim],'k--',lw=1);ax[1].set(xlabel='Before MAF (g/s)',ylabel='After MAF (g/s)',title=f'{len(g)} shared cells; >=20 samples per phase')
fig.tight_layout();fig.savefig(ROOT/'maf_comparison.png',dpi=160)
