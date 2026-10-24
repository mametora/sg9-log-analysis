from pathlib import Path
from .paths import LOG_DIR, OUTPUT_DIR
import numpy as np
from src.loader import load_log
from src.plots import plt
from matplotlib import font_manager
from matplotlib.patches import Patch
font='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':12})
out=OUTPUT_DIR
out.mkdir(parents=True, exist_ok=True)
d=load_log(LOG_DIR/'romraiderlog_20260912_160818.csv').data
part=d.iloc[d.time_s.diff().idxmax():]
start,end=part.time_s.iloc[[0,-1]]
groups=(part.time_s.diff().fillna(.05)>.2).cumsum()
fig,axes=plt.subplots(4,1,figsize=(14,9),sharex=True)
for ax,col,ylabel,lim in zip(axes,['maf','rpm','throttle','boost_rel'],['MAF (g/s)','回転数 (rpm)','スロットル (%)','相対過給圧 (kPa)'],[(0,225),(0,7000),(0,105),(-85,65)]):
 for _,p in part.groupby(groups):ax.plot((p.time_s-start)/60,p[col],color='#1676ae',lw=.8)
 for i in np.flatnonzero(part.time_s.diff().to_numpy()>1):ax.axvspan((part.time_s.iloc[i-1]-start)/60,(part.time_s.iloc[i]-start)/60,color='#ddd',alpha=.7)
 ax.set(ylabel=ylabel,ylim=lim);ax.grid(alpha=.2)
axes[-1].set(xlabel='経過時間（分）',xlim=(0,(end-start)/60))
fig.suptitle('2026年9月12日　MAFセンサー交換後',fontsize=19,y=.975)
axes[0].legend(handles=[Patch(facecolor='#ddd',alpha=.7,label='記録中断')],loc='upper left',frameon=False,fontsize=10)
fig.tight_layout(rect=(0,0,1,.945))
fig.savefig(out/'MAF交換後_メカニック共有用.png',dpi=200)
plt.close(fig)
print('Saved mechanic-facing PNG.')
