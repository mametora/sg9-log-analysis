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
fig,ax=plt.subplots(figsize=(16,6));right=ax.twinx()
for _,p in part.groupby(groups):
 t=(p.time_s-start)/60
 ax.plot(t,p.rpm,color='#1f77b4',lw=.9)
 right.plot(t,p.maf,color='#d62728',lw=.9,alpha=.85)
for i in np.flatnonzero(part.time_s.diff().to_numpy()>1):ax.axvspan((part.time_s.iloc[i-1]-start)/60,(part.time_s.iloc[i]-start)/60,color='#ddd',alpha=.7)
ax.set(xlabel='経過時間（分）',xlim=(0,(end-start)/60),ylim=(0,7000),ylabel='回転数 (rpm)')
right.set(ylim=(0,225),ylabel='MAF (g/s)')
ax.yaxis.label.set_color('#1f77b4');ax.tick_params(axis='y',colors='#1f77b4')
right.yaxis.label.set_color('#d62728');right.tick_params(axis='y',colors='#d62728')
ax.grid(alpha=.2)
ax.set_title('2026年9月12日　MAFセンサー交換後 — RPM・MAF',pad=14)
ax.legend(handles=[Patch(facecolor='#ddd',alpha=.7,label='記録中断')],loc='upper left',frameon=False,fontsize=10)
fig.tight_layout()
fig.savefig(out/'MAF交換後_RPM・MAF_2軸.png',dpi=220)
plt.close(fig)
print(f'rows={len(part)}; original elapsed={start:.3f}..{end:.3f} seconds')
