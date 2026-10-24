from pathlib import Path
from .paths import LOG_DIR, OUTPUT_DIR
import numpy as np
from src.loader import load_log
from src.plots import plt
from matplotlib import font_manager
font='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11})
out=OUTPUT_DIR
out.mkdir(parents=True, exist_ok=True)
d=load_log(LOG_DIR/'romraiderlog_20260912_160818.csv').data
boundary=d.time_s.diff().idxmax()
for phase,label,color,part in [('before','交換前（中断前）','#cf6a16',d.iloc[:boundary]),('after','交換後（中断後）','#1676ae',d.iloc[boundary:])]:
 start,end=part.time_s.iloc[[0,-1]]
 groups=(part.time_s.diff().fillna(.05)>.2).cumsum()
 fig,axes=plt.subplots(4,1,figsize=(14,9),sharex=True)
 for ax,col,ylabel,lim in zip(axes,['maf','rpm','throttle','boost_rel'],['MAF (g/s)','回転数 (rpm)','スロットル (%)','相対過給圧 (kPa)'],[(0,225),(0,7000),(0,105),(-85,65)]):
  for _,p in part.groupby(groups):ax.plot((p.time_s-start)/60,p[col],color=color,lw=.75)
  for i in np.flatnonzero(part.time_s.diff().to_numpy()>1):ax.axvspan((part.time_s.iloc[i-1]-start)/60,(part.time_s.iloc[i]-start)/60,color='#ddd',alpha=.7)
  ax.set(ylabel=ylabel,ylim=lim);ax.grid(alpha=.2)
 axes[-1].set(xlabel='この区間の先頭からの経過時間（分）',xlim=(0,(end-start)/60))
 fig.suptitle(f'2026/09/12  16:08:18開始ログ ｜ MAF {label}',fontsize=17,y=.985)
 fig.text(.5,.941,f'元ログ経過 {start:.3f}～{end:.3f} 秒 ｜ 縦軸は前後共通・平滑化なし ｜ 交換区分はユーザー説明に基づく',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,0,1,.92));fig.savefig(out/f'trace_160818_{phase}.png',dpi=160);plt.close(fig)
 print(phase,'rows',len(part),'start',start,'end',end,'maf max',part.maf.max())
# Historical analysis notes live under docs/analysis and are not rewritten.
