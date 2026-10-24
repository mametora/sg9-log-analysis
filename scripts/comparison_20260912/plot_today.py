from pathlib import Path
from .paths import LOG_DIR, OUTPUT_DIR
import numpy as np
from src.loader import load_logs
from src.plots import plt
from matplotlib import font_manager
font_manager.fontManager.addfont('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
out=OUTPUT_DIR
out.mkdir(parents=True, exist_ok=True)
logs=load_logs(LOG_DIR.glob('*20260912*.csv'))
fig,axes=plt.subplots(4,1,figsize=(14,11))
for l,ax in zip(logs,axes):
 d=l.data; suffix=l.metadata.path.stem[-6:]; after=suffix=='160818';color='#1676ae' if after else '#cf6a16'
 t=(d.time_s-d.time_s.iloc[0])/60
 groups=(d.time_s.diff().fillna(.05)>.2).cumsum()
 for _,part in d.groupby(groups):
  ax.plot((part.time_s-d.time_s.iloc[0])/60,part.maf,color=color,lw=.7)
 for i in np.flatnonzero(d.time_s.diff().to_numpy()>1):
  ax.axvspan(t.iloc[i-1],t.iloc[i],color='#ddd',alpha=.75)
  if t.iloc[i]-t.iloc[i-1]>1:ax.text((t.iloc[i-1]+t.iloc[i])/2,110,'記録なし',ha='center',color='#666')
 ax.set(ylim=(0,225),xlim=(0,t.iloc[-1]),ylabel='MAF (g/s)',xlabel='各ログ開始からの経過時間（分）')
 ax.set_title(f'{"交換後" if after else "交換前"}  {suffix[:2]}:{suffix[2:4]}:{suffix[4:]} 開始  ／  最大 {d.maf.max():.2f} g/s',loc='left',color=color)
 ax.grid(alpha=.18)
 # individual four-channel trace, all raw samples, no smoothing
 f,axs=plt.subplots(4,1,figsize=(15,10),sharex=True)
 for a,c,label,lim in zip(axs,['maf','rpm','throttle','boost_rel'],['MAF (g/s)','回転数 (rpm)','スロットル (%)','相対過給圧 (kPa)'],[(0,225),(0,7000),(0,105),(-85,65)]):
  for _,part in d.groupby(groups):a.plot((part.time_s-d.time_s.iloc[0])/60,part[c],lw=.65,color=color)
  for i in np.flatnonzero(d.time_s.diff().to_numpy()>1):a.axvspan(t.iloc[i-1],t.iloc[i],color='#ddd',alpha=.75)
  a.set(ylabel=label,ylim=lim);a.grid(alpha=.2)
 axs[-1].set(xlabel='各ログ開始からの経過時間（分）',xlim=(0,t.iloc[-1]))
 f.suptitle(f'2026/09/12  MAF {"交換後" if after else "交換前"} ／ {suffix}  ｜ ECUリセットなし')
 f.tight_layout(rect=(0,0,1,.96));f.savefig(out/f'trace_{suffix}.png',dpi=160);plt.close(f)
fig.suptitle('2026/09/12  MAFセンサー交換前・交換後',fontsize=19,y=.993)
fig.text(.5,.955,'縦軸は全段共通・横軸の長さはログごとに異なります。走行条件が違うため、ピーク値だけで改善は判断できません。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,0,1,.943));fig.savefig(out/'maf_today_timeseries.png',dpi=160);plt.close(fig)
# Historical analysis notes live under docs/analysis and are not rewritten.
print('Saved overview and 4 context plots; raw samples preserved, recording gaps disconnected.')
