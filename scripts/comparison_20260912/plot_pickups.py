from pathlib import Path
from .paths import LOG_DIR, OUTPUT_DIR
from src.loader import load_log
from src.plots import plt
from matplotlib import font_manager
font='/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc';font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':12})
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
d=load_log(LOG_DIR/'romraiderlog_20260912_160818.csv').data
origin=d.loc[d.time_s.diff().idxmax(),'time_s']
for name,start,title in [('acceleration',2020,'加速区間'),('light_throttle',2220,'小開度の加速区間')]:
 p=d[d.time_s.between(start,start+30)];fig,ax=plt.subplots(figsize=(16,6));r=ax.twinx()
 ax.plot(p.time_s-origin,p.rpm,color='#1f77b4',lw=1);r.plot(p.time_s-origin,p.maf,color='#d62728',lw=1,alpha=.85)
 ax.set(xlabel='交換後区間の先頭からの経過時間（秒）',ylabel='回転数 (rpm)',xlim=(start-origin,start+30-origin))
 r.set_ylabel('MAF (g/s)');ax.tick_params(axis='y',colors='#1f77b4');r.tick_params(axis='y',colors='#d62728');ax.yaxis.label.set_color('#1f77b4');r.yaxis.label.set_color('#d62728');ax.grid(alpha=.2)
 ax.set_title(f'2026年9月12日　MAFセンサー交換後 — {title}（30秒）',pad=14)
 fig.tight_layout();fig.savefig(OUTPUT_DIR/f'MAF交換後_30秒_{name}.png',dpi=220);plt.close(fig)
 print(name,'original',start,start+30,'after',start-origin,start+30-origin,'max gap',p.time_s.diff().max())
p=d[d.time_s.between(2022.505,2032.806)];q=p[p.throttle>=99];print('full throttle',q.time_s.min(),q.time_s.max(),len(q),'rpm',q.rpm.min(),q.rpm.max())
