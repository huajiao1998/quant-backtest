#!/usr/bin/env python3
"""止损翻转策略 冠军版 TP=9/10 SL=-10% 反手"""

import sys, json, warnings, sqlite3
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

BDB = 'data/SQLite/backtest_data.db'
CFG = json.load(open('scripts/backtest/reversal_config.json'))
ST = CFG['signal_thresholds']; SW = CFG['signal_weights']

LA = 500; INIT = 3; MAX_ADD = 0; LEV = 5; CM = 0.1; MH = 500; MP = 1
SL_N = -10.0; SL_AFTER_TP1 = -10.0; SL_TIGHT = -10.0
TP1_N = 9.0; TP2_N = 10.0; TP1_W = 11.0; TP2_W = 12.0

conn = sqlite3.connect(BDB)
d = pd.read_sql('SELECT * FROM ohlcv_1h ORDER BY ts', conn); conn.close()
for c in 'open,high,low,close,vol'.split(','): d[c]=pd.to_numeric(d[c])
d['cp']=d['close']; d['hp']=d['high']; d['lp']=d['low']
d['ma']=d['close'].rolling(10).mean()
d['e20']=d['close'].ewm(span=20).mean(); d['e50']=d['close'].ewm(span=50).mean()
e12=d['close'].ewm(span=12).mean(); e26=d['close'].ewm(span=26).mean()
d['macd']=e12-e26; d['ms']=d['macd'].ewm(span=9).mean()
dl=d['close'].diff(); g=dl.clip(lower=0); l_dn=-dl.clip(upper=0)
d['rsi']=100-(100/(1+g.ewm(span=14).mean()/l_dn.ewm(span=14).mean().replace(0,np.nan)))
bm=d['close'].rolling(20).mean(); bs_=d['close'].rolling(20).std()
d['bu']=bm+2*bs_; d['bl']=bm-2*bs_
d['tr']=np.maximum(d['hp']-d['lp'],np.maximum(abs(d['hp']-d['cp'].shift(1)),abs(d['lp']-d['cp'].shift(1))))
d['atr']=d['tr'].ewm(span=14).mean(); d['atr_ma']=d['atr'].rolling(50).mean()
df=d.iloc[70:].reset_index(drop=True); N=len(df)
pr=df['cp'].values; ma=df['ma'].values; ts_arr=df['ts'].values
rsi=df['rsi'].values; atr=df['atr'].values; atr_ma=df['atr_ma'].values
crash=(rsi<30)&(atr>atr_ma*1.5)&(pr<ma); boom=(rsi>70)&(atr>atr_ma*1.5)&(pr>ma)

theta=np.zeros(N); psi=np.zeros(N)
for i in range(1,N):
    l=df.iloc[i]; p=df.iloc[i-1]; bb=0; ss=0; c=l['cp']
    if c is None or pd.isna(c): continue
    e20=l['e20']; e50=l['e50']
    if not pd.isna(e20) and not pd.isna(e50):
        if e20>e50 and c>e20: bb+=SW['ma_trend']
        elif e20<e50 and c<e20: ss+=SW['ma_trend']
    macd=l['macd']; ms=l['ms']; pm=p['macd']; pms=p['ms']
    if not any(pd.isna(x) for x in [macd,ms,pm,pms]):
        if macd>ms and pm<=pms: bb+=SW['macd']
        elif macd<ms and pm>=pms: ss+=SW['macd']
        elif macd>ms: bb+=SW['macd']*0.5
        elif macd<ms: ss+=SW['macd']*0.5
    r=l['rsi']
    if not pd.isna(r):
        if r<ST['rsi_oversold']: bb+=SW['rsi_extreme']
        if r>ST['rsi_overbought']: ss+=SW['rsi_extreme']
    if c>l['bu']: ss+=SW['bollinger']
    if c<l['bl']: bb+=SW['bollinger']
    theta[i]=bb; psi[i]=ss

def sim(sd, idx):
    ep=pr[idx]; ct=INIT; ec=ep; tot=0.0
    t1t=False; t2t=False; er='timeout'; exp=None
    tight=False; tp1_hit=False
    for off in range(1, min(LA, len(pr)-idx)):
        cp=pr[idx+off]
        if np.isnan(cp): break
        if MH>0 and off>=MH:
            er='to'; exp=cp
            tot+=(cp-ec)*ct*CM if sd=='l' else (ec-cp)*ct*CM; ct=0; break
        cur=(cp-ec)/ec*LEV*100 if sd=='l' else (ec-cp)/ec*LEV*100
        bar=idx+off; _c=bar<N and crash[bar]; _b=bar<N and boom[bar]
        if not tight and ((sd=='l' and _c) or (sd=='s' and _b)): tight=True
        u1=TP1_W if ((sd=='s' and _c) or (sd=='l' and _b)) else TP1_N
        u2=TP2_W if ((sd=='s' and _c) or (sd=='l' and _b)) else TP2_N
        if not t2t and cur>=u2:
            if not t1t:
                t1t=True; sz=max(1,int(ct*0.5))
                tot+=(cp-ec)*sz*CM if sd=='l' else (ec-cp)*sz*CM; ct-=sz; tp1_hit=True
            t2t=True
            tot+=(cp-ec)*ct*CM if sd=='l' else (ec-cp)*ct*CM; ct=0; er='tp2'; exp=cp; break
        if not t1t and cur>=u1:
            t1t=True; sz=max(1,int(ct*0.5))
            tot+=(cp-ec)*sz*CM if sd=='l' else (ec-cp)*sz*CM; ct-=sz; tp1_hit=True
        us=max(SL_TIGHT if tight else SL_N, SL_AFTER_TP1 if tp1_hit else SL_N)
        if cur<=us: er='sl'; exp=cp; break
    if ct>0:
        if exp is None: exp=pr[min(idx+LA, len(pr)-1)]
        tot+=(exp-ec)*ct*CM if sd=='l' else (ec-exp)*ct*CM
    return {'pnl':round(tot,2),'win':tot>0,'er':er,'ex':idx+off,'ep':ep,'sd':sd}

print('=== 止损翻转（冠军版）===')
print(f'TP1={TP1_N}%→半仓 TP2={TP2_N}%→全平 | SL={SL_N}%→反手\n')

PERIODS = [
    ('2025-01-01','2025-06-01','1~5月 下跌'),
    ('2025-06-01','2025-10-01','6~9月 上涨'),
    ('2025-10-01','2026-03-01','10~2月 震荡偏跌'),
    ('2026-03-01','2026-04-01','3月 筑底反弹'),
    ('2026-04-01','2026-06-13','4~6月 暴跌'),
    ('2025-01-01','2026-06-13','全周期'),
]

for st, en, lb in PERIODS:
    tss=int(pd.Timestamp(st).timestamp()*1000)
    tse=int(pd.Timestamp(en).timestamp()*1000)
    idx=[i for i in range(N) if tss<=ts_arr[i]<tse]
    if len(idx)<10: continue
    pos=None; trades=[]
    for i in idx:
        if i>=N-2: break
        if np.isnan(ma[i]) or ma[i]<=0: continue
        if pos and i>=pos['ex']: pos=None
        if pos is None:
            wl=(theta[i]>psi[i]) and (pr[i]>ma[i]*1.000)
            ws=(psi[i]>theta[i]) and (pr[i]<ma[i]*1.000)
            if wl:
                r=sim('l',i); trades.append(r)
                if r['er']=='sl':
                    r2=sim('s',i); trades.append(r2); pos=r2
                else: pos=r
            elif ws:
                r=sim('s',i); trades.append(r)
                if r['er']=='sl':
                    r2=sim('l',i); trades.append(r2); pos=r2
                else: pos=r
    ttl=sum(p['pnl'] for p in trades)
    if trades:
        rd=pd.DataFrame(trades)
        for tag, sd_filter in [('做多','l'),('做空','s')]:
            sub=rd[rd['sd']==sd_filter]
            if len(sub)==0: continue
            ssl=sub[sub['er']=='sl']['pnl'].sum() if 'sl' in sub['er'].values else 0
            stp2=sub[sub['er']=='tp2']['pnl'].sum() if 'tp2' in sub['er'].values else 0
            print(f'  {tag} {len(sub):>3}笔 {sub["win"].mean()*100:.0f}% {sub["pnl"].sum():>+8.0f}  '
                  f'SL{len(sub[sub["er"]=="sl"]):>2}笔{ssl:>+7.0f}  TP2{len(sub[sub["er"]=="tp2"]):>2}笔{stp2:>+7.0f}')
        sls=rd[rd['er']=='sl']['pnl'].sum() if 'sl' in rd['er'].values else 0
        tp2s=rd[rd['er']=='tp2']['pnl'].sum() if 'tp2' in rd['er'].values else 0
        print(f'  合计 {len(rd):>3}笔 {rd["win"].mean()*100:.0f}% {ttl:>+8.0f}  '
              f'SL{len(rd[rd["er"]=="sl"]):>2}笔{sls:>+7.0f}  TP2{len(rd[rd["er"]=="tp2"]):>2}笔{tp2s:>+7.0f}')
    print(f'  >>> {lb} {ttl:+.0f}\n')
print('=== 完成 ===')
