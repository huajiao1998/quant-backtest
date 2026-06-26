#!/usr/bin/env python3
"""对冲翻转v7 — 去掉MA过滤，纯信号+翻转（双数据源切换）"""
import json,warnings,sqlite3,uuid
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

DATA_SOURCE='trading'  # 'backtest_data' 或 'trading'
BDB='data/SQLite/backtest_data.db'
TD='data/SQLite/trading.db'
CFG=json.load(open('strategy_config.json'))
SW={'ma_trend':3,'macd':1,'rsi_extreme':1,'bollinger':1}
LEV=5;CM=0.1;TAKER_FEE=0.0005
SIG_SL=-7;SIG_TP=5;SIG_SZ=1;FLIP_SL=-8;FLIP_TP=8;FLIP_SZ=6

if DATA_SOURCE=='trading':
    conn=sqlite3.connect(TD)
    d=pd.read_sql("SELECT timestamp,close_price,high_price,low_price FROM indicators_snapshot ORDER BY id",conn);conn.close()
    d['ts']=d['timestamp'].apply(lambda t:int(pd.Timestamp(t).timestamp()*1000))
    d.rename(columns={'close_price':'close','high_price':'high','low_price':'low'},inplace=True)
    d['open']=d['close'];d['vol']=1.0
else:
    conn=sqlite3.connect(BDB)
    d=pd.read_sql('SELECT * FROM ohlcv_1h ORDER BY ts',conn);conn.close()

for c in 'open,high,low,close,vol'.split(','):d[c]=pd.to_numeric(d[c])
d['cp']=d['close'];d['hp']=d['high'];d['lp']=d['low']
d['ma']=d['close'].rolling(40).mean()
d['e45']=d['close'].ewm(span=45).mean();d['e50']=d['close'].ewm(span=50).mean()
e12=d['close'].ewm(span=12).mean();e26=d['close'].ewm(span=26).mean()
d['macd']=e12-e26;d['ms']=d['macd'].ewm(span=9).mean();d['mh']=d['macd']-d['ms']
dl=d['close'].diff();g=dl.clip(lower=0);l_dn=-dl.clip(upper=0)
d['rsi']=100-(100/(1+g.ewm(span=14).mean()/l_dn.ewm(span=14).mean().replace(0,np.nan)))
bm=d['close'].rolling(40).mean();bs_=d['close'].rolling(40).std()
d['bu']=bm+2*bs_;d['bl']=bm-2*bs_
df=d.iloc[70:].reset_index(drop=True);N=len(df)
pr=df['cp'].values;ma=df['ma'].values;ts_arr=df['ts'].values

theta=np.zeros(N);psi=np.zeros(N)
for i in range(1,N):
    l=df.iloc[i];p=df.iloc[i-1];bb=0;ss=0
    if pd.isna(l['cp']):continue
    e45,l_e50=l['e45'],l['e50']
    if not pd.isna(e45) and not pd.isna(l_e50):
        if e45>l_e50 and l['cp']>e45:bb+=3
        elif e45<l_e50 and l['cp']<e45:ss+=3
    macd,ms=l['macd'],l['ms'];pm=p['macd'];pms=p['ms']
    if not any(pd.isna(x) for x in [macd,ms,pm,pms]):
        if macd>ms and pm<=pms:bb+=1
        elif macd<ms and pm>=pms:ss+=1
        elif macd>ms:bb+=0.5
        elif macd<ms:ss+=0.5
    r=l['rsi']
    if not pd.isna(r):
        if r<30:bb+=1
        if r>70:ss+=1
    if l['cp']>l['bu']:ss+=1
    if l['cp']<l['bl']:bb+=1
    theta[i]=bb;psi[i]=ss

W=9;hp_arr=df['hp'].values;lp_arr=df['lp'].values;mh_arr=df['mh'].values
bearish_div=np.zeros(N);bullish_div=np.zeros(N)
for i in range(W*2,N):
    s1=i-W*2;s1e=i-W;s2=i-W;s2e=i
    pl_s2=lp_arr[s2:s2e+1].min();pl_s1=lp_arr[s1:s1e+1].min()
    if pl_s2<pl_s1:
        si=s2+lp_arr[s2:s2e+1].argmin()
        ml_s2=mh_arr[s2:s2e+1].min();ml_s1=mh_arr[s1:s1e+1].min()
        if ml_s2>ml_s1 and mh_arr[si]>ml_s1:bullish_div[i]=1
    ph_s2=hp_arr[s2:s2e+1].max();ph_s1=hp_arr[s1:s1e+1].max()
    if ph_s2>ph_s1:
        si=s2+hp_arr[s2:s2e+1].argmax()
        mh_s2=mh_arr[s2:s2e+1].max();mh_s1=mh_arr[s1:s1e+1].max()
        if mh_s2<mh_s1 and mh_arr[si]<mh_s1:bearish_div[i]=1

class Position:
    __slots__=('pos_id','dir','sz','entry_p','entry_i','sl','tp','flipped','fee_paid')
    def __init__(self,dir_,sz,i,p,sl,tp,flipped=False):
        self.pos_id=uuid.uuid4().hex[:12];self.dir=dir_;self.sz=sz
        self.entry_p=p;self.entry_i=i;self.sl=sl;self.tp=tp
        self.flipped=flipped;self.fee_paid=0.0
    def cur(self,cp):
        return (cp-self.entry_p)/self.entry_p*LEV*100 if self.dir=='l' else (self.entry_p-cp)/self.entry_p*LEV*100
    def pnl(self,cp):
        return (cp-self.entry_p)*self.sz*CM if self.dir=='l' else (self.entry_p-cp)*self.sz*CM

if DATA_SOURCE=='trading':
    PERIODS=[
        ('2026-03-29','2026-04-19','3.29~4.19 筑底'),
        ('2026-04-19','2026-05-10','4.19~5.10 震荡'),
        ('2026-05-10','2026-06-06','5.10~6.06 下跌'),
        ('2026-06-06','2026-06-23','6.06~6.23 反弹'),
        ('2026-06-23','2026-06-26','6.23~6.26 实盘'),
        ('2026-03-29','2026-06-26','全量实盘期'),
    ]
else:
    PERIODS=[
        ('2025-01-01','2025-06-01','1~5月 下跌'),
        ('2025-06-01','2025-10-01','6~9月 上涨'),
        ('2025-10-01','2026-03-01','10~2月 震荡偏跌'),
        ('2026-03-01','2026-04-01','3月 筑底反弹'),
        ('2026-04-01','2026-06-13','4~6月 暴跌'),
        ('2025-01-01','2026-06-13','全周期'),
    ]

print(f'=== 对冲翻转v7（去MA过滤+双数据源）===\n')

for st,en,lb in PERIODS:
    tss=int(pd.Timestamp(st).timestamp()*1000)
    tse=int(pd.Timestamp(en).timestamp()*1000)
    idx=[i for i in range(N) if tss<=ts_arr[i]<tse]
    if len(idx)<10:continue
    
    trades=[];positions=[]
    for i in idx:
        if i>=N-2:break
        if np.isnan(ma[i]) or ma[i]<=0:continue
        cp=pr[i];new_flips=[]
        alive=[]
        for pos in positions:
            c=pos.cur(cp)
            if c<=pos.sl:
                pnl=pos.pnl(cp);pos.fee_paid+=abs(pnl)*TAKER_FEE
                trades.append({'pnl':round(pnl-pos.fee_paid,2),'dir':pos.dir,'er':'sl','tag':'flip' if pos.flipped else 'sig'})
                fd='s' if pos.dir=='l' else 'l'
                found=False
                for ap in alive:
                    if ap.dir==fd:
                        ap.sz+=FLIP_SZ;found=True;break
                if not found:new_flips.append(fd)
            elif c>=pos.tp:
                pnl=pos.pnl(cp);pos.fee_paid+=abs(pnl)*TAKER_FEE
                trades.append({'pnl':round(pnl-pos.fee_paid,2),'dir':pos.dir,'er':'tp','tag':'flip' if pos.flipped else 'sig'})
            else:
                alive.append(pos)
        for fd in new_flips:
            alive.append(Position(fd,FLIP_SZ,i,cp,FLIP_SL,FLIP_TP,flipped=True))
        
        bb=theta[i];ss=psi[i]
        if bearish_div[i]==1:ss=0
        if bullish_div[i]==1:bb=0
        hl=any(p.dir=='l' for p in alive);hs=any(p.dir=='s' for p in alive)
        
        if bb>ss and not hl:
            conf=int(bb/(bb+ss)*100)
            if conf>=50:
                p=Position('l',SIG_SZ,i,cp,SIG_SL,SIG_TP)
                p.fee_paid+=abs(p.pnl(cp))*TAKER_FEE;alive.append(p)
        if ss>bb and not hs:
            conf=int(ss/(bb+ss)*100)
            if conf>=50:
                p=Position('s',SIG_SZ,i,cp,SIG_SL,SIG_TP)
                p.fee_paid+=abs(p.pnl(cp))*TAKER_FEE;alive.append(p)
        
        positions=alive
    
    if not trades:continue
    rd=pd.DataFrame(trades)
    ttl=round(rd['pnl'].sum(),2)
    tp_cnt=len(rd[rd['er']=='tp'])
    sl_cnt=len(rd[rd['er']=='sl'])
    wr=tp_cnt/max(tp_cnt+sl_cnt,1)*100
    sig_trades=rd[rd['tag']=='sig']
    flip_trades=rd[rd['tag']=='flip']
    
    print(f'【{lb}】{DATA_SOURCE}')
    print(f'  信号: {len(sig_trades)}笔 {sig_trades["pnl"].sum():>+8.0f}  翻转: {len(flip_trades)}笔 {flip_trades["pnl"].sum():>+8.0f}')
    print(f'  合计{len(rd):>4}笔 胜率{wr:.0f}% 总盈亏{ttl:>+8.0f}  TP{tp_cnt} SL{sl_cnt}\n')

print('=== 完成 ===')
