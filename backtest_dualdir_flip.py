#!/usr/bin/env python3
"""对冲翻转v5 — 多空各1+SL对仓加6或新开+置信度"""
import json, warnings, sqlite3, uuid
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

DATA_SOURCE = 'trading'
BDB = 'data/SQLite/backtest_data.db'
TD = 'data/SQLite/trading.db'
CFG = json.load(open('strategy_config.json'))
ST = CFG['signal_thresholds']
SW = {'ma_trend':3,'macd':1,'rsi_extreme':1,'bollinger':1}

LEV=5; CM=0.1; TAKER_FEE=0.0005
SIG_SL=-7; SIG_TP=5; SIG_SZ=1
FLIP_SL=-8; FLIP_TP=8; FLIP_SZ=6

if DATA_SOURCE=='trading':
    conn=sqlite3.connect(TD)
    d=pd.read_sql("SELECT timestamp,close_price,high_price,low_price FROM indicators_snapshot ORDER BY id",conn)
    conn.close()
    d['ts']=d['timestamp'].apply(lambda t:int(pd.Timestamp(t).timestamp()*1000))
    d.rename(columns={'close_price':'close','high_price':'high','low_price':'low'},inplace=True)
    d['open']=d['close'];d['vol']=1.0
else:
    conn=sqlite3.connect(BDB)
    d=pd.read_sql('SELECT * FROM ohlcv_1h ORDER BY ts',conn);conn.close()

for c in 'open,high,low,close,vol'.split(','): d[c]=pd.to_numeric(d[c])
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
    __slots__=('pos_id','dir','sz','entry_p','entry_i','sl','tp','flipped','fee_paid','pa')
    def __init__(self,dir_,sz,i,p,sl,tp,flipped=False):
        self.pos_id=uuid.uuid4().hex[:12]
        self.dir=dir_;self.sz=sz;self.entry_p=p
        self.entry_i=i;self.sl=sl;self.tp=tp
        self.flipped=flipped;self.fee_paid=0.0;self.pa=False
    def cur(self,cp):
        return (cp-self.entry_p)/self.entry_p*LEV*100 if self.dir=='l' else (self.entry_p-cp)/self.entry_p*LEV*100
    def pnl(self,cp):
        return (cp-self.entry_p)*self.sz*CM if self.dir=='l' else (self.entry_p-cp)*self.sz*CM
    def add(self,sz,cp):
        old=self.sz;self.sz+=sz
        self.entry_p=(self.entry_p*old+cp*sz)/self.sz

def run():
    trades=[];positions=[]
    for i in range(N):
        if i>=N-2:break
        if np.isnan(ma[i]) or ma[i]<=0:continue
        cp=pr[i];new_flips=[]
        alive=[]
        for pos in positions:
            c=pos.cur(cp)
            if c<=pos.sl:
                pnl=pos.pnl(cp);pos.fee_paid+=abs(pnl)*TAKER_FEE
                trades.append({'pos_id':pos.pos_id,'dir':pos.dir,'sz':pos.sz,
                    'entry_i':pos.entry_i,'exit_i':i,'entry_p':pos.entry_p,'exit_p':cp,
                    'pnl':round(pnl-pos.fee_paid,2),'fee':round(pos.fee_paid,4),
                    'er':'sl','tag':'flip' if pos.flipped else 'sig','flipped':pos.flipped})
                fd='s' if pos.dir=='l' else 'l'
                # 找对仓是否还在→加6；不在→新开6
                found=False
                for ap in alive:
                    if ap.dir==fd:
                        ap.add(FLIP_SZ,cp)
                        trades.append({'pos_id':ap.pos_id,'dir':ap.dir,'sz':FLIP_SZ,
                            'entry_i':i,'exit_i':i,'entry_p':cp,'exit_p':cp,
                            'pnl':0,'fee':0,'er':'add','tag':'flip_add','flipped':True})
                        found=True;break
                if not found:
                    new_flips.append((fd,cp))
            elif c>=pos.tp:
                pnl=pos.pnl(cp);pos.fee_paid+=abs(pnl)*TAKER_FEE
                trades.append({'pos_id':pos.pos_id,'dir':pos.dir,'sz':pos.sz,
                    'entry_i':pos.entry_i,'exit_i':i,'entry_p':pos.entry_p,'exit_p':cp,
                    'pnl':round(pnl-pos.fee_paid,2),'fee':round(pos.fee_paid,4),
                    'er':'tp','tag':'flip' if pos.flipped else 'sig','flipped':pos.flipped})
            else:
                alive.append(pos)
        for fd,cp_ in new_flips:
            np_=Position(fd,FLIP_SZ,i,cp_,FLIP_SL,FLIP_TP,flipped=True)
            np_.fee_paid+=abs(np_.pnl(cp_))*TAKER_FEE
            alive.append(np_)
            trades.append({'pos_id':np_.pos_id,'dir':fd,'sz':FLIP_SZ,
                'entry_i':i,'exit_i':i,'entry_p':cp_,'exit_p':cp_,
                'pnl':0,'fee':round(np_.fee_paid,4),'er':'open','tag':'flip','flipped':True})
        
        # 置信度+信号→开双边
        bb=theta[i];ss=psi[i]
        if bearish_div[i]==1:ss=0
        if bullish_div[i]==1:bb=0
        hl=any(p.dir=='l' for p in alive)
        hs=any(p.dir=='s' for p in alive)
        # 开多
        if not hl and bb>ss:
            bc=int(bb/(bb+ss)*100)
            if not (cp<ma[i]*(1-0.009)):  # MA过滤不拦截
                p=Position('l',SIG_SZ,i,cp,SIG_SL,SIG_TP)
                p.fee_paid+=abs(p.pnl(cp))*TAKER_FEE
                alive.append(p)
                trades.append({'pos_id':p.pos_id,'dir':'l','sz':SIG_SZ,
                    'entry_i':i,'exit_i':i,'entry_p':cp,'exit_p':cp,
                    'pnl':0,'fee':round(p.fee_paid,4),'er':'open','tag':'sig','flipped':False})
        # 开空
        if not hs and ss>bb:
            bc=int(ss/(bb+ss)*100)
            if not (cp>ma[i]*(1+0.009)):
                p=Position('s',SIG_SZ,i,cp,SIG_SL,SIG_TP)
                p.fee_paid+=abs(p.pnl(cp))*TAKER_FEE
                alive.append(p)
                trades.append({'pos_id':p.pos_id,'dir':'s','sz':SIG_SZ,
                    'entry_i':i,'exit_i':i,'entry_p':cp,'exit_p':cp,
                    'pnl':0,'fee':round(p.fee_paid,4),'er':'open','tag':'sig','flipped':False})
        
        positions=alive
    return trades

print('=== 对冲翻转v5（双边+对仓加6+置信度）===')
print(f'信号: 多空各{SIG_SZ}张 SL={SIG_SL}% TP={SIG_TP}%')
print(f'翻转: {FLIP_SZ}张 SL={FLIP_SL}% TP={FLIP_TP}%')
print(f'MACD背离, MA40+0.9%缓冲\n')

trades=run()
rd=pd.DataFrame(trades)
PERIODS=[
    ('2025-01-01','2025-06-01','1~5月 下跌'),
    ('2025-06-01','2025-10-01','6~9月 上涨'),
    ('2025-10-01','2026-03-01','10~2月 震荡偏跌'),
    ('2026-03-01','2026-04-01','3月 筑底反弹'),
    ('2026-04-01','2026-06-13','4~6月 暴跌'),
    ('2025-01-01','2026-06-13','全周期'),
]

for st,en,lb in PERIODS:
    tss=int(pd.Timestamp(st).timestamp()*1000)
    tse=int(pd.Timestamp(en).timestamp()*1000)
    pt=rd[[tss<=ts_arr[r['entry_i']]<tse for _,r in rd.iterrows()]] if len(rd)>0 else rd
    if len(pt)==0:continue
    closed=pt[pt['er'].isin(['tp','sl'])]
    opens=pt[pt['er']=='open']
    adds=pt[pt['er']=='add']
    ttl=round(closed['pnl'].sum(),2)
    tp_cnt=len(closed[closed['er']=='tp'])
    sl_cnt=len(closed[closed['er']=='sl'])
    wr=tp_cnt/max(tp_cnt+sl_cnt,1)*100
    print(f'【{lb}】')
    print(f'  信号: {len(opens)}次 | 加仓: {len(adds)}次')
    for d in ['l','s']:
        sub=closed[closed['dir']==d]
        if len(sub)==0:continue
        print(f'  {"做多" if d=="l" else "做空"} {len(sub)}笔 {sub["pnl"].sum():>+8.0f}  '
              f'TP{len(sub[sub["er"]=="tp"])} SL{len(sub[sub["er"]=="sl"])}')
    print(f'  合计 {len(closed)}笔 胜率{wr:.0f}% 总盈亏{ttl:>+8.0f}  '
          f'TP{tp_cnt} SL{sl_cnt}')
    print()

pos_groups=rd.groupby('pos_id')
lifetimes=[]
for pid,grp in pos_groups:
    last=grp.iloc[-1]
    if last['er'] in ('tp','sl'):
        first=grp.iloc[0]
        hrs=(last['exit_i']-first['entry_i'])
        lifetimes.append({'pos_id':pid,'dir':first['dir'],'tag':first['tag'],
            'hours':hrs,'pnl':last['pnl'],'fee':grp['fee'].sum(),'er':last['er']})

if lifetimes:
    lf=pd.DataFrame(lifetimes)
    print('=== 持仓生命周期 ===')
    for tag in ['sig','flip']:
        sub=lf[lf['tag']==tag]
        if len(sub)==0:continue
        print(f'  {tag:>4}: {len(sub):>4}笔 | 均{sub["hours"].mean():.0f}h | '
              f'总{sub["pnl"].sum():>+8.0f} | 费{sub["fee"].sum():>+.2f}')

print(f'\n总交易: {len(rd)}笔')
print('=== 完成 ===')
