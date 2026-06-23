#!/usr/bin/env python3
"""
final_backtest.py — 最终策略回测脚本

策略参数：
  渐进补仓: -5/-6/-7/-8/-9/-10/-11/-12%
  最大补仓: 8次
  硬止损(T1前): -40%
  硬止损(T1后): -40%（即永不收紧）
  紧止损(crash/boom): -20%（三条件AND触发）
  动态TP: ✅（每次重算版）
  循环规则: ✅（补仓清TP1重置，TP1清补仓计数）
  收盘价计算: ✅（与实盘每小时查一次标记价格一致）
  杠杆: 5x  |  开仓: 3张  |  补仓: 3张
  超时清理: 500小时  |  单方向持仓: 1个
  方向信号匹配: 关闭（仅看亏损补仓）
"""

import sys, json, warnings, sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════
# 数据源配置（改这里切换）
# ═══════════════════════════════════════════════════════════
DATA_SOURCE = 'backtest_data'  # 'trading' 或 'backtest_data'
# DATA_SOURCE = 'trading'

# 自定义时间范围（设为 None 则用下方 PERIODS）
CUSTOM_START = None   # 如 '2026-03-29'
CUSTOM_END   = None   # 如 '2026-06-22'
# ═══════════════════════════════════════════════════════════

CFG = json.load(open('strategy_config.json'))
ST = CFG['signal_thresholds']
SW = CFG['signal_weights']

LA = 500
INIT = 3
ADD_SZ = 3
MAX_ADD = 8
LEV = 5
CM = 0.1
MH = 500
MP = 1

SL_N = -40.0
SL_AFTER_TP1 = -40.0
SL_TIGHT = -20.0

TP1_N = 5.0
TP2_N = 10.0
TP1_W = 7.0
TP2_W = 14.0

ADD_T = [-5, -6, -7, -8, -9, -10, -11, -12]
SMA_PERIOD = 15
BEIJING_TZ = ZoneInfo('Asia/Shanghai')

# ═══════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════
if DATA_SOURCE == 'trading':
    conn = sqlite3.connect('data/SQLite/trading.db')
    d = pd.read_sql('SELECT timestamp, close_price, high_price, low_price FROM indicators_snapshot ORDER BY id', conn)
    conn.close()
    def _to_ms(t):
        try: return int(datetime.fromisoformat(t).timestamp() * 1000)
        except: return 0
    d['ts'] = d['timestamp'].apply(_to_ms)
    d.rename(columns={'close_price': 'close', 'high_price': 'high', 'low_price': 'low'}, inplace=True)
    d['open'] = d['close']
    d['vol'] = 1.0
    src_label = 'trading.db indicators_snapshot'

else:  # backtest_data (默认)
    conn = sqlite3.connect('data/SQLite/backtest_data.db')
    d = pd.read_sql('SELECT * FROM ohlcv_1h ORDER BY ts', conn)
    conn.close()
    src_label = 'backtest_data.db ohlcv_1h'

for c in 'open,high,low,close,vol'.split(','):
    d[c] = pd.to_numeric(d[c])

d['cp'] = d['close']
d['hp'] = d['high']
d['lp'] = d['low']
d['ma'] = d['close'].rolling(SMA_PERIOD).mean()
d['e20'] = d['close'].ewm(span=20, adjust=False).mean()
d['e50'] = d['close'].ewm(span=50, adjust=False).mean()
e12 = d['close'].ewm(span=12, adjust=False).mean()
e26 = d['close'].ewm(span=26, adjust=False).mean()
d['macd'] = e12 - e26
d['ms'] = d['macd'].ewm(span=9, adjust=False).mean()

dl = d['close'].diff()
g = dl.clip(lower=0)
l_dn = -dl.clip(upper=0)
d['rsi'] = 100 - (100 / (1 + g.ewm(span=14, adjust=False).mean() / l_dn.ewm(span=14, adjust=False).mean().replace(0, np.nan)))

bm = d['close'].rolling(20).mean()
bs_ = d['close'].rolling(20).std()
d['bu'] = bm + 2 * bs_
d['bl'] = bm - 2 * bs_
d['tr'] = np.maximum(d['hp'] - d['lp'],
                     np.maximum(abs(d['hp'] - d['cp'].shift(1)),
                                abs(d['lp'] - d['cp'].shift(1))))
d['atr'] = d['tr'].ewm(span=14, adjust=False).mean()
d['atr_ma'] = d['atr'].rolling(50).mean()

df = d.iloc[70:].reset_index(drop=True)
N = len(df)
pr = df['cp'].values
ma = df['ma'].values
ts_arr = df['ts'].values
rsi = df['rsi'].values
atr = df['atr'].values
atr_ma = df['atr_ma'].values

crash = (rsi < 30) & (atr > atr_ma * 1.5) & (pr < ma)
boom = (rsi > 70) & (atr > atr_ma * 1.5) & (pr > ma)

theta = np.zeros(N)
psi = np.zeros(N)
for i in range(1, N):
    l = df.iloc[i]
    p = df.iloc[i - 1]
    bb = 0
    ss = 0
    c = l['cp']
    if c is None or pd.isna(c):
        continue

    e20 = l['e20']
    e50 = l['e50']
    if not pd.isna(e20) and not pd.isna(e50):
        if e20 > e50 and c > e20:
            bb += SW['ma_trend']
        elif e20 < e50 and c < e20:
            ss += SW['ma_trend']

    macd = l['macd']
    ms = l['ms']
    pm = p['macd']
    pms = p['ms']
    if not any(pd.isna(x) for x in [macd, ms, pm, pms]):
        if macd > ms and pm <= pms:
            bb += SW['macd']
        elif macd < ms and pm >= pms:
            ss += SW['macd']
        elif macd > ms:
            bb += SW['macd'] * 0.5
        elif macd < ms:
            ss += SW['macd'] * 0.5

    r = l['rsi']
    if not pd.isna(r):
        if r < ST['rsi_oversold']:
            bb += SW['rsi_extreme']
        if r > ST['rsi_overbought']:
            ss += SW['rsi_extreme']

    if c > l['bu']:
        ss += SW['bollinger']
    if c < l['bl']:
        bb += SW['bollinger']

    theta[i] = bb
    psi[i] = ss


def sim(sd, idx):
    ep = pr[idx]
    ct = INIT
    ec = ep
    ac = 0
    tot = 0.0
    t1t = False
    t2t = False
    er = 'timeout'
    exp = None
    tight = False
    tp1_hit = False

    for off in range(1, min(LA, len(pr) - idx)):
        cp = pr[idx + off]
        if np.isnan(cp):
            break

        if MH > 0 and off >= MH:
            er = 'to'
            exp = cp
            tot += (cp - ec) * ct * CM if sd == 'l' else (ec - cp) * ct * CM
            ct = 0
            break

        cur = (cp - ec) / ec * LEV * 100 if sd == 'l' else (ec - cp) / ec * LEV * 100

        bar = idx + off
        _c = bar < N and crash[bar]
        _b = bar < N and boom[bar]

        if not tight and ((sd == 'l' and _c) or (sd == 's' and _b)):
            tight = True

        crash_sl = SL_TIGHT if tight else SL_N
        tp1_sl = SL_AFTER_TP1 if tp1_hit else SL_N
        use_sl = max(crash_sl, tp1_sl)

        use_t1 = TP1_W if ((sd == 's' and _c) or (sd == 'l' and _b)) else TP1_N
        use_t2 = TP2_W if ((sd == 's' and _c) or (sd == 'l' and _b)) else TP2_N

        if cur <= use_sl:
            er = 'sl'
            exp = cp
            break

        if not t2t and cur >= use_t2:
            if not t1t:
                t1t = True
                sz = max(1, int(ct * 0.5))
                tot += (cp - ec) * sz * CM if sd == 'l' else (ec - cp) * sz * CM
                ct -= sz
                ac = 0
                tp1_hit = True
            t2t = True
            tot += (cp - ec) * ct * CM if sd == 'l' else (ec - cp) * ct * CM
            ct = 0
            er = 'tp2'
            exp = cp
            break

        if not t1t and cur >= use_t1:
            t1t = True
            sz = max(1, int(ct * 0.5))
            tot += (cp - ec) * sz * CM if sd == 'l' else (ec - cp) * sz * CM
            ct -= sz
            ac = 0
            tp1_hit = True

        if ac < MAX_ADD and cur <= ADD_T[min(ac, len(ADD_T) - 1)]:
            ec = (ec * ct + cp * ADD_SZ) / (ct + ADD_SZ)
            ct += ADD_SZ
            ac += 1
            t1t = False

    if ct > 0:
        if exp is None:
            exp = pr[min(idx + LA, len(pr) - 1)]
        tot += (exp - ec) * ct * CM if sd == 'l' else (ec - exp) * ct * CM

    return {'pnl': round(tot, 2), 'win': tot > 0, 'er': er, 'ex': idx + off}


print('=== 最终策略回测 ===')
print(f'MA={SMA_PERIOD} | 缓冲=0.8% | SL_AFTER_TP1={SL_AFTER_TP1:.0f}% | 数据源: {src_label}\n')

PERIODS = [
    ('2025-06-01', '2025-10-01', '6~9月 上涨周期'),
    ('2025-10-01', '2026-03-01', '10~2月 震荡周期'),
    ('2025-01-01', '2025-06-01', '1~5月 下跌周期'),
    ('2026-03-01', '2026-04-01', '3月 筑底反弹'),
    ('2026-04-01', '2026-06-11', '4~6月 暴跌'),
    ('2025-01-01', '2026-06-11', '全周期'),
] if DATA_SOURCE == 'backtest_data' else [
    ('2026-03-21', '2026-06-23', '全量数据期 (3.21~6.23)'),
    ('2026-03-29', '2026-06-22', '实盘期 (3.29~6.22)'),
]

# 自定义时间范围优先
if CUSTOM_START and CUSTOM_END:
    PERIODS = [(CUSTOM_START, CUSTOM_END, f'{CUSTOM_START} ~ {CUSTOM_END}')]

for st, en, lb in PERIODS:
    tss = int(pd.Timestamp(st).timestamp() * 1000)
    tse = int(pd.Timestamp(en).timestamp() * 1000)
    idx = [i for i in range(N) if tss <= ts_arr[i] < tse]
    if len(idx) < 10:
        continue

    al, as_, dl, ds = [], [], [], []
    for i in idx:
        if i >= N - 2:
            break
        if np.isnan(ma[i]) or ma[i] <= 0:
            continue
        wl = (theta[i] > psi[i]) and (pr[i] > ma[i] * 1.008)
        ws = (psi[i] > theta[i]) and (pr[i] < ma[i] * 0.992)
        al = [p for p in al if i < p['ex']]
        as_ = [p for p in as_ if i < p['ex']]
        if wl and len(al) < MP:
            r = sim('l', i)
            al.append(r)
            dl.append(r)
        if ws and len(as_) < MP:
            r = sim('s', i)
            as_.append(r)
            ds.append(r)

    ttl = 0
    if dl:
        rd = pd.DataFrame(dl)
        sls = rd[rd['er'] == 'sl']['pnl'].sum() if 'sl' in rd['er'].values else 0
        l = rd['pnl'].sum()
        ttl += l
        print(f'  做多 {len(rd):>3}笔 {rd["win"].mean() * 100:.0f}% {l:>+8.0f}  SL{sls:+.0f}')

    if ds:
        rd = pd.DataFrame(ds)
        sls = rd[rd['er'] == 'sl']['pnl'].sum() if 'sl' in rd['er'].values else 0
        l = rd['pnl'].sum()
        ttl += l
        print(f'  做空 {len(rd):>3}笔 {rd["win"].mean() * 100:.0f}% {l:>+8.0f}  SL{sls:+.0f}')

    print(f'  >>> {lb} {ttl:+.0f}\n')

print('=== 回测完成 ===')
print(f'数据源: {src_label} ({N}条记录)')
print(f'MA={SMA_PERIOD} | 缓冲=0.8% | 杠杆={LEV}x | 开仓={INIT}张 | 补仓={ADD_SZ}张x{MAX_ADD}次')
