#!/usr/bin/env python3
"""
สแกนหุ้นไทย สำหรับใช้ส่วนตัว
---------------------------------
ดึงราคาหุ้นจาก Yahoo Finance (ผ่าน yfinance) คำนวณ 6 สัญญาณ
แล้วสร้างหน้าเว็บแบบการ์ด (scanner.html) เปิดดูในเบราว์เซอร์ได้เลย

วิธีใช้
  pip install yfinance pandas numpy
  python thai_stock_scanner.py            # ข้อมูลจริง
  python thai_stock_scanner.py --buzz     # นับข่าวจาก Google News ด้วย (ช้าลง)
  python thai_stock_scanner.py --demo     # ข้อมูลสุ่ม ไว้ลองหน้าตา
  python thai_stock_scanner.py --top 0    # แสดงทุกตัว ไม่ตัดแค่ 20

บน Google Colab: วางโค้ดทั้งไฟล์ในเซลล์ แล้วเพิ่มบรรทัดแรก
  !pip install -q yfinance
หน้าเว็บจะแสดงใต้เซลล์ทันที

หมายเหตุ
- ข้อมูล Yahoo ช้ากว่าตลาดจริงประมาณ 15–20 นาที
- ถ้ารันระหว่างวัน วอลุ่มของวันนี้ยังไม่ครบ คะแนน Volume จะต่ำกว่าจริง
  แนะนำให้รันหลังตลาดปิด (หลัง 17:00)
- ไม่ใช่คำแนะนำการลงทุน
"""

import argparse
import datetime as dt
import json
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ตั้งค่า
# ---------------------------------------------------------------------------

# รายชื่อหุ้นที่จะสแกน (ประมาณ SET50) แก้ไข เพิ่ม หรือลดได้ตามใจ
# รายชื่อ SET50 เปลี่ยนทุกครึ่งปี ตัวที่ดึงข้อมูลไม่ได้จะถูกข้ามอัตโนมัติ
UNIVERSE = [
    "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BGRIM", "BH",
    "BJC", "BTS", "CBG", "CCET", "CENTEL", "COM7", "CPALL", "CPF", "CPN",
    "CRC", "DELTA", "EGCO", "GLOBAL", "GPSC", "GULF", "HMPRO", "ITC", "IVL",
    "KBANK", "KKP", "KTB", "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT",
    "PTTEP", "PTTGC", "RATCH", "SCB", "SCC", "SCGP", "TCAP", "TIDLOR",
    "TISCO", "TLI", "TOP", "TRUE", "TTB", "WHA",
]

INDEX_TICKER = "^SET.BK"

# น้ำหนักของแต่ละสัญญาณในคะแนนรวม (สัญญาณที่ไม่มีข้อมูลจะไม่ถูกนับ)
WEIGHTS = {
    "volume": 25,
    "breakout": 20,
    "relstr": 20,
    "ma": 20,
    "rsi": 15,
    "buzz": 10,
}

# เกณฑ์แบ่งกลุ่ม
HOT_MIN = 70     # ร้อนแรง
WARM_MIN = 55    # เริ่มต้น  (ต่ำกว่านี้ = จับตา)

SIGNAL_NAMES = {
    "breakout": "Breakout",
    "relstr": "Rel.Str",
    "volume": "Volume",
    "rsi": "RSI",
    "buzz": "Buzz",
    "ma": "MA",
}
SIGNAL_ORDER = ["breakout", "relstr", "volume", "rsi", "buzz", "ma"]


# ---------------------------------------------------------------------------
# ตัวช่วยคำนวณ
# ---------------------------------------------------------------------------

BKK = dt.timezone(dt.timedelta(hours=7))


def now_bkk():
    return dt.datetime.now(BKK)


def session_fraction(now=None):
    """สัดส่วนเวลาซื้อขายที่ผ่านไปแล้ววันนี้ (เช้า 10:00–12:30, บ่าย 14:30–16:30)"""
    now = now or now_bkk()
    m = now.hour * 60 + now.minute
    morning = max(0, min(m, 750) - 600)          # 10:00–12:30 = 150 นาที
    afternoon = max(0, min(m, 990) - 870)        # 14:30–16:30 = 120 นาที
    return (morning + afternoon) / 270


def clip(x):
    return float(max(0.0, min(100.0, x)))


def rsi_wilder(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def pct(x, digits=1):
    return f"{x * 100:+.{digits}f}%"


# ---------------------------------------------------------------------------
# วิเคราะห์หุ้นหนึ่งตัว
# ---------------------------------------------------------------------------

def analyze(sym, df, idx_ret20=None, buzz=None, bench_name="SET", vol_frac=1.0):
    df = df.dropna(subset=["Close"])
    if len(df) < 60:
        return None

    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    h = df["High"].astype(float)

    last = float(c.iloc[-1])
    chg1 = last / float(c.iloc[-2]) - 1
    chg5 = last / float(c.iloc[-6]) - 1

    sig = {}

    # Volume: วอลุ่มวันนี้เทียบค่าเฉลี่ย 20 วันก่อนหน้า
    avg20 = float(v.iloc[-21:-1].mean())
    today_vol = float(v.iloc[-1])
    est = ""
    if vol_frac < 1.0 and vol_frac > 0.05:
        today_vol = today_vol / vol_frac          # ประมาณการวอลุ่มทั้งวันระหว่างตลาดเปิด
        est = " (ประมาณการทั้งวัน)"
    ratio = today_vol / avg20 if avg20 > 0 else 0.0
    sig["volume"] = (
        clip((ratio - 0.5) / 2.5 * 100),          # 1 เท่า = 20, 2 เท่า = 60, 3 เท่า = 100
        f"วอลุ่มวันนี้ {ratio:.1f} เท่าของค่าเฉลี่ย 20 วัน{est}",
    )

    # RSI: ให้คะแนนสูงสุดที่ราว 65 (มีแรงส่งแต่ยังไม่ร้อนเกิน)
    rsi = float(rsi_wilder(c).iloc[-1])
    note = ""
    if rsi >= 75:
        note = " (ซื้อมากเกินไป ระวังย่อ)"
    elif rsi <= 30:
        note = " (ขายมากเกินไป)"
    sig["rsi"] = (clip(100 - abs(rsi - 65) * 2.5), f"RSI(14) = {rsi:.1f}{note}")

    # MA: ราคาเทียบเส้นค่าเฉลี่ย 20 และ 50 วัน
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    s_ma = (35 if last > ma20 else 0) + (35 if last > ma50 else 0) + (30 if ma20 > ma50 else 0)
    parts = [
        "เหนือ MA20" if last > ma20 else "ใต้ MA20",
        "เหนือ MA50" if last > ma50 else "ใต้ MA50",
        "MA20 อยู่เหนือ MA50" if ma20 > ma50 else "MA20 อยู่ใต้ MA50",
    ]
    sig["ma"] = (float(s_ma), "ราคา" + " · ".join(parts[:2]) + f", {parts[2]}")

    # Breakout: ราคาเทียบจุดสูงสุด 20 วันก่อนหน้า
    hi20 = float(h.iloc[-21:-1].max())
    dist = last / hi20 - 1
    if dist >= 0:
        txt = f"ทะลุจุดสูงสุด 20 วัน {pct(dist)}"
    else:
        txt = f"ต่ำกว่าจุดสูงสุด 20 วัน {pct(dist)}"
    sig["breakout"] = (clip((dist + 0.10) / 0.15 * 100), txt)   # -10% = 0, ที่จุดสูงสุด = 67, +5% = 100

    # Rel.Str: ผลตอบแทน 20 วันเทียบดัชนี SET
    ret20 = last / float(c.iloc[-21]) - 1
    if idx_ret20 is not None:
        rel = ret20 - idx_ret20
        sig["relstr"] = (
            clip(50 + rel * 500),
            f"20 วัน: หุ้น {pct(ret20)} เทียบ {bench_name} {pct(idx_ret20)}",
        )
    else:
        sig["relstr"] = (None, "ไม่มีข้อมูลดัชนี SET ไม่นับสัญญาณนี้")

    # Buzz: จำนวนข่าวใน 3 วัน
    if buzz is not None:
        sig["buzz"] = (clip(buzz / 15 * 100), f"ข่าว {buzz} ชิ้นใน 3 วันล่าสุด")
    else:
        sig["buzz"] = (None, "ไม่ได้เปิดการนับข่าว (ใช้ --buzz)")

    # คะแนนรวม
    tot_w = sum(WEIGHTS[k] for k, (s, _) in sig.items() if s is not None)
    score = sum(WEIGHTS[k] * s for k, (s, _) in sig.items() if s is not None) / tot_w

    if score >= HOT_MIN:
        label = "hot"
    elif score >= WARM_MIN:
        label = "warm"
    else:
        label = "watch"

    return {
        "sym": sym,
        "price": round(last, 2),
        "chg1": round(chg1 * 100, 2),
        "chg5": round(chg5 * 100, 2),
        "score": round(score),
        "label": label,
        "signals": [
            {
                "key": k,
                "name": SIGNAL_NAMES[k],
                "score": None if sig[k][0] is None else round(sig[k][0]),
                "detail": sig[k][1],
            }
            for k in SIGNAL_ORDER
        ],
    }


# ---------------------------------------------------------------------------
# ดึงข้อมูล
# ---------------------------------------------------------------------------

def fetch_real(symbols, period="1y"):
    import yfinance as yf

    tickers = [s + ".BK" for s in symbols]
    print(f"กำลังดึงข้อมูล {len(tickers)} ตัวจาก Yahoo Finance ...")
    raw = yf.download(
        tickers, period=period, interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    data = {}
    for s, t in zip(symbols, tickers):
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(how="all")
            if len(df):
                data[s] = df
        except KeyError:
            pass

    idx_close = None
    try:
        ix = yf.download(INDEX_TICKER, period=period, interval="1d",
                         auto_adjust=True, progress=False)
        cl = ix["Close"]
        if isinstance(cl, pd.DataFrame):
            cl = cl.iloc[:, 0]
        cl = cl.dropna()
        if len(cl) > 25:
            idx_close = cl
    except Exception as e:
        print("ดึงดัชนี SET ไม่ได้:", e)

    missing = [s for s in symbols if s not in data]
    if missing:
        print("ข้ามตัวที่ไม่มีข้อมูล:", ", ".join(missing))
    return data, idx_close


def fetch_buzz(symbols, days=3):
    """นับข่าวจาก Google News RSS (ชื่อย่อสั้นๆ อย่าง OR, TOP อาจนับเกินจริง)"""
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor

    def one(sym):
        q = urllib.parse.quote(f'"{sym}" หุ้น when:{days}d')
        url = f"https://news.google.com/rss/search?q={q}&hl=th&gl=TH&ceid=TH:th"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                root = ET.fromstring(r.read())
            return sym, len(root.findall("./channel/item"))
        except Exception:
            return sym, None

    print("กำลังนับข่าว ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, symbols))


def demo_data(symbols, days=260, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {}
    for s in symbols:
        rets = rng.normal(rng.normal(0.0003, 0.0008), 0.017, days)
        kind = rng.random()
        if kind < 0.22:
            rets[-6:] += rng.uniform(0.008, 0.035, 6)
        elif kind < 0.45:
            rets[-15:] += rng.uniform(0.002, 0.010, 15)
        elif kind > 0.85:
            rets[-5:] -= rng.uniform(0.005, 0.025, 5)
        price = rng.uniform(5, 200) * np.exp(np.cumsum(rets))
        vol = rng.uniform(1e6, 5e7) * rng.lognormal(0, 0.35, days)
        if kind < 0.22:
            vol[-1] *= rng.uniform(2.2, 4)
        elif kind < 0.45:
            vol[-1] *= rng.uniform(1.2, 2.2)
        data[s] = pd.DataFrame({
            "Open": price,
            "High": price * (1 + np.abs(rng.normal(0, 0.007, days))),
            "Low": price * (1 - np.abs(rng.normal(0, 0.007, days))),
            "Close": price,
            "Volume": vol,
        }, index=idx)
    idx_close = pd.Series(1400 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, days))), index=idx)
    buzz = {s: int(rng.poisson(5)) for s in symbols}
    return data, idx_close, buzz


# ---------------------------------------------------------------------------
# สร้างหน้าเว็บ
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
__REFRESH__<title>สแกนหุ้น</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme: light;
  --bg:#EEF1F5; --surface:#FFFFFF; --surface-2:#F4F6F9; --line:#DCE2EA;
  --ink:#152033; --muted:#5E6B7E; --track:#E3E8EF;
  --hot:#D9482F; --hot-soft:#FBE6E1;
  --warm:#B98309; --warm-soft:#FBF0D3;
  --watch:#3A6FD8; --watch-soft:#E2EAFB;
  --up:#16865A; --up-soft:#DDF3E8; --down:#C23B3B; --down-soft:#F9E1E1;
  --focus:#3A6FD8;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px);
  padding-bottom:env(safe-area-inset-bottom,0px);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme: dark;
    --bg:#0D1422; --surface:#151E30; --surface-2:#1B2539; --line:#25314A;
    --ink:#E7ECF4; --muted:#8B98AD; --track:#25314A;
    --hot:#F2694F; --hot-soft:#3A1F1D;
    --warm:#E9B640; --warm-soft:#36301A;
    --watch:#6E9BF2; --watch-soft:#1D2A47;
    --up:#45C48B; --up-soft:#16342A; --down:#F07070; --down-soft:#3A1E22;
    --focus:#6E9BF2;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --bg:#0D1422; --surface:#151E30; --surface-2:#1B2539; --line:#25314A;
  --ink:#E7ECF4; --muted:#8B98AD; --track:#25314A;
  --hot:#F2694F; --hot-soft:#3A1F1D;
  --warm:#E9B640; --warm-soft:#36301A;
  --watch:#6E9BF2; --watch-soft:#1D2A47;
  --up:#45C48B; --up-soft:#16342A; --down:#F07070; --down-soft:#3A1E22;
  --focus:#6E9BF2;
}
*,*::before,*::after{box-sizing:inherit}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"IBM Plex Sans Thai", "Noto Sans Thai", Tahoma, system-ui, sans-serif;
  font-size:16px; line-height:1.5; -webkit-font-smoothing:antialiased;
  font-variant-numeric: tabular-nums;
}
.wrap{max-width:640px; margin:0 auto; padding:20px 16px 48px}
header h1{font-size:28px; line-height:1.2; margin:0 0 4px; font-weight:700; letter-spacing:-.01em}
.meta{color:var(--muted); font-size:14px; margin:0}
.demo{margin:14px 0 0; padding:10px 12px; border-radius:10px; background:var(--warm-soft); color:var(--ink); font-size:14px}

.filters{display:flex; gap:8px; margin:20px 0 12px; overflow-x:auto; padding-bottom:2px}
.chip{
  flex:0 0 auto; border:1px solid var(--line); background:var(--surface); color:var(--ink);
  font:inherit; font-size:14px; font-weight:500; padding:7px 14px; border-radius:999px; cursor:pointer;
  display:inline-flex; align-items:center; gap:8px;
}
.chip .n{color:var(--muted); font-weight:600}
.chip[aria-pressed="true"]{background:var(--ink); color:var(--bg); border-color:var(--ink)}
.chip[aria-pressed="true"] .n{color:inherit; opacity:.7}
.dot{width:8px; height:8px; border-radius:50%; display:inline-block}
.dot.hot{background:var(--hot)} .dot.warm{background:var(--warm)} .dot.watch{background:var(--watch)}

.tools{display:flex; gap:8px; margin-bottom:16px}
.tools input,.tools select{
  font:inherit; font-size:15px; color:var(--ink); background:var(--surface);
  border:1px solid var(--line); border-radius:10px; padding:9px 12px;
}
.tools input{flex:1; min-width:0; text-transform:uppercase}
.tools input::placeholder{text-transform:none; color:var(--muted)}
:focus-visible{outline:2px solid var(--focus); outline-offset:2px}

.list{display:flex; flex-direction:column; gap:10px}
.card{
  background:var(--surface); border-radius:14px; border:1px solid var(--line);
  border-left:5px solid var(--rail); overflow:hidden;
}
.card.hot{--rail:var(--hot); --tone:var(--hot); --tone-soft:var(--hot-soft)}
.card.warm{--rail:var(--warm); --tone:var(--warm); --tone-soft:var(--warm-soft)}
.card.watch{--rail:var(--watch); --tone:var(--watch); --tone-soft:var(--watch-soft)}
.card summary{list-style:none; cursor:pointer; padding:14px 14px 12px}
.card summary::-webkit-details-marker{display:none}

.head{display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:4px 12px}
.rank{color:var(--muted); font-size:13px; font-weight:600; min-width:24px}
.name{display:flex; align-items:baseline; gap:8px; flex-wrap:wrap}
.sym{font-size:21px; font-weight:700; letter-spacing:.01em}
.price{color:var(--muted); font-size:14px}
.score{text-align:right; line-height:1}
.score b{font-size:30px; font-weight:700; color:var(--tone)}
.score small{display:block; font-size:11px; color:var(--muted); margin-top:3px}
.sub{grid-column:2 / 4; display:flex; align-items:center; gap:8px; flex-wrap:wrap}
.tag{font-size:13px; font-weight:600; padding:2px 10px; border-radius:999px; background:var(--tone-soft); color:var(--tone)}
.chg{font-size:13px; font-weight:600; padding:2px 8px; border-radius:6px}
.chg.up{background:var(--up-soft); color:var(--up)}
.chg.down{background:var(--down-soft); color:var(--down)}
.chg5{font-size:13px; color:var(--muted)}

.bars{margin-top:12px; display:grid; grid-template-columns:64px 1fr 30px; gap:7px 10px; align-items:center}
.bars .lb{font-size:13px; color:var(--muted)}
.bars .tr{height:6px; background:var(--track); border-radius:3px; overflow:hidden}
.bars .fl{height:100%; background:var(--tone); border-radius:3px}
.bars .vl{font-size:13px; font-weight:600; text-align:right}
.bars .na{font-size:12px; color:var(--muted); text-align:right}
.more{margin-top:10px; font-size:13px; color:var(--muted)}
.card[open] .more .open{display:none}
.card:not([open]) .more .close{display:none}

.detail{padding:0 14px 14px 14px; border-top:1px solid var(--line); background:var(--surface-2)}
.detail dl{margin:0; display:grid; grid-template-columns:72px 1fr; gap:8px 12px; padding-top:12px}
.detail dt{font-size:13px; color:var(--muted)}
.detail dd{margin:0; font-size:14px}

.empty{padding:32px 16px; text-align:center; color:var(--muted)}
footer{margin-top:28px; color:var(--muted); font-size:13px}
footer details{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px 14px; margin-bottom:14px; color:var(--ink)}
footer summary{cursor:pointer; font-weight:600; font-size:14px}
footer table{border-collapse:collapse; width:100%; margin-top:10px; font-size:13px}
footer td{padding:6px 0; border-top:1px solid var(--line); vertical-align:top}
footer td:first-child{width:76px; font-weight:600}
footer td:nth-child(2){width:44px; color:var(--muted)}
@media (prefers-reduced-motion: no-preference){
  .bars .fl{transition:width .4s ease}
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>สแกนหุ้น</h1>
    <p class="meta" id="meta"></p>
    <p class="demo" id="demo" hidden>นี่คือข้อมูลสุ่มไว้ดูหน้าตา รันสคริปต์โดยไม่ใส่ --demo เพื่อดูหุ้นจริง</p>
  </header>

  <div class="filters" role="group" aria-label="กรองตามกลุ่ม">
    <button class="chip" data-f="all" aria-pressed="true">ทั้งหมด <span class="n" id="n-all"></span></button>
    <button class="chip" data-f="hot" aria-pressed="false"><span class="dot hot"></span>ร้อนแรง <span class="n" id="n-hot"></span></button>
    <button class="chip" data-f="warm" aria-pressed="false"><span class="dot warm"></span>เริ่มต้น <span class="n" id="n-warm"></span></button>
    <button class="chip" data-f="watch" aria-pressed="false"><span class="dot watch"></span>จับตา <span class="n" id="n-watch"></span></button>
  </div>

  <div class="tools">
    <input id="q" type="search" placeholder="ค้นหาชื่อหุ้น" aria-label="ค้นหาชื่อหุ้น" autocomplete="off">
    <select id="sort" aria-label="เรียงตาม">
      <option value="score">คะแนนสูงสุด</option>
      <option value="chg1">ขึ้นมากสุดวันนี้</option>
      <option value="chg5">ขึ้นมากสุด 5 วัน</option>
    </select>
  </div>

  <div class="list" id="list"></div>

  <footer>
    <details>
      <summary>คะแนนคิดอย่างไร</summary>
      <table id="how"></table>
    </details>
    <p>คำนวณจากราคาและวอลุ่มย้อนหลังเท่านั้น ใช้ประกอบการตัดสินใจ ไม่ใช่คำแนะนำการลงทุน</p>
  </footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  var payload = JSON.parse(document.getElementById('data').textContent);
  var rows = payload.rows;
  var LABEL = {hot:'ร้อนแรง', warm:'เริ่มต้น', watch:'จับตา'};
  var state = {f:'all', q:'', sort:'score'};

  function el(tag, cls, text){
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function fmtPct(x){ return (x > 0 ? '+' : '') + x.toFixed(2) + '%'; }

  document.getElementById('meta').textContent =
    'สแกน ' + payload.scanned + ' ตัว, แสดง ' + rows.length + ' ตัว, อัปเดต ' + payload.updated;
  if (payload.demo) document.getElementById('demo').hidden = false;

  var counts = {all:rows.length, hot:0, warm:0, watch:0};
  rows.forEach(function(r){ counts[r.label]++; });
  Object.keys(counts).forEach(function(k){ document.getElementById('n-'+k).textContent = counts[k]; });

  var how = document.getElementById('how');
  payload.how.forEach(function(h){
    var tr = el('tr');
    tr.appendChild(el('td', null, h.name));
    tr.appendChild(el('td', null, h.weight));
    tr.appendChild(el('td', null, h.text));
    how.appendChild(tr);
  });

  function card(r, i){
    var d = el('details', 'card ' + r.label);
    var s = el('summary');
    var head = el('div', 'head');
    head.appendChild(el('span', 'rank', '#' + (i + 1)));
    var name = el('div', 'name');
    name.appendChild(el('span', 'sym', r.sym));
    name.appendChild(el('span', 'price', r.price.toFixed(2) + ' บาท'));
    head.appendChild(name);
    var sc = el('div', 'score');
    sc.appendChild(el('b', null, r.score));
    sc.appendChild(el('small', null, 'คะแนน'));
    head.appendChild(sc);
    var sub = el('div', 'sub');
    sub.appendChild(el('span', 'tag', LABEL[r.label]));
    sub.appendChild(el('span', 'chg ' + (r.chg1 >= 0 ? 'up' : 'down'), fmtPct(r.chg1) + ' วันนี้'));
    sub.appendChild(el('span', 'chg5', '5 วัน ' + fmtPct(r.chg5)));
    head.appendChild(sub);
    s.appendChild(head);

    var bars = el('div', 'bars');
    r.signals.forEach(function(g){
      bars.appendChild(el('span', 'lb', g.name));
      var tr = el('div', 'tr');
      var fl = el('div', 'fl');
      fl.style.width = (g.score == null ? 0 : g.score) + '%';
      tr.appendChild(fl);
      bars.appendChild(tr);
      if (g.score == null) bars.appendChild(el('span', 'na', 'ไม่นับ'));
      else bars.appendChild(el('span', 'vl', g.score));
    });
    s.appendChild(bars);

    var more = el('div', 'more');
    more.appendChild(el('span', 'open', 'แตะเพื่อดูที่มาของคะแนน'));
    more.appendChild(el('span', 'close', 'ซ่อนรายละเอียด'));
    s.appendChild(more);
    d.appendChild(s);

    var det = el('div', 'detail');
    var dl = el('dl');
    r.signals.forEach(function(g){
      dl.appendChild(el('dt', null, g.name));
      dl.appendChild(el('dd', null, g.detail));
    });
    det.appendChild(dl);
    d.appendChild(det);
    return d;
  }

  function render(){
    var list = document.getElementById('list');
    list.textContent = '';
    var q = state.q.trim().toUpperCase();
    var out = rows.filter(function(r){
      return (state.f === 'all' || r.label === state.f) && (!q || r.sym.indexOf(q) !== -1);
    });
    out.sort(function(a, b){ return b[state.sort] - a[state.sort]; });
    if (!out.length){
      list.appendChild(el('div', 'empty', q ? 'ไม่พบหุ้น "' + q + '" ในกลุ่มนี้ ลองเลือก "ทั้งหมด"' : 'วันนี้ไม่มีหุ้นในกลุ่มนี้'));
      return;
    }
    out.forEach(function(r, i){ list.appendChild(card(r, i)); });
  }

  document.querySelectorAll('.chip').forEach(function(b){
    b.addEventListener('click', function(){
      state.f = b.getAttribute('data-f');
      document.querySelectorAll('.chip').forEach(function(x){
        x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
      });
      render();
    });
  });
  document.getElementById('q').addEventListener('input', function(e){ state.q = e.target.value; render(); });
  document.getElementById('sort').addEventListener('change', function(e){ state.sort = e.target.value; render(); });
  render();
})();
</script>
</body>
</html>
"""

HOW = [
    ("Breakout", "breakout", "ราคาเทียบจุดสูงสุด 20 วัน ต่ำกว่า 10% ได้ 0, แตะจุดสูงสุดได้ราว 67, ทะลุ 5% ได้ 100"),
    ("Rel.Str", "relstr", "ผลตอบแทน 20 วันเทียบดัชนี SET เท่ากันได้ 50, ชนะ 10% ได้ 100"),
    ("Volume", "volume", "วอลุ่มวันนี้เทียบค่าเฉลี่ย 20 วัน 1 เท่าได้ 20, 2 เท่าได้ 60, 3 เท่าขึ้นไปได้ 100"),
    ("RSI", "rsi", "RSI 14 วัน ได้เต็มที่ราว 65 และลดลงเมื่อต่ำไปหรือร้อนเกินไป"),
    ("Buzz", "buzz", "จำนวนข่าวใน 3 วันจาก Google News 15 ชิ้นขึ้นไปได้ 100 (เปิดด้วย --buzz)"),
    ("MA", "ma", "เหนือ MA20 ได้ 35, เหนือ MA50 ได้ 35, MA20 อยู่เหนือ MA50 ได้ 30"),
]


def build_html(rows, scanned, demo, refresh=0):
    total_w = sum(WEIGHTS.values())
    payload = {
        "rows": rows,
        "scanned": scanned,
        "demo": demo,
        "updated": now_bkk().strftime("%d/%m/%Y %H:%M"),
        "how": [
            {"name": n, "weight": f"{WEIGHTS[k] * 100 // total_w}%", "text": t}
            for n, k, t in HOW
        ]
        + [{"name": "กลุ่ม", "weight": "",
            "text": f"ร้อนแรง {HOT_MIN} ขึ้นไป, เริ่มต้น {WARM_MIN}–{HOT_MIN - 1}, ต่ำกว่านั้นคือจับตา"}],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    meta = f'<meta http-equiv="refresh" content="{int(refresh)}">\n' if refresh > 0 else ""
    return HTML_TEMPLATE.replace("__DATA__", data).replace("__REFRESH__", meta)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="สแกนหุ้นไทยแบบหลายสัญญาณ")
    ap.add_argument("--demo", action="store_true", help="ใช้ข้อมูลสุ่ม ไม่ต้องต่อเน็ต")
    ap.add_argument("--buzz", action="store_true", help="นับข่าวจาก Google News")
    ap.add_argument("--top", type=int, default=20, help="จำนวนหุ้นที่แสดง (0 = ทั้งหมด)")
    ap.add_argument("--out", default="scanner.html", help="ชื่อไฟล์หน้าเว็บ")
    ap.add_argument("--no-open", action="store_true", help="ไม่ต้องเปิดเบราว์เซอร์อัตโนมัติ")
    ap.add_argument("--refresh", type=int, default=0, help="ให้หน้าเว็บโหลดใหม่เองทุกกี่วินาที (0 = ปิด)")
    args, _ = ap.parse_known_args()   # parse_known_args ให้รันบน Colab/Jupyter ได้

    if args.demo:
        data, idx_close, buzz = demo_data(UNIVERSE)
        if not args.buzz:
            buzz = {}
    else:
        data, idx_close = fetch_real(UNIVERSE)
        buzz = fetch_buzz(list(data)) if args.buzz else {}

    idx_ret20 = None
    bench_name = "SET"
    if idx_close is not None and len(idx_close) > 21:
        idx_ret20 = float(idx_close.iloc[-1] / idx_close.iloc[-21] - 1)
    else:
        # ดึงดัชนีไม่ได้: ใช้ค่ากลางผลตอบแทน 20 วันของหุ้นที่สแกนแทน
        rets = []
        for df in data.values():
            c = df["Close"].dropna()
            if len(c) > 21:
                rets.append(float(c.iloc[-1] / c.iloc[-21] - 1))
        if rets:
            idx_ret20 = float(np.median(rets))
            bench_name = "ค่ากลางกลุ่ม"
            print("ใช้ค่ากลางของหุ้นที่สแกนแทนดัชนี SET")

    # ถ้าแท่งล่าสุดคือวันนี้และตลาดยังไม่ปิด ให้ประมาณการวอลุ่มทั้งวัน
    vol_frac = 1.0
    if not args.demo and data:
        last_day = max(df.index[-1] for df in data.values())
        if pd.Timestamp(last_day).date() == now_bkk().date():
            vol_frac = session_fraction()
            if vol_frac < 1.0:
                print(f"ตลาดยังเปิดอยู่ (ผ่านไป {vol_frac:.0%}) ประมาณการวอลุ่มทั้งวัน")

    rows = []
    for sym, df in data.items():
        try:
            r = analyze(sym, df, idx_ret20, buzz.get(sym), bench_name, vol_frac)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"วิเคราะห์ {sym} ไม่ได้: {e}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    if args.top > 0:
        rows = rows[: args.top]

    html = build_html(rows, len(data), args.demo, args.refresh)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"วิเคราะห์ได้ {len(rows)} ตัว")
    print(f"สร้างหน้าเว็บแล้ว: {os.path.abspath(args.out)}")

    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        # แสดงใน iframe แยก เพื่อให้สคริปต์ของหน้าเว็บทำงานแน่นอน
        import html as _h
        from IPython.display import HTML, display
        display(HTML(
            '<iframe srcdoc="' + _h.escape(html, quote=True) + '" '
            'style="width:100%;height:1400px;border:0;border-radius:12px"></iframe>'
        ))
    elif not args.no_open:
        webbrowser.open("file://" + os.path.abspath(args.out))


if __name__ == "__main__":
    main()
