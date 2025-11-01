#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_es_iforest_per_source.py
- 분류별 모델 pack + 분류별 임계값을 로드
- 각 분류 인덱스에서 최근 로그를 수집하여 분류별로 점수/라벨
- 어느 서버(host)에서 언제 어떤 message가 이상인지 CSV/ES로 출력
"""

import argparse, json, re, sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from joblib import load

SOURCES = ["auth","syslog","cron","dpkg","kern"]

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--es-url", required=True)
    p.add_argument("--es-user", default=None)
    p.add_argument("--es-pass", default=None)
    p.add_argument("--es-api-key", default=None)

    p.add_argument("--auth-index", default="auth-*")
    p.add_argument("--syslog-index", default="syslog-*")
    p.add_argument("--cron-index", default="cron-*")
    p.add_argument("--dpkg-index", default="dpkg-*")
    p.add_argument("--kern-index", default="kern-*")

    p.add_argument("--time-from", default="now-1d")
    p.add_argument("--time-to",   default="now")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--max-docs", type=int, default=1_000_000)

    p.add_argument("--model-pack", default="models_iforest.joblib")
    p.add_argument("--thresholds", default="thresholds.json")

    p.add_argument("--out", default="anomalies.csv")
    p.add_argument("--top", type=int, default=200)

    p.add_argument("--es-index-out", default=None)
    return p.parse_args()

def es_client(args):
    if args.es_api_key:
        return Elasticsearch(args.es_url, api_key=args.es_api_key)
    if args.es_user:
        return Elasticsearch(args.es_url, basic_auth=(args.es_user, args.es_pass))
    return Elasticsearch(args.es_url)

FIELDS = ["@timestamp","message","host.name","host.hostname","host.ip","process.name","log.logger","syslog.program"]

def _first(*vals):
    for v in vals:
        if v is None: continue
        if isinstance(v, list) and v: return v[0]
        if isinstance(v, str) and v.strip(): return v
        if not isinstance(v, (list, str)) and v: return v
    return None

def extract(hit, source_label):
    s = hit.get("_source",{}) or {}
    ts = _first(s.get("@timestamp"))
    try:
        ts = pd.to_datetime(ts, utc=True).tz_convert(None)
    except Exception:
        ts = pd.Timestamp.utcnow()
    msg = _first(s.get("message"), s.get("log",{}).get("original")) or ""
    host = _first(s.get("host",{}).get("name"), s.get("host",{}).get("hostname"), s.get("host",{}).get("ip")) or "NA"
    prog = _first(s.get("process",{}).get("name"), s.get("log",{}).get("logger"), s.get("syslog",{}).get("program")) or "NA"
    return dict(timestamp=ts.to_pydatetime(), source=source_label, host=host, program=prog, msg=msg)

def es_scan(es, index, tf, tt, size):
    q = {"bool":{"filter":[{"range":{"@timestamp":{"gte":tf,"lte":tt}}}]}}
    return helpers.scan(es, index=index, size=size, query={"_source":FIELDS,"query":q}, preserve_order=False, raise_on_error=False)

def load_group(es, idx_pat, label, args):
    rows=[]
    for pat in [x.strip() for x in idx_pat.split(",") if x.strip()]:
        try:
            for hit in es_scan(es, pat, args.time_from, args.time_to, args.size):
                rows.append(extract(hit, label))
                if len(rows)>=args.max_docs: break
        except Exception as e:
            print(f"[warn] {label}({pat}) load err: {e}", file=sys.stderr)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","source","host","program","msg"])

def featurize(df: pd.DataFrame, kw_list):
    df=df.copy()
    t = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2*np.pi*t.dt.hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*t.dt.hour/24.0)
    df["weekday"]  = t.dt.weekday
    msg = df["msg"].fillna("")
    df["msg_len"] = msg.str.len()
    df["digit_ratio"]= msg.str.count(r"\d").div(df["msg_len"].replace(0,1))
    low = msg.str.lower()
    for kw in kw_list:
        col = "kw_"+re.sub(r"[^a-z0-9]+","_",kw)
        df[col]= low.str.contains(re.escape(kw), na=False).astype(int)
    for c in ["host","program"]:
        df[c]=df[c].astype(str).fillna("NA").replace({"nan":"NA"})
    return df

def maybe_index_results(es, index_name, df):
    if not index_name or df.empty: return
    actions = (
        {"_index": index_name,
         "_source":{
           "@timestamp": pd.to_datetime(r["timestamp"]).isoformat(),
           "source": r["source"], "host": r["host"], "program": r["program"],
           "message": r["msg"], "anomaly_score": float(r["anomaly_score"]),
           "anomaly": int(r["anomaly"]), "model": "iforest"
         }}
        for _, r in df.iterrows()
    )
    helpers.bulk(es, actions)

def main():
    args = get_args()
    pack = load(args.model_pack)     # {"kw_list": [...], "models": {"auth":pipe,...}}
    kw_list = pack["kw_list"]; models = pack["models"]
    with open(args.thresholds,"r",encoding="utf-8") as f:
        thrs = json.load(f)

    es = es_client(args)
    idx_map = dict(auth=args.auth_index, syslog=args.syslog_index, cron=args.cron_index, dpkg=args.dpkg_index, kern=args.kern_index)

    all_rows=[]
    for s in SOURCES:
        if s not in models: 
            print(f"[i] skip {s} (no model)")
            continue
        print(f"[*] detecting {s}…")
        df = load_group(es, idx_map[s], s, args)
        if df.empty:
            print(f"  - no data")
            continue
        df_feat = featurize(df, kw_list)
        pipe = models[s]; thr = float(thrs.get(s, 0.0))
        scores = -pipe.decision_function(df_feat)
        df["anomaly_score"]=scores
        df["anomaly"]=(scores>=thr).astype(int)
        all_rows.append(df)

    if not all_rows:
        print("no data detected.")
        return

    out = pd.concat(all_rows, ignore_index=True)
    out_sorted = out.sort_values("anomaly_score", ascending=False)
    top = out_sorted.head(args.top)
    cols = ["timestamp","source","host","program","msg","anomaly_score","anomaly"]
    top.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[+] saved top {len(top)} anomalies → {args.out}")

    if args.es_index_out:
        print(f"[*] indexing results → {args.es_index_out}")
        maybe_index_results(es, args.es_index_out, top[cols].rename(columns={"msg":"message"}))

    # preview
    print("\n=== PREVIEW ===")
    for _, r in top.head(10).iterrows():
        ts = pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{r['anomaly_score']:.4f}] {ts} host={r['host']} src={r['source']} prog={r['program']} :: {(str(r['msg']) or '')[:140]}")

if __name__=="__main__":
    main()
