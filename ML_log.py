#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_es_iforest_per_source.py
- Elasticsearch의 분류별 인덱스(auth/syslog/cron/dpkg/kern)를 기간으로 수집
- 각 분류별 Isolation Forest 모델을 따로 학습해 하나의 pack으로 저장
"""

import argparse, json, re, sys
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from joblib import dump

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

    p.add_argument("--time-from", default="now-14d")
    p.add_argument("--time-to",   default="now")
    p.add_argument("--size", type=int, default=1000)
    p.add_argument("--max-docs", type=int, default=1_000_000)

    p.add_argument("--model-pack", default="models_iforest.joblib")
    p.add_argument("--thresholds", default="thresholds.json")
    p.add_argument("--contam", type=float, default=0.005)
    p.add_argument("--val-q", type=float, default=0.995)

    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--max-df", type=float, default=0.5)
    p.add_argument("--max-features", type=int, default=20000)
    return p.parse_args()

def es_client(args):
    if args.es_api_key:
        return Elasticsearch(args.es_url, api_key=args.es_api_key)
    if args.es_user:
        return Elasticsearch(args.es_url, basic_auth=(args.es_user, args.es_pass))
    return Elasticsearch(args.es_url)

FIELDS = [
    "@timestamp","message",
    "host.name","host.hostname","host.ip",
    "process.name","log.logger","syslog.program",
    "log.file.path","event.dataset"
]

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
    host = _first(s.get("host",{}).get("name"), s.get("host",{}).get("hostname"), s.get("host",{}).get("ip"))
    prog = _first(s.get("process",{}).get("name"), s.get("log",{}).get("logger"), s.get("syslog",{}).get("program")) or "NA"
    return dict(timestamp=ts.to_pydatetime(), source=source_label, host=host or "NA", program=prog, msg=msg)

def es_scan(es, index, tf, tt, size):
    q = {"bool":{"filter":[{"range":{"@timestamp":{"gte":tf,"lte":tt}}}]}}
    return helpers.scan(es, index=index, size=size, query={"_source":FIELDS, "query":q}, preserve_order=False, raise_on_error=False)

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

KW_LIST = [
    "failed","invalid user","refused","error","critical","segfault","timeout","disconnect","denied","oom","panic",
    "permission denied","broken pipe","unreachable",
    "sshd","authentication failure","pam_unix","sudo","session opened","session closed",
    "cron","crond","dpkg","install ","remove ","upgrade ","kernel","soft lockup"
]

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

def time_split(df, a=0.7, b=0.2):
    if len(df)<10: return df, df, df
    q1 = df["timestamp"].quantile(a); q2 = df["timestamp"].quantile(a+b)
    tr = df[df["timestamp"]<q1]; va = df[(df["timestamp"]>=q1)&(df["timestamp"]<q2)]; te = df[df["timestamp"]>=q2]
    return tr, va, te

def build_pipeline(tfidf_params, kw_cols, contamination):
    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(ngram_range=(1,2),
                                min_df=tfidf_params["min_df"],
                                max_df=tfidf_params["max_df"],
                                max_features=tfidf_params["max_features"]), "msg"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["program"]),   # host는 식별용이므로 피처에선 제외(과적합 방지)
        ("num", StandardScaler(with_mean=False),
         ["hour_sin","hour_cos","weekday","msg_len","digit_ratio"] + list(kw_cols))
    ], sparse_threshold=0.3)

    iso = IsolationForest(n_estimators=400, max_samples=4096,
                          contamination=contamination, bootstrap=True,
                          n_jobs=-1, random_state=42)
    return Pipeline([("prep", pre), ("iso", iso)])

def main():
    args = get_args()
    es = es_client(args)
    idx_map = dict(auth=args.auth_index, syslog=args.syslog_index, cron=args.cron_index, dpkg=args.dpkg_index, kern=args.kern_index)

    models={}, {}
    models_dict={}
    thresholds={}
    pack = {"kw_list": KW_LIST, "models": {}}

    for s in SOURCES:
        print(f"[*] load {s}…")
        df = load_group(es, idx_map[s], s, args).sort_values("timestamp")
        if df.empty:
            print(f"  - no data for {s}, skip")
            continue
        df_feat = featurize(df, KW_LIST)
        train, val, test = time_split(df_feat)
        kw_cols = [c for c in df_feat.columns if c.startswith("kw_")]
        tfidf_params = dict(min_df=args.min_df, max_df=args.max_df, max_features=args.max_features)
        print(f"  - train={len(train)}, val={len(val)}, test={len(test)}")

        pipe = build_pipeline(tfidf_params, kw_cols, args.contam)
        pipe.fit(train)
        val_scores = -pipe.decision_function(val if len(val) else train)
        thr = float(np.quantile(val_scores, args.val_q)) if len(val_scores) else float(np.quantile(-pipe.decision_function(train), args.val_q))

        pack["models"][s] = pipe
        thresholds[s] = thr
        print(f"  - {s}: thr={thr:.6f}")

    if not pack["models"]:
        raise SystemExit("no models trained (no data). check indices/time range.")

    print(f"[*] save {args.model_pack}")
    dump(pack, args.model_pack)
    print(f"[*] save thresholds → {args.thresholds}")
    with open(args.thresholds,"w",encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

if __name__=="__main__":
    main()
