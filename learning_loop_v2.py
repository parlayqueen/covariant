#!/usr/bin/env python3
from __future__ import annotations

import json, math, os, time, glob, random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

RUNS = Path("runs")
PARAMS_PATH = RUNS / "learn_params.json"
REPORT_PATH = RUNS / "learn_report.json"

# -------------------- math --------------------
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def logit(p: float) -> float:
    eps = 1e-9
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p/(1.0-p))

def coerce_prob(v: Any) -> Optional[float]:
    try:
        if v is None: return None
        if isinstance(v, str):
            t = v.strip().replace("%","")
            if t == "": return None
            v = float(t)
        if isinstance(v, (int,float)):
            v = float(v)
            if v > 1.0 and v <= 100.0:
                v /= 100.0
            if 0.0 <= v <= 1.0:
                return v
    except Exception:
        return None
    return None

def logloss(ps: List[float], ys: List[int]) -> float:
    eps = 1e-9
    s = 0.0
    for p,y in zip(ps,ys):
        p = min(max(p, eps), 1.0-eps)
        s += -(y*math.log(p) + (1-y)*math.log(1-p))
    return s / max(1,len(ps))

def brier(ps: List[float], ys: List[int]) -> float:
    return sum((p-y)**2 for p,y in zip(ps,ys)) / max(1,len(ps))

# -------------------- schema extraction --------------------
PROB_KEYS = ("model_prob","model","p_model","prob","p","model_probability")
MARKET_KEYS = ("market","bet_type","type","wager_type")
BOOK_KEYS = ("book","bookmaker","sportsbook","source_book")

def extract_label(pick: Dict[str,Any]) -> Optional[int]:
    # Map many common representations to y in {0,1}
    v = None
    for k in ("y","label","won","win","is_win","result","outcome","grade"):
        if k in pick:
            v = pick.get(k)
            break
    if v is None:
        return None
    # bools
    if isinstance(v,bool):
        return 1 if v else 0
    # ints/floats
    if isinstance(v,(int,float)):
        iv = int(v)
        if iv in (0,1): return iv
        return None
    # strings
    if isinstance(v,str):
        t = v.strip().lower()
        # wins
        if t in ("w","win","won","1","true","✅","success"):
            return 1
        # losses
        if t in ("l","loss","lost","0","false","❌","fail","failed"):
            return 0
        # pushes/void/no action => skip
        if t in ("p","push","void","no_action","na","n/a","canceled","cancelled","pending",""):
            return None
    return None


def extract_prob(pick: Dict[str,Any]) -> Optional[float]:
    # Accept your schema: model_prob (0..1), model_pct (0..100), implied_prob, implied_pct
    for k in ("p","prob","p_model","model_prob","implied_prob","calibrated_p","p_cal"):
        v = pick.get(k)
        if isinstance(v,(int,float)):
            p=float(v)
            if 0.0 <= p <= 1.0: return p
    for k in ("model_pct","implied_pct","prob_pct","p_pct"):
        v = pick.get(k)
        if isinstance(v,(int,float)):
            p=float(v)/100.0
            if 0.0 <= p <= 1.0: return p
    # Sometimes nested
    for k in ("model","prediction","pred"):
        v = pick.get(k)
        if isinstance(v,dict):
            for kk in ("prob","p","model_prob","p_model"):
                vv=v.get(kk)
                if isinstance(vv,(int,float)):
                    p=float(vv)
                    if 0.0 <= p <= 1.0: return p
    return None


def extract_str(pick: Dict[str,Any], keys: Tuple[str,...]) -> Optional[str]:
    for k in keys:
        if k in pick and isinstance(pick[k], str):
            t = pick[k].strip()
            if t: return t
    return None

def iter_picks(obj: Any) -> List[Dict[str,Any]]:
    if isinstance(obj, dict):
        if isinstance(obj.get("picks"), list):
            return [x for x in obj["picks"] if isinstance(x,dict)]
        if isinstance(obj.get("singles"), list):
            return [x for x in obj["singles"] if isinstance(x,dict)]
        if isinstance(obj.get("singles"), dict) and isinstance(obj["singles"].get("picks"), list):
            return [x for x in obj["singles"]["picks"] if isinstance(x,dict)]
        out=[]
        for v in obj.values():
            if isinstance(v,list) and v and all(isinstance(x,dict) for x in v):
                out.extend(v)
        return out
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x,dict)]
    return []

# -------------------- data collection --------------------
def latest_files(patterns: List[str], limit: int) -> List[str]:
    cands=[]
    for pat in patterns:
        cands.extend(glob.glob(str(RUNS / pat)))
    cands = sorted(set(cands), key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[:limit]

def load_dataset(max_files: int = 25) -> List[Dict[str,Any]]:
    files = latest_files(["graded_picks_full_*.json","graded_picks_*.json"], max_files)
    rows=[]
    for fp in files:
        try:
            obj = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue
        for pick in iter_picks(obj):
            y = extract_label(pick)
            p = extract_prob(pick)
            if y is None or p is None: 
                continue
            market = extract_str(pick, MARKET_KEYS) or "unknown_market"
            book = extract_str(pick, BOOK_KEYS) or "unknown_book"
            rows.append({"p": p, "y": y, "market": market, "book": book, "src": Path(fp).name})
    return rows

# -------------------- fitting: platt via SGD on logloss --------------------
def fit_platt(rows: List[Dict[str,Any]], iters: int = 800, lr: float = 0.05, l2: float = 0.001) -> Tuple[float,float]:
    # p' = sigmoid(a*logit(p) + b)
    a,b = 1.0,0.0
    xs=[logit(r["p"]) for r in rows]
    ys=[r["y"] for r in rows]
    n=max(1,len(xs))
    for _ in range(iters):
        da=db=0.0
        for x,y in zip(xs,ys):
            z=a*x+b
            p2=sigmoid(z)
            da += (p2 - y) * x
            db += (p2 - y)
        # L2 regularization toward (1,0)
        da = da/n + l2*(a-1.0)
        db = db/n + l2*(b-0.0)
        a -= lr*da
        b -= lr*db
    return a,b

def apply_platt(p: float, a: float, b: float) -> float:
    return sigmoid(a*logit(p)+b)

def shrink_to_global(a_g: float, b_g: float, n_g: int, a0: float, b0: float, k: float) -> Tuple[float,float]:
    # alpha ~ how much we trust group vs global
    alpha = n_g / (n_g + k)
    return (alpha*a_g + (1-alpha)*a0, alpha*b_g + (1-alpha)*b0)

# -------------------- evaluation & reliability --------------------
def reliability_bins(ps: List[float], ys: List[int], bins: int = 10) -> List[Dict[str,Any]]:
    # returns list of {bin, n, p_mean, y_mean}
    out=[]
    pairs = sorted(zip(ps,ys), key=lambda t: t[0])
    if not pairs: return out
    # equal-count bins
    step = max(1, len(pairs)//bins)
    for i in range(0, len(pairs), step):
        chunk = pairs[i:i+step]
        p_mean = sum(p for p,_ in chunk)/len(chunk)
        y_mean = sum(y for _,y in chunk)/len(chunk)
        out.append({"bin": len(out)+1, "n": len(chunk), "p_mean": p_mean, "y_mean": y_mean, "gap": (p_mean - y_mean)})
    return out

def split_train_holdout(rows: List[Dict[str,Any]], holdout_frac: float = 0.2, seed: int = 1337):
    r = rows[:]
    random.Random(seed).shuffle(r)
    h = max(1, int(len(r)*holdout_frac))
    return r[h:], r[:h]

# -------------------- main loop --------------------
def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(max_files=40)

    if len(rows) < 30:
        print(json.dumps({"ok": False, "error": "Not enough labeled samples (need >= 30).", "labeled": len(rows)}, indent=2))
        raise SystemExit(2)

    train, hold = split_train_holdout(rows, holdout_frac=0.2)

    # GLOBAL fit
    a0,b0 = fit_platt(train)
    # group fits
    by_market: Dict[str,List[Dict[str,Any]]] = {}
    by_book: Dict[str,List[Dict[str,Any]]] = {}

    for r in train:
        by_market.setdefault(r["market"], []).append(r)
        by_book.setdefault(r["book"], []).append(r)

    # shrinkage strength (bigger = more conservative group params)
    K_MARKET = 200.0
    K_BOOK   = 250.0

    market_models={}
    for m, rr in by_market.items():
        if len(rr) < 25: 
            continue
        am,bm = fit_platt(rr)
        am,bm = shrink_to_global(am,bm,len(rr),a0,b0,K_MARKET)
        market_models[m] = {"a": am, "b": bm, "n": len(rr)}

    book_models={}
    for bk, rr in by_book.items():
        if len(rr) < 30:
            continue
        ab,bb = fit_platt(rr)
        ab,bb = shrink_to_global(ab,bb,len(rr),a0,b0,K_BOOK)
        book_models[bk] = {"a": ab, "b": bb, "n": len(rr)}

    # choose how to apply: market overrides book overrides global (simple, robust)
    def calibrate(p: float, market: str, book: str) -> float:
        if market in market_models:
            mm = market_models[market]
            return apply_platt(p, mm["a"], mm["b"])
        if book in book_models:
            bm = book_models[book]
            return apply_platt(p, bm["a"], bm["b"])
        return apply_platt(p, a0, b0)

    # evaluate on holdout
    p_raw  = [r["p"] for r in hold]
    y_hold = [r["y"] for r in hold]
    p_cal  = [calibrate(r["p"], r["market"], r["book"]) for r in hold]

    metrics = {
        "holdout": {
            "n": len(hold),
            "logloss_raw": logloss(p_raw, y_hold),
            "logloss_cal": logloss(p_cal, y_hold),
            "brier_raw": brier(p_raw, y_hold),
            "brier_cal": brier(p_cal, y_hold),
        }
    }

    # drift gate: only deploy if logloss improves OR doesn't worsen more than tiny epsilon
    eps = 0.0005
    deploy_ok = metrics["holdout"]["logloss_cal"] <= (metrics["holdout"]["logloss_raw"] + eps)

    params = {
        "ok": True,
        "ts": int(time.time()),
        "labeled_total": len(rows),
        "train_n": len(train),
        "holdout_n": len(hold),
        "deploy_ok": deploy_ok,
        "strategy": "hier_platt_shrinkage",
        "global": {"a": a0, "b": b0},
        "by_market": market_models,
        "by_book": book_models,
        "thresholds": {"min_market_n": 25, "min_book_n": 30, "K_market": K_MARKET, "K_book": K_BOOK, "eps_gate": eps},
        "metrics": metrics,
    }

    report = {
        "ok": True,
        "ts": params["ts"],
        "deploy_ok": deploy_ok,
        "metrics": metrics,
        "reliability_holdout_raw": reliability_bins(p_raw, y_hold, bins=10),
        "reliability_holdout_cal": reliability_bins(p_cal, y_hold, bins=10),
        "top_markets": sorted(
            [{"market": k, "n": v["n"], "a": v["a"], "b": v["b"]} for k,v in market_models.items()],
            key=lambda x: x["n"],
            reverse=True
        )[:12],
        "top_books": sorted(
            [{"book": k, "n": v["n"], "a": v["a"], "b": v["b"]} for k,v in book_models.items()],
            key=lambda x: x["n"],
            reverse=True
        )[:12],
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if deploy_ok:
        PARAMS_PATH.write_text(json.dumps(params, indent=2), encoding="utf-8")
        out = {"ok": True, "deployed": True, "learn_params": str(PARAMS_PATH), "learn_report": str(REPORT_PATH), **metrics["holdout"]}
    else:
        out = {"ok": True, "deployed": False, "learn_report": str(REPORT_PATH), "reason": "gate_blocked", **metrics["holdout"]}

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
