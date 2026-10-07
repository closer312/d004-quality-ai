import pandas as pd
from config import OUT, RES, REGION
from qc import run

raw = pd.read_pickle(OUT / "raw.pkl")
p, sig, sc, ref, iso = run(raw)
pd.to_pickle({"ref": ref, "iso": iso}, OUT / "models.pkl")
sig.to_csv(RES / "signals.csv.gz", index=False)
sc.to_csv(RES / "risk_scores.csv.gz", index=False)
sig[sig.layer == "auto"][["NOMER", "q", "code", "text"]].to_csv(RES / "auto_corrections.csv", index=False, encoding="utf-8-sig")

n = len(sc)
summary = sig.groupby(["layer", "code"]).agg(units=("NOMER", lambda s: len(set(zip(s, sig.loc[s.index, "q"]))))).reset_index()
summary["share_pct"] = (summary["units"] / n * 100).round(2)
summary.to_csv(RES / "t1_signals_by_rule.csv", index=False)
lay = sc.assign(flagged=sc.risk > 0).groupby("layers", dropna=False).size().rename("units").reset_index()
lay.to_csv(RES / "t2_layer_overlap.csv", index=False)
print(summary.to_string()); print(lay.to_string())
print("units:", n, "queue (risk>0):", (sc.risk > 0).sum(), "auto-fixed:", (sc.auto_fixes > 0).sum(), "risk>=50:", (sc.risk >= 50).sum())
