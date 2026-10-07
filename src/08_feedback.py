"""Closed loop: the system learns from interviewers' verdicts.

The survey is quarterly. In quarter 1 interviewers recontact the top questionnaires by risk and mark each one
"error confirmed" / "false alarm" (10% of households cannot be reached and give no label). From quarter 2 on,
a classifier trained on all earlier verdicts re-ranks the queue once at least MIN_LABELS verdicts exist. Errors are injected as in 03_evaluate.py,
so the verdicts are known. We compare errors found with the static ranking and with the learned one
at the same number of recontacts."""
import sys
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.ensemble import HistGradientBoostingClassifier
from config import OUT, RES, SEED
from qc import run, KEY

inject = import_module("03_evaluate").inject
BUDGETS = [0.02, 0.05]
NO_CONTACT = 0.10
TRUST_N = 500
MIN_LABELS = 400       # below this the static ranking is kept (cold start)
SEEDS = [SEED, SEED + 1, SEED + 2]


def unit_features(p, sig, sc):
    s = sig[sig.layer != "auto"]
    codes = s.pivot_table(index=KEY, columns="code", values="p", aggfunc="max")
    codes.columns = [f"c_{c}" for c in codes.columns]
    X = p[KEY].merge(sc[KEY + ["risk"]], on=KEY).merge(codes.reset_index(), on=KEY, how="left")
    X["n_signals"] = X.filter(like="c_").notna().sum(axis=1)
    return X.fillna(0)


def one_seed(raw, models, clean_units, seed):
    rng = np.random.default_rng(seed)
    r, truth = inject(raw, clean_units, seed)
    p, sig, sc, _, _ = run(r, ref=models["ref"], iso_model=models["iso"])
    X = unit_features(p, sig, sc)
    key = pd.MultiIndex.from_frame(X[KEY])
    X["y"] = key.isin(pd.MultiIndex.from_frame(truth[KEY])).astype(int)
    X["auto"] = key.isin(pd.MultiIndex.from_frame(sig.loc[sig.layer == "auto", KEY]))
    X["reached"] = rng.random(len(X)) >= NO_CONTACT
    feats = [c for c in X.columns if c not in KEY + ["y", "auto", "reached"]]
    mono = [1 if c == "risk" else 0 for c in feats]
    rows, rule_rows = [], []
    for b in BUDGETS:
        labeled = []
        for q in sorted(X.q.unique()):
            g = X[(X.q == q) & ~X.auto & (X.risk > 0)]
            k = int(b * (X.q == q).sum())
            static = g.sort_values(["risk", "NOMER"], ascending=[False, True], kind="mergesort").head(k)
            train = pd.concat(labeled) if labeled else None
            if train is not None and len(train) >= MIN_LABELS and train.y.nunique() == 2:
                clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_depth=3, min_samples_leaf=20,
                                                     monotonic_cst=mono, random_state=seed)
                clf.fit(train[feats], train.y)
                w = len(train) / (len(train) + TRUST_N)      # trust the verdicts more as they accumulate
                base = g.risk.rank(pct=True)
                learned = g.assign(s=(1 - w) * base + w * pd.Series(clf.predict_proba(g[feats])[:, 1], index=g.index).rank(pct=True)
                                   ).sort_values(["s", "NOMER"], ascending=[False, True], kind="mergesort").head(k)
            else:
                learned = static
            for name, sel in (("static", static), ("learned", learned)):
                found = int((sel.y & sel.reached).sum())
                rows.append({"seed": seed, "budget": b, "q": q, "method": name, "checks": k,
                             "errors_in_quarter": int(g.y.sum()), "found": found, "precision": found / k})
            labeled.append(learned[learned.reached])
        if b == BUDGETS[0]:
            lab = pd.concat(labeled)
            for c in [c for c in feats if c.startswith("c_")]:
                h = lab[lab[c] > 0]
                if len(h) >= 10:
                    rule_rows.append({"seed": seed, "code": c[2:], "checked": len(h), "confirmed": h.y.mean()})
    return pd.DataFrame(rows), pd.DataFrame(rule_rows)


def main():
    seeds = [int(a) for a in sys.argv[1:]] or SEEDS
    raw = pd.read_pickle(OUT / "raw.pkl")
    models = pd.read_pickle(OUT / "models.pkl")
    clean = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str})
    clean_units = set(map(tuple, clean.loc[(clean.risk == 0) & (clean.auto_fixes == 0), KEY].values.tolist()))
    for seed in seeds:
        res, rules = one_seed(raw, models, clean_units, seed)
        res.to_csv(OUT / f"feedback_{seed}.csv", index=False)
        rules.to_csv(OUT / f"feedback_rules_{seed}.csv", index=False)
        print(res.pivot_table(index=["budget", "q"], columns="method", values="found").to_string())
    if all((OUT / f"feedback_{s}.csv").exists() for s in SEEDS):
        res = pd.concat(pd.read_csv(OUT / f"feedback_{s}.csv") for s in SEEDS)
        t = (res.groupby(["budget", "q", "method"])[["checks", "errors_in_quarter", "found", "precision"]]
                .mean().unstack("method"))
        t.columns = [f"{a}_{b}" for a, b in t.columns]
        t = t.drop(columns=[c for c in t.columns if c.startswith(("checks_", "errors_in_quarter_")) and not c.endswith("static")]).rename(
            columns={"checks_static": "checks", "errors_in_quarter_static": "errors_in_quarter"})
        t["gain_learned_pct"] = (t.found_learned / t.found_static - 1) * 100
        t.round(3).to_csv(RES / "t7_feedback_loop.csv")
        rules = pd.concat(pd.read_csv(OUT / f"feedback_rules_{s}.csv") for s in SEEDS)
        rules.groupby("code")[["checked", "confirmed"]].mean().sort_values("confirmed").round(3).to_csv(RES / "t8_rule_confirmation.csv")
        print(t.round(2).to_string())


if __name__ == "__main__":
    main()
