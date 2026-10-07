"""What the system saves: interviewer contacts and distortion of published averages.

Errors of all 9 types are injected into clean questionnaires (as in 03_evaluate.py). For each check budget
we compare three ways to choose questionnaires for recontact: random choice, rules only, the full system.
Errors in chosen questionnaires are corrected (true values restored), then we measure how far the
averages of household income and spending still are from the true ones, overall and by region."""
import numpy as np
import pandas as pd
from config import OUT, RES, SEED, REGION
from qc import run, build_profile, KEY
from importlib import import_module

inject = import_module("03_evaluate").inject
MIN_PER_CHECK = 20          # assumption: minutes per recontact (call + correction)
BUDGETS = np.round(np.arange(0, 0.2001, 0.005), 3)
STATS = ["income", "spending", "wage"]


def distortion(p, true):
    """Distortion of averages vs the true data, % of the true value."""
    out = {}
    for c in STATS:
        out[f"{c}_bias_pct"] = (p[c].mean() / true[c].mean() - 1) * 100
        out[f"{c}_abs_err_pct"] = (p[c] - true[c]).abs().sum() / true[c].sum() * 100
        reg = p.groupby("TE")[c].mean() / true.groupby("TE")[c].mean() - 1
        out[f"{c}_region_mae_pct"] = reg.abs().mean() * 100
        out[f"{c}_region_max_pct"] = reg.abs().max() * 100
    return out


def main():
    raw = pd.read_pickle(OUT / "raw.pkl")
    models = pd.read_pickle(OUT / "models.pkl")
    clean = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str})
    clean_units = set(map(tuple, clean.loc[(clean.risk == 0) & (clean.auto_fixes == 0), KEY].values.tolist()))
    r, truth = inject(raw, clean_units, SEED)
    cols = KEY + ["TE"] + STATS
    true = build_profile(raw)[cols].sort_values(KEY).reset_index(drop=True)
    p_err, sig, sc, _, _ = run(r, ref=models["ref"], iso_model=models["iso"])
    err = p_err[cols].sort_values(KEY).reset_index(drop=True)
    assert (true[KEY].values == err[KEY].values).all()
    n = len(err)
    key = pd.MultiIndex.from_frame(err[KEY])
    bad = key.isin(pd.MultiIndex.from_frame(truth[KEY]))
    auto = key.isin(pd.MultiIndex.from_frame(sig.loc[sig.layer == "auto", KEY]))

    rng = np.random.default_rng(SEED)
    rules = sig[sig.layer == "rules"].groupby(KEY)["p"].apply(lambda x: 1 - np.prod(1 - x.values))
    rank = {
        "random": pd.Series(rng.permutation(n), index=key),
        "rules": pd.Series(rules.reindex(key).fillna(0).values + rng.random(n) * 1e-6, index=key)
                   .rank(ascending=False, method="first") - 1,
        "system": pd.Series(np.arange(len(sc)), index=pd.MultiIndex.from_frame(sc[KEY])).reindex(key),
    }
    rows = []
    for method, rk in rank.items():
        rk = rk.values
        for b in BUDGETS:
            found = (rk < b * n) | (auto if method != "random" else False)
            fixed = err.copy()
            fixed.loc[found & bad, STATS] = true.loc[found & bad, STATS]
            checks = int(b * n)
            rows.append({"method": method, "budget": b, "checks": checks,
                         "hours": checks * MIN_PER_CHECK / 60, "errors": int(bad.sum()),
                         "found": int((found & bad).sum()), "recall": (found & bad).sum() / bad.sum(),
                         **distortion(fixed, true)})
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RES / "t5_impact.csv", index=False)

    # Headline: how many checks each method needs to find 90% of errors
    head = []
    for method, g in res.groupby("method", sort=False):
        hit = g[g.recall >= 0.9]
        b = hit.budget.min() if len(hit) else np.nan
        need = int(b * n) if len(hit) else int(0.9 * n)
        at5 = g[g.budget == 0.05].iloc[0]
        head.append({"method": method, "checks_for_90pct": need, "hours_for_90pct": need * MIN_PER_CHECK / 60,
                     "recall_at_5pct": at5.recall, "spending_abs_err_at_5pct": at5.spending_abs_err_pct,
                     "income_abs_err_at_5pct": at5.income_abs_err_pct,
                     "spending_region_max_at_5pct": at5.spending_region_max_pct})
    head = pd.DataFrame(head)
    base = res[res.budget == 0].iloc[0]
    head["spending_abs_err_no_check"] = base.spending_abs_err_pct
    head["income_abs_err_no_check"] = base.income_abs_err_pct
    head["spending_region_max_no_check"] = base.spending_region_max_pct
    head.round(3).to_csv(RES / "t6_savings.csv", index=False)
    print(head.round(2).to_string())
    print(res[res.budget.isin([0, 0.02, 0.05, 0.1])].round(2).to_string())


if __name__ == "__main__":
    main()
