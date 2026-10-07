import copy
import numpy as np
import pandas as pd
from config import OUT, RES, SEED
from qc import run, KEY, INC_COLS

ERRORS = {
    "E1_extra_zero": "лишний ноль в сумме покупки (×10)",
    "E2_thousands": "зарплата записана в тысячах тенге",
    "E3_lost_income": "пропущены все доходы семьи",
    "E4_wrong_column": "зарплата записана в графу «пенсия»",
    "E5_double_entry": "раздел расходов введён дважды",
    "E6_orphan_answers": "ответы о кредите без суммы кредита",
    "E7_lost_spending": "пропущен весь раздел расходов",
    "E8_debt_unfunded": "погашение кредита без источника денег",
    "E9_skip_logic": "ответы на вопросы, которые бланк велит пропустить",
    "E1_big": "лишний ноль, увеличивший расходы семьи на 20% и более",
}
N_PER_TYPE = 150
BUDGETS = [0.02, 0.05, 0.10]


def inject(raw, clean_units, seed):
    rng = np.random.default_rng(seed)
    r = copy.deepcopy(raw)
    m, it, v = r["members"], r["items"], r["v12"]
    used, truth = set(), []

    def pick(cands, n=N_PER_TYPE):
        cands = [u for u in cands if u not in used and u in clean_units]
        sel = [cands[i] for i in rng.choice(len(cands), size=min(n, len(cands)), replace=False)]
        used.update(sel)
        return sel

    units_items = list(it.groupby(KEY).size().index)
    units_wage = list(m[m.wage > 0].groupby(KEY).size().index)
    units_young = list(m[(m.wage > 0) & m.age.between(18, 39)].groupby(KEY).size().index)
    units_no12 = sorted(set(r["hh"].set_index(KEY).index) - set(v.set_index(KEY).index))
    units_inc = list(m[m[INC_COLS].sum(axis=1) > 0].groupby(KEY).size().index)
    units_both = sorted(set(units_items) & set(units_inc))

    it_idx = it.set_index(KEY).index
    m_idx = m.set_index(KEY).index

    for u in pick(units_items):
        rows = np.where(it_idx == u)[0]
        i = rows[rng.integers(len(rows))]
        extra = it.at[i, "STOIMK"] * 9
        it.at[i, "STOIMK"] *= 10
        truth.append((*u, "E1_extra_zero", extra / max(it.loc[rows, "STOIMK"].sum(), 1)))
    for u in pick(units_wage):
        rows = [i for i in np.where(m_idx == u)[0] if m.at[i, "wage"] > 0]
        i = rows[rng.integers(len(rows))]
        m.at[i, "wage"] = round(m.at[i, "wage"] / 1000)
        truth.append((*u, "E2_thousands"))
    for u in pick(units_both):
        m.loc[np.where(m_idx == u)[0], INC_COLS] = 0
        truth.append((*u, "E3_lost_income"))
    for u in pick(units_young):
        rows = [i for i in np.where(m_idx == u)[0] if m.at[i, "wage"] > 0 and 18 <= m.at[i, "age"] <= 39]
        i = rows[rng.integers(len(rows))]
        m.at[i, "pension"], m.at[i, "wage"] = m.at[i, "wage"], 0
        truth.append((*u, "E4_wrong_column"))
    dup_units = pick([u for u in units_items])
    for u in dup_units:
        truth.append((*u, "E5_double_entry"))
    new_items = it[it_idx.isin(dup_units)]
    for u in pick(units_no12):
        v.loc[len(v)] = {"NOMER": u[0], "KODNU": np.nan, "NOMVNEW": "21", "KOD": str(rng.integers(1, 10)),
                         "STOIMK": np.nan, "q": u[1]}
        truth.append((*u, "E6_orphan_answers"))
    drop_units = pick(units_both)
    for u in drop_units:
        truth.append((*u, "E7_lost_spending"))
    prof = r["hh"][KEY].merge(m.groupby(KEY)[INC_COLS].sum().sum(axis=1).rename("inc").reset_index(), on=KEY, how="left").fillna(0)
    inc = dict(zip(map(tuple, prof[KEY].values.tolist()), prof["inc"]))
    for u in pick(units_no12):
        v.loc[len(v)] = {"NOMER": u[0], "KODNU": "2.1.1.4.1", "NOMVNEW": "20", "KOD": np.nan,
                         "STOIMK": str(int(max(3 * inc.get(u, 0), 300000))), "q": u[1]}
        truth.append((*u, "E8_debt_unfunded"))
    v9 = r["v9"]
    no_land = v9[(v9.NOMVOPR == "91") & (v9.NOMOTV == "2")]
    has94 = set(map(tuple, v9.loc[v9.NOMVOPR.isin(["92", "93", "94"]), KEY].values.tolist()))
    cands = [u for u in map(tuple, no_land[KEY].values.tolist()) if u not in has94]
    for u in pick(cands):
        v9.loc[len(v9)] = {"NOMER": u[0], "NOMVOPR": "94", "NOMOTV": "1", "q": u[1]}
        truth.append((*u, "E9_skip_logic"))
    it = pd.concat([it[~it_idx.isin(drop_units)], new_items], ignore_index=True)
    r["items"] = it
    truth = [t if len(t) == 4 else (*t, np.nan) for t in truth]
    return r, pd.DataFrame(truth, columns=["NOMER", "q", "error", "impact"])


def evaluate(sc, sig, truth):
    n = len(sc)
    sc = sc.reset_index(drop=True).assign(rank=lambda d: np.arange(len(d)))
    t = truth.merge(sc[KEY + ["rank", "risk"]], on=KEY)
    lay = sig.groupby(KEY)["layer"].agg(set).reset_index()
    t = t.merge(lay, on=KEY, how="left")
    t["layer"] = t["layer"].apply(lambda x: x if isinstance(x, set) else set())
    rows = []
    big = t[(t.error == "E1_extra_zero") & (t.impact >= 0.2)]
    for e, g in list(t.groupby("error")) + [("E1_big", big), ("ALL", t)]:
        row = {"error": e, "n": len(g)}
        auto = g["layer"].apply(lambda s: "auto" in s)
        for b in BUDGETS:
            row[f"recall@{int(b*100)}%"] = ((g["rank"] < b * n) | auto).mean()
        row["auto_fixed"] = auto.mean()
        for L in ("rules", "outliers", "ml", "auto"):
            row[f"caught_by_{L}"] = g["layer"].apply(lambda s: L in s).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    raw = pd.read_pickle(OUT / "raw.pkl")
    models = pd.read_pickle(OUT / "models.pkl")
    clean = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str})
    clean_units = set(map(tuple, clean.loc[(clean.risk == 0) & (clean.auto_fixes == 0), KEY].values.tolist()))
    res, prec = [], []
    for k, seed in enumerate([SEED, SEED + 1, SEED + 2]):
        r, truth = inject(raw, clean_units, seed)
        _, sig, sc, _, _ = run(r, ref=models["ref"], iso_model=models["iso"])
        ev = evaluate(sc, sig, truth).assign(run=k)
        for b in BUDGETS:
            top = sc.head(int(b * len(sc)))
            prec.append({"run": k, "budget": b, "precision": top.merge(truth, on=KEY).shape[0] / len(top),
                         "max_possible": min(1, len(truth) / len(top))})
        res.append(ev)
        if k == 0:
            sig.merge(truth, on=KEY).to_csv(RES / "eval_signals_example.csv.gz", index=False)
        print(ev.round(3).to_string())
    res = pd.concat(res)
    mean = res.groupby("error", sort=False).mean(numeric_only=True).drop(columns="run")
    mean.insert(0, "description", [ERRORS.get(e, "все типы") for e in mean.index])
    mean.round(3).to_csv(RES / "t3_evaluation.csv")
    pd.DataFrame(prec).groupby("budget").mean().drop(columns="run").round(3).to_csv(RES / "t4_precision.csv")
    print(mean.round(3).to_string())


if __name__ == "__main__":
    main()
