import glob
import numpy as np
import pandas as pd
from config import DATA_DIR, OUT, YEAR, QUARTERS, KODNU12, INCOME, read

num = lambda s: pd.to_numeric(s, errors="coerce")


def item_names():
    books = {1: "kv_vopr1 (непроды)", 2: "kv_vopr2(коммунальные)", 3: "kv_vopr3(связь)",
             4: "kv_vopr4(образование)", 5: "kv_vopr5(здравоохранение)", 6: "kv_vopr6*", 7: "kv_vopr7(транспорт)"}
    rows = []
    for t, pat in books.items():
        for f in glob.glob(str(DATA_DIR / f"documentation/d004/2023/коды/{pat}.xls")):
            x = pd.read_excel(f, dtype=str)
            x.columns = [str(c).strip().lower() for c in x.columns]
            if {"kodnu", "name"} <= set(x.columns):
                rows.append(x[["kodnu", "name"]].assign(table=t))
    d = pd.concat(rows).dropna()
    d["kodnu"] = d["kodnu"].str.strip()
    return d.drop_duplicates(["table", "kodnu"])


def main():
    hh, members, items, v12, v9, s4 = [], [], [], [], [], []
    for q in QUARTERS:
        base = f"d004/{YEAR}/{q}"
        qn = int(q[0])
        hh.append(read(f"{base}/kv_vopr0.csv")[["NOMER", "TE", "K"]].assign(q=qn))
        m = read(f"{base}/kv_vopr11.csv")
        g = m[[f"GR{k}" for k in INCOME]].apply(num).fillna(0)
        g.columns = list(INCOME.values())
        g.insert(0, "sex", m["POL"])
        g.insert(0, "age", num(m["VOZ"]))
        g.insert(0, "NOMP", num(m["NOMP"]).astype("Int64"))
        g.insert(0, "NOMER", m["NOMER"])
        members.append(g.assign(q=qn))
        for t in range(1, 8):
            cols = ["NOMER", "KODNU", "STOIMK"] + (["CELPOK", "MESTPOK"] if t == 1 else [])
            d = read(f"{base}/kv_vopr{t}.csv", usecols=cols)
            items.append(d.assign(table=t, q=qn, STOIMK=num(d["STOIMK"])))
        a = read(f"{base}/kv_vopr9.csv", usecols=["NOMER", "NOMVOPR", "NOMOTV", "GR7", "GR8"])
        v9.append(a[["NOMER", "NOMVOPR", "NOMOTV"]].assign(q=qn))
        # Section 4 sales, tenge: 4.5 crops and 4.11 animal products (gr. 8-9 of the form = GR7-GR8),
        # 4.12 goods produced by the household (kv_vopr10: GR2 sold, GR3 sold from earlier output)
        b10 = read(f"{base}/kv_vopr10.csv", usecols=["NOMER", "NOMVOPR", "GR2", "GR3"])
        sales = pd.concat([
            a[a.NOMVOPR == "95"].assign(kind="crop", v=num(a.GR7).fillna(0) + num(a.GR8).fillna(0)),
            a[a.NOMVOPR == "914"].assign(kind="animal", v=num(a.GR7).fillna(0) + num(a.GR8).fillna(0)),
            b10[b10.NOMVOPR == "101"].assign(kind="goods", v=num(b10.GR2).fillna(0) + num(b10.GR3).fillna(0))])
        s4.append(sales.pivot_table(index="NOMER", columns="kind", values="v", aggfunc="sum")
                  .reindex(columns=["crop", "animal", "goods"]).fillna(0).reset_index().assign(q=qn))
        b = read(f"{base}/kv_vopr12.csv")
        v12.append(b[["NOMER", "KODNU", "NOMVNEW", "KOD", "STOIMK"]].assign(q=qn))
        print(q, "ok")
    hh, members, items, v12, v9, s4 = (pd.concat(x, ignore_index=True) for x in (hh, members, items, v12, v9, s4))
    items["KODNU"] = items["KODNU"].str.strip()

    r = read(f"d008/{YEAR}/kontr_k.csv")
    roster = pd.DataFrame({"NOMER": r["NOMER"], "NOMP": num(r["NOMP"]).astype("Int64"),
                           "age": YEAR - num(r["GOD_ROJD"]), "kol_chl": num(r["KOL_CHL"]), "sex": r["POL"]})

    names = item_names()
    pd.to_pickle({"hh": hh, "members": members, "items": items, "v12": v12, "v9": v9, "s4": s4,
                  "roster": roster, "names": names}, OUT / "raw.pkl")
    print({k: len(v) for k, v in [("hh", hh), ("members", members), ("items", items),
                                  ("v12", v12), ("roster", roster), ("names", names)]})
    print("items with a name:", round(items.merge(names, left_on=["table", "KODNU"],
          right_on=["table", "kodnu"], how="left")["name"].notna().mean(), 3))


if __name__ == "__main__":
    main()
