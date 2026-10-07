import gzip
import html
import json
import pandas as pd
from config import OUT, RES, REGION, TABLE_RU
from qc import tg, card_lines, risk_label

TOP = 50
raw = pd.read_pickle(OUT / "raw.pkl")
sc = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str, "TE": str, "K": str})
sig = pd.read_csv(RES / "signals.csv.gz", dtype={"NOMER": str, "detail": str})
names = raw["names"].set_index(["table", "kodnu"])["name"]
it = raw["items"]
big = it.sort_values("STOIMK", ascending=False).drop_duplicates(["NOMER", "q"]).set_index(["NOMER", "q"])

by_unit = {k: g for k, g in sig.groupby(["NOMER", "q"])}


def card(u):
    s = by_unit.get((u.NOMER, u.q), sig.iloc[:0])
    b = big.loc[(u.NOMER, u.q)] if (u.NOMER, u.q) in big.index else None
    return card_lines(s, u, (names.get((b.table, b.KODNU), b.KODNU), b.STOIMK) if b is not None else None)


rows, cards = [], []
for _, u in sc.head(TOP).iterrows():
    lines = card(u)
    where = f"{REGION.get(u.TE, u.TE)}, {'город' if u.K == '1' else 'село'}, {int(u.persons)} чел."
    rows.append({"NOMER": u.NOMER, "квартал": u.q, "риск": min(99, round(u.risk)), "где": where,
                 "причины": " | ".join(t for _, t, _ in lines), "что проверить": " | ".join(dict.fromkeys(c for *_, c in lines))})
    items = "".join(f"<li><span class='tag {l}'>{html.escape(l)}</span> {html.escape(t)}<br><i>→ {html.escape(c)}</i></li>"
                    for l, t, c in lines)
    cards.append(f"<div class='card'><div class='hd'><b>Домохозяйство {u.NOMER}</b>, {u.q} кв. 2024 · {html.escape(where)}"
                 f" · <span class='risk'>{risk_label(u.risk)}</span></div><ul>{items}</ul></div>")
pd.DataFrame(rows).to_csv(RES / "interviewer_worklist_top50.csv", index=False, encoding="utf-8-sig")
css = """body{font:15px/1.45 system-ui,Segoe UI,sans-serif;max-width:900px;margin:2rem auto;padding:0 16px;background:#fcfcfb;color:#0b0b0b}
.card{border:1px solid #e3e2dc;border-radius:8px;padding:.7em 1em;margin:.8em 0;background:#fff}.hd{display:flex;gap:.6em;flex-wrap:wrap;align-items:baseline}
.risk{margin-left:auto;background:#fdeee8;color:#9a3412;padding:.1em .6em;border-radius:99px;font-weight:600}ul{margin:.4em 0;padding-left:1.1em}li{margin:.35em 0}
i{color:#52514e}.tag{font-size:.75em;padding:.05em .45em;border-radius:4px;background:#eef3fb;color:#1d4f91}"""
(RES / "interviewer_report_top50.html").write_text(
    f"<!doctype html><meta charset='utf-8'><title>Контроль качества Д-004</title><style>{css}</style>"
    f"<h1>Анкеты Д-004 для перепроверки, 2024</h1><p>Топ-{TOP} из {len(sc):,} анкет".replace(",", " ") + f" по оценке риска.</p>{''.join(cards)}",
    encoding="utf-8")
print(open(RES / "interviewer_worklist_top50.csv", encoding="utf-8-sig").read()[:2500])


# all questionnaires in the interviewer queue, for the web app (it needs only results/, not the raw data)
queue = sc[sc.risk > 0]
export = []
for _, u in queue.iterrows():
    export.append({"NOMER": u.NOMER, "q": int(u.q), "TE": u.TE, "region": REGION.get(u.TE, u.TE),
                   "urban": u.K == "1", "persons": int(u.persons), "income": float(u.income),
                   "spending": float(u.spending), "risk": round(float(u.risk), 1),
                   "layers": u.layers if isinstance(u.layers, str) else "",
                   "lines": [list(x) for x in card(u)]})
with gzip.open(RES / "cards.json.gz", "wt", encoding="utf-8") as f:
    json.dump(export, f, ensure_ascii=False)
print("cards exported:", len(export))
