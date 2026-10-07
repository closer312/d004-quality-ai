import base64
import html
import pandas as pd
from config import OUT, RES, FIG, ROOT, REGION
from qc import tg, card_lines, risk_label

TEAM = "DataX"
AUTHORS = "Тариелова Алия Тариеловна, Ырысбеков Жаныбек Ырысбекович, Рахимбекова Салия Элнуровна, Джанболотов Кубат"

raw = pd.read_pickle(OUT / "raw.pkl")
sc = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str, "TE": str, "K": str})
sig = pd.read_csv(RES / "signals.csv.gz", dtype={"NOMER": str, "detail": str})
t1 = pd.read_csv(RES / "t1_signals_by_rule.csv")
ev = pd.read_csv(RES / "t3_evaluation.csv", index_col=0)
pr = pd.read_csv(RES / "t4_precision.csv", index_col=0)

P = lambda x: f"{x*100:.0f}%"
N = lambda x: f"{x:,.0f}".replace(",", " ")
img = lambda n: "data:image/png;base64," + base64.b64encode((FIG / n).read_bytes()).decode()
n_units = len(sc)
flagged = (sc.risk > 0).mean()
auto_share = (sc.auto_fixes > 0).mean()
import re
n_rules = len(set(re.findall(r'"(R\d\d)_', (ROOT / "src" / "qc.py").read_text(encoding="utf-8"))))
ml_only = int(pd.read_csv(RES / "t2_layer_overlap.csv").set_index("layers").loc["ml", "units"])

# example card: highest-risk unit flagged by all three layers
EXAMPLE = ("2024001784", 1)
ex = sc[(sc.NOMER == EXAMPLE[0]) & (sc.q == EXAMPLE[1])]
three = ex.iloc[0] if len(ex) and ex.iloc[0].risk > 0 else sc[sc.layers.fillna("").str.count(",") == 2].iloc[0]
s = sig[(sig.NOMER == three.NOMER) & (sig.q == three.q)]
names = raw["names"].set_index(["table", "kodnu"])["name"]
it_u = raw["items"][(raw["items"].NOMER == three.NOMER) & (raw["items"].q == three.q)].sort_values("STOIMK")
bg = (names.get((it_u.iloc[-1].table, it_u.iloc[-1].KODNU), ""), it_u.iloc[-1].STOIMK) if len(it_u) else None
lines = card_lines(s, three, bg)
lis = "".join(f"<li><span class='tag'>{html.escape(l)}</span> {html.escape(x)}</li>" for l, x, _ in lines)
checks = "".join(f"<li>{html.escape(c)}</li>" for c in dict.fromkeys(c for *_, c in lines))
card = (f"<div class='card'><div class='hd'><b>Домохозяйство {three.NOMER}</b> · {three.q} кв. 2024 · "
        f"{REGION.get(three.TE, three.TE)}, {'город' if three.K == '1' else 'село'}"
        f"<span class='risk'>{risk_label(three.risk)}</span></div><ul>{lis}</ul>"
        f"<div class='chk'><b>Что уточнить у респондента:</b><ul>{checks}</ul></div></div>")

a = ev.loc["ALL"]
slides = []
S = lambda h: slides.append(f'<section class="slide">{h}</section>')

S(f"""<p class="kicker">STAT.DATATHON 2026 · Результат 2 · Трек «AI for Official Statistics» · 07.10.2026</p>
<h1>ИИ-контроль качества анкет бюджетов домохозяйств (Д-004)</h1>
<p class="muted">Команда «{TEAM}» · {AUTHORS}</p>
<div class="cols">
<div><h3>Проблема</h3><ul>
<li>Анкета Д-004: 12 000 семей × 4 квартала, 12 таблиц, <b>{N(len(raw['items']))}</b> записей о покупках только за 2024 год.</li>
<li>Ошибки ввода — лишний ноль, сумма в тысячах вместо тенге, не та графа, двойной ввод — искажают оценки доходов, расходов и бедности.</li>
<li>Сейчас проверка ручная. Инструкция к Д-004, п. 9: <i>«интервьюер повторно просматривает форму, в случае обнаружения каких-либо несоответствий, вновь обращается в домашнее хозяйство… и выясняет недостающую информацию»</i>. Наша система автоматизирует именно этот шаг.</li>
<li>В плане ИИ-ассистентов Бюро: «Контроль качества первичных данных» (2027) и «Обратная связь по качеству отчёта» (2028).</li>
</ul></div>
<div><h3>Решение</h3>
<p>Система <b>ранжирует анкеты по риску ошибки</b> и <b>объясняет интервьюеру простым языком</b>, что перепроверить. Решение остаётся за специалистом.</p>
<h3>Гипотезы</h3><ol>
<li><b>H1.</b> Логические правила надёжно находят известные типы ошибок.</li>
<li><b>H2.</b> Статистические выбросы и ML-модели находят то, что правила не описывают: нетипичные суммы и редкие сочетания ответов.</li>
<li><b>H3.</b> Проверка 5% анкет с наибольшим риском находит большую часть ошибок — во много раз эффективнее случайной проверки.</li>
</ol></div></div>""")

S(f"""<h2>Как это работает</h2>
<div class="cols c55">
<div class="flow">
<div class="box">Анкета Д-004 (12 таблиц) + состав семьи Д-008</div><div class="arr">↓</div>
<div class="box">Профиль домохозяйства: доходы, расходы, долги, состав</div><div class="arr">↓</div>
<div class="row3">
<div class="box l1"><b>Правила</b><br>{n_rules} логических проверок: переходы и коды бланка Д-004, сверка с Д-008, пенсия у 25-летнего, двойной ввод, долги без источника денег…</div>
<div class="box l2"><b>Выбросы и модель покупки</b><br>цена против 550 позиций справочника; проверка «лишнего нуля»; бустинг оценивает обычную цену покупки для этой семьи</div>
<div class="box l3"><b>ИИ-модели</b><br>Isolation Forest — редкие сочетания; бустинг — «ожидаемые расходы» похожих семей</div>
</div><div class="arr">↓</div>
<div class="row2"><div class="box">Однозначные нарушения бланка → <b>автоисправление</b> + журнал изменений</div>
<div class="box">Остальное → <b>риск 0–100</b> и объяснение для интервьюера</div></div><div class="arr">↓</div>
<div class="box hl">Специалист подтверждает или исправляет → обратная связь для настройки</div>
<p class="muted">Из {N(n_units)} анкет 2024 года {P(auto_share)} исправлены автоматически, у {P(flagged)} есть сигнал для интервьюера — проверяются по убыванию риска.</p>
</div>
<div><h3>Пример объяснения (запись из синтетических данных 2024 года)</h3>{card}</div>
</div>""")

S(f"""<h2>Первые результаты: эксперимент с внесёнными ошибками</h2>
<div class="cols c55"><div><img src="{img('f1_evaluation.png')}">
<p class="muted">В чистые синтетические анкеты 2024 года внесли {len(ev) - 2} типов ошибок по 150 штук ({N(a['n'])} анкет), прогнали систему и посмотрели, где они оказались в рейтинге риска. Среднее по 3 повторам.</p></div>
<div>
<div class="kpis">
<div class="kpi"><b>{P(a['recall@5%'])}</b><span>ошибок найдено при проверке 5% анкет (случайная проверка — 5%)</span></div>
<div class="kpi"><b>{P(a['recall@10%'])}</b><span>найдено при проверке 10% анкет</span></div>
<div class="kpi"><b>{P(pr.loc[0.02,'precision'])}</b><span>топ-2% по риску — действительно ошибки</span></div>
</div>
<ul>
<li><b>H1 подтверждается:</b> правила полностью находят {ev.drop(['ALL','E1_big']).query('caught_by_rules > 0.9 or caught_by_auto > 0.9').shape[0]} из {len(ev) - 2} типов ошибок; нарушения переходов бланка исправляются автоматически и не занимают очередь интервьюера.</li>
<li><b>H2 — частично:</b> выбросы находят все суммы в тысячах тенге, ИИ-модель — пропущенные разделы расходов и ещё {N(ml_only)} анкет, которые не описывает ни одно правило. Слабое место — «лишний ноль» в одной покупке: {P(ev.loc['E1_extra_zero','recall@5%'])} при 5% и {P(ev.loc['E1_extra_zero','recall@10%'])} при 10% (крупные — {P(ev.loc['E1_big','recall@5%'])} и {P(ev.loc['E1_big','recall@10%'])}).</li>
<li><b>H3 подтверждается:</b> топ-5% по риску в {a['recall@5%']/0.05:.0f} раз эффективнее случайной выборки.</li>
</ul>
</div></div>""")

css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--accent:#2a78d6;--warm:#eb6834;--line:#e6e5e0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.slide{width:100%;max-width:1400px;min-height:100vh;margin:0 auto;padding:3.5vh 4vw;display:flex;flex-direction:column;justify-content:center;border-bottom:1px solid var(--line)}
h1{font-size:2rem;line-height:1.15;margin:.2em 0}h2{font-size:1.6rem;margin:0 0 .5em}h3{font-size:1rem;color:var(--muted);margin:.6em 0 .3em}
.kicker{color:var(--accent);font-weight:600;margin:0;font-size:.9rem}.muted{color:var(--muted);font-size:.82rem}
img{max-width:100%;display:block}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:2.5vw;align-items:start}.cols.c55{grid-template-columns:1.1fr 1fr;align-items:center}
ul,ol{padding-left:1.15em;margin:.3em 0}li{margin:.28em 0}
.flow{display:flex;flex-direction:column;gap:.15em}.box{border:1px solid var(--line);background:#fff;border-radius:8px;padding:.45em .8em;font-size:.88rem}
.row3{display:grid;grid-template-columns:repeat(3,1fr);gap:.5em}.row2{display:grid;grid-template-columns:1fr 1fr;gap:.5em}.l1{border-top:4px solid #2a78d6}.l2{border-top:4px solid #eb6834}.l3{border-top:4px solid #1baf7a}
.hl{background:#eef3fb}.arr{text-align:center;color:var(--muted);line-height:1}
.card{border:1px solid var(--line);border-radius:8px;padding:.7em 1em;background:#fff;font-size:.9rem}.hd{display:flex;gap:.5em;flex-wrap:wrap;align-items:baseline}
.risk{margin-left:auto;background:#fdeee8;color:#9a3412;padding:.1em .6em;border-radius:99px;font-weight:600}
.tag{font-size:.72em;padding:.05em .45em;border-radius:4px;background:#eef3fb;color:#1d4f91}.chk{border-top:1px dashed var(--line);margin-top:.4em;padding-top:.3em}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:.6em;margin-bottom:.5em}.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:.5em .7em}
.kpi b{display:block;font-size:1.7rem;color:var(--accent)}.kpi span{font-size:.8rem;color:var(--muted)}
@media(max-width:900px){.cols,.cols.c55,.row3,.row2,.kpis{grid-template-columns:1fr}.slide{padding:4vh 16px}}
@media print{@page{size:1400px 800px;margin:0}.slide{height:800px;min-height:0;page-break-after:always;border:0;overflow:hidden}}
"""
out = ROOT / "presentation" / "index.html"
out.write_text(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Результат 2 — ИИ-контроль качества Д-004</title><style>{css}</style></head><body>{''.join(slides)}</body></html>""", encoding="utf-8")
print(out.name, len(slides), "slides;", three.NOMER, three.q)
