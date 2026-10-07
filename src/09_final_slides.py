"""Final presentation (defense, up to 15 minutes) -> presentation/final.html; print to PDF in a browser."""
import base64
import html
import re
import pandas as pd
from config import OUT, RES, FIG, ROOT, REGION, CODE_RU
from qc import card_lines, risk_label

TEAM = "DataX"
AUTHORS = ["Тариелова Алия Тариеловна", "Ырысбеков Жаныбек Ырысбекович", "Рахимбекова Салия Элнуровна", "Джанболотов Кубат"]
REPO = "github.com/closer312/d004-quality-ai"

raw = pd.read_pickle(OUT / "raw.pkl")
sc = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str, "TE": str, "K": str})
sig = pd.read_csv(RES / "signals.csv.gz", dtype={"NOMER": str, "detail": str})
t1 = pd.read_csv(RES / "t1_signals_by_rule.csv").set_index("code")
ev = pd.read_csv(RES / "t3_evaluation.csv", index_col=0)
pr = pd.read_csv(RES / "t4_precision.csv", index_col=0)
sv = pd.read_csv(RES / "t6_savings.csv").set_index("method")
fl = pd.read_csv(RES / "t7_feedback_loop.csv")
rc = pd.read_csv(RES / "t8_rule_confirmation.csv")

P = lambda x: f"{x * 100:.0f}%"
dec1 = lambda x: f"{x:.1f}".replace(".", ",")
dec2 = lambda x: f"{x:.2f}".replace(".", ",")
P1 = lambda x: f"{x * 100:.1f}%".replace(".", ",")
N = lambda x: f"{x:,.0f}".replace(",", " ")
img = lambda n: "data:image/png;base64," + base64.b64encode((FIG / n).read_bytes()).decode()
n_units = len(sc)
queue = (sc.risk > 0).mean()
auto_share = (sc.auto_fixes > 0).mean()
n_rules = len(set(re.findall(r'"(R\d\d)_', (ROOT / "src" / "qc.py").read_text(encoding="utf-8"))))
ml_only = int(pd.read_csv(RES / "t2_layer_overlap.csv").set_index("layers").loc["ml", "units"])
u = lambda c: t1.loc[c, "units"] if c in t1.index else 0
a = ev.loc["ALL"]
n_full = int((ev.drop(["ALL", "E1_big"])["recall@5%"] >= .99).sum())
f2 = fl[fl.budget == .02].set_index("q")
loc = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in list((ROOT / "src").glob("*.py")) + [ROOT / "app" / "app.py"])

# example card (synthetic record)
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

slides = []
S = lambda h: slides.append(f'<section class="slide">{h}<div class="pg">{len(slides) + 1}</div></section>')
r, sy, ru = sv.loc["random"], sv.loc["system"], sv.loc["rules"]

# 1 title
S(f"""<p class="kicker">STAT.DATATHON 2026 · Финальная защита · Трек «AI for Official Statistics»</p>
<h1 class="big">ИИ-контроль качества анкет бюджетов домохозяйств (Д-004)</h1>
<p class="lead">Программа сама находит подозрительные анкеты, объясняет интервьюеру, что в них не так,
и становится точнее с каждой проверкой.</p>
<div class="team"><b>Команда «{TEAM}»</b><br>{'<br>'.join(AUTHORS)}</div>
<p class="muted">Все примеры — из синтетических (учебных) данных Бюро национальной статистики за 2024 год. Код: {REPO}</p>""")

# 2 problem
S(f"""<h2>Проблема: ошибки в анкетах ищут вручную</h2>
<div class="cols"><div><ul class="lg">
<li>Анкету Д-004 каждый квартал заполняют <b>12 000 семей</b>. Только за 2024 год в ней <b>{N(len(raw['items']))}</b> записей о покупках.</li>
<li>При вводе бывают ошибки: лишний ноль, сумма в тысячах вместо тенге, доход не в той графе, раздел введён дважды.</li>
<li>Из-за них неверно считаются средние доходы и расходы семей, а значит, и уровень бедности.</li>
<li>По Инструкции к Д-004 (п. 9) интервьюер перепроверяет анкету и при сомнениях снова звонит семье. Но проверить всё невозможно.</li>
</ul></div>
<div><div class="qbox"><b>Наши три вопроса</b><ol class="qs">
<li>Можно ли найти большинство ошибок, проверив лишь малую часть анкет?</li>
<li>Сколько работы это экономит?</li>
<li>Может ли программа учиться на ответах интервьюеров?</li></ol>
<span class="muted">Ответы — на последнем слайде.</span></div>
<p class="muted">Это совпадает с планом Бюро: ИИ-помощник «Контроль качества первичных данных» (2027).</p></div></div>""")

# 3 data
S(f"""<h2>Данные</h2>
<div class="kpis k4">
<div class="kpi"><b>{N(n_units)}</b><span>анкет за 2024 год (семья × квартал)</span></div>
<div class="kpi"><b>{N(len(raw['members']))}</b><span>записей о доходах</span></div>
<div class="kpi"><b>{N(len(raw['items']))}</b><span>записей о покупках</span></div>
<div class="kpi"><b>{len(REGION)}</b><span>регионов, город и село</span></div></div>
<div class="cols"><div><h3>Что используем</h3><ul class="lg">
<li>Анкету Д-004: расходы, займы, доходы, своё хозяйство.</li>
<li>Анкету Д-008: кто живёт в семье и сколько им лет.</li>
<li>Справочник товаров (550 позиций) и Инструкцию к Д-004.</li></ul></div>
<div><h3>Важно знать</h3><ul class="lg">
<li>Данные учебные: связи между разделами анкеты в них слабее, чем в жизни.</li>
<li>Где настоящие ошибки, никто не знает. Поэтому мы сами добавили ошибки и проверили, найдёт ли их программа.</li></ul></div></div>""")

# 4 architecture
S(f"""<h2>Как работает</h2>
<div class="flow wide">
<div class="box">Берём анкету семьи: доходы, расходы, долги, состав семьи</div><div class="arr">↓</div>
<div class="row3">
<div class="box l1"><b>Правила — {n_rules}</b><br>проверки по бланку и Инструкции: например, пенсия у 25-летнего, доход у ребёнка, расходы без денег на них</div>
<div class="box l2"><b>Проверка каждой покупки</b><br>сравниваем цену с обычной для этого товара; проверяем, не лишний ли ноль</div>
<div class="box l3"><b>Модели машинного обучения</b><br>ищут необычные анкеты и расходы, которые не похожи на расходы похожих семей</div>
</div><div class="arr">↓</div>
<div class="row2"><div class="box ok"><b>Явные ошибки</b> исправляются сами, по Инструкции: {P1(auto_share)} анкет, звонить не нужно</div>
<div class="box"><b>Остальные получают оценку риска</b> от 0 до 100 и объяснение: {P1(queue)} анкет</div></div><div class="arr">↓</div>
<div class="row2"><div class="box hl"><b>Интервьюер</b> проверяет анкеты по порядку, от самых подозрительных</div>
<div class="box hl"><b>Его ответы</b> помогают программе лучше выбирать анкеты в следующем квартале</div></div>
</div>""")

# 5 example card
S(f"""<h2>Что видит интервьюер</h2>
<div class="cols c55"><div>{card}</div>
<div><ul class="lg">
<li>Вместо кода ошибки — <b>понятная фраза</b> с цифрами из анкеты.</li>
<li>Рядом — <b>что спросить у семьи</b>.</li>
<li>Метка показывает, кто нашёл: правило, проверка покупки или модель.</li>
<li>Решает всегда человек. Сама программа исправляет только явные ошибки по Инструкции.</li>
</ul></div></div>""")

# 6 evaluation
S(f"""<h2>Насколько хорошо находит ошибки</h2>
<img class="chart" src="{img('f1_evaluation.png')}">
<div class="cols"><div><div class="kpis">
<div class="kpi"><b>{P(a['recall@2%'])}</b><span>ошибок найдено, если проверить 2% анкет</span></div>
<div class="kpi"><b>{P(a['recall@5%'])}</b><span>если проверить 5% анкет (наугад — нашли бы 5%)</span></div>
<div class="kpi"><b>{P(pr.loc[0.02, 'precision'])}</b><span>самых подозрительных анкет действительно с ошибкой</span></div></div>
<p class="muted">Мы добавили в анкеты {len(ev) - 2} видов ошибок, по 150 каждого ({N(a['n'])} анкет), и повторили опыт 3 раза.</p></div>
<div><ul class="lg"><li>{n_full} видов ошибок из {len(ev) - 2} находятся полностью.</li>
<li>Труднее всего — лишний ноль в одной мелкой покупке. Если он сильно меняет расходы семьи, находится {P(ev.loc['E1_big', 'recall@10%'])} таких ошибок при проверке 10% анкет.</li></ul></div></div>""")

# 7 savings
S(f"""<h2>Сколько времени экономит</h2>
<img class="chart" src="{img('f2_impact.png')}">
<div class="cols"><div><table><tr><th>Как выбирать анкеты</th><th>Сколько проверить, чтобы найти 90% ошибок</th><th>Часов работы*</th></tr>
<tr><td>наугад</td><td>{N(r.checks_for_90pct)}</td><td>{N(r.hours_for_90pct)}</td></tr>
<tr><td>только по правилам</td><td>{N(ru.checks_for_90pct)}</td><td>{N(ru.hours_for_90pct)}</td></tr>
<tr class="hlr"><td>наша программа</td><td>{N(sy.checks_for_90pct)}</td><td>{N(sy.hours_for_90pct)}</td></tr></table></div>
<div><ul class="lg"><li>Звонков семьям в <b>{r.checks_for_90pct / sy.checks_for_90pct:.0f} раз</b> меньше, чем при проверке наугад.</li>
<li>Ошибка в общей сумме расходов: было {dec2(sy.spending_abs_err_no_check)}%, стало {dec2(sy.spending_abs_err_at_5pct)}%. Ошибки в доходах исчезают полностью.</li></ul>
<p class="muted">* считаем 20 минут на один звонок.</p></div></div>""")

# 8 feedback loop
S(f"""<h2>Программа учится на ответах интервьюеров</h2>
<img class="chart" src="{img('f3_feedback.png')}">
<div class="cols"><div><ul class="lg">
<li>Интервьюер отмечает: «ошибка подтвердилась» или «всё верно».</li>
<li>Когда наберётся 400 отметок, программа учится на них и в следующем квартале выбирает анкеты точнее.</li></ul></div>
<div><ul class="lg"><li>При проверке 2% анкет ошибок находится больше: <b>+{f2.loc[3, 'gain_learned_pct']:.0f}%</b> в 3 квартале и <b>+{f2.loc[4, 'gain_learned_pct']:.0f}%</b> в 4.</li>
<li>Справа видно, каким сигналам можно доверять больше.</li></ul>
<p class="muted">Это расчёт на учебных данных. На настоящих отметках цифры будут другими.</p></div></div>""")

# 9 instruction-based checks
S(f"""<h2>Проверки по Инструкции к Д-004</h2>
<table class="wide"><tr><th>Что проверяем</th><th>Пункт Инструкции</th><th>Что делает программа</th><th>Доля анкет</th></tr>
<tr><td>Ответы на вопросы, которые надо было пропустить</td><td>п. 8</td><td>исправляет сама</td><td>{P1(u('R16_skip_logic') / n_units)}</td></tr>
<tr><td>Семья продала урожай или продукцию хозяйства, но этих денег нет в доходах</td><td>п. 7</td><td>исправляет сама</td><td>{P1(u('R22_agri_sales') / n_units)}</td></tr>
<tr><td>Доход от продажи урожая больше, чем весь доход от своего дела</td><td>логика анкеты</td><td>исправляет сама</td><td>{P1(u('R24_agri_gt_self') / n_units)}</td></tr>
<tr><td>Доход от продажи урожая есть, а самих продаж в анкете нет</td><td>п. 7</td><td>спрашивает интервьюера</td><td>{P1(u('R23_agri_no_sales') / n_units)}</td></tr>
<tr><td>Доход записан у ребёнка младше 15 лет</td><td>п. 7</td><td>спрашивает интервьюера</td><td>{P1(u('R20_income_under15') / n_units)}</td></tr>
<tr><td>Сумма не в целых тенге или код не из справочника</td><td>п. 5</td><td>спрашивает интервьюера</td><td>{P1((u('R21_not_whole') + u('R17_invalid_code')) / n_units)}</td></tr>
</table>
<p class="muted">Каждое исправление записывается в журнал. Срабатываний по продажам много из-за особенностей учебных данных.</p>""")

# 10 web app
S(f"""<h2>Рабочее место интервьюера</h2>
<div class="cols c64"><div><img class="shot" src="{img('app_queue.png')}"></div>
<div><ul class="lg">
<li><b>Список анкет</b> — от самых подозрительных. Можно выбрать, сколько анкет проверять, регион и квартал.</li>
<li><b>Карточка анкеты:</b> что не так и что спросить у семьи.</li>
<li><b>Кнопка «Сохранить»</b> — ответ интервьюера идёт в обучение.</li>
<li>Отдельно: журнал автоисправлений, сводка по регионам, качество программы.</li></ul></div></div>""")

# 11 summary
S(f"""<h2>Ответы на наши вопросы</h2>
<ol class="ans">
<li><b>Можно ли найти большинство ошибок, проверив малую часть анкет?</b> Да: проверка 5% анкет с самым высоким риском находит {P(a['recall@5%'])} ошибок.</li>
<li><b>Сколько работы это экономит?</b> Звонков семьям в {r.checks_for_90pct / sy.checks_for_90pct:.0f} раз меньше, чем при проверке наугад.</li>
<li><b>Может ли программа учиться?</b> Да: после 400 отметок интервьюеров она находит до +{f2.loc[4, 'gain_learned_pct']:.0f}% ошибок.</li></ol>
<div class="kpis k4">
<div class="kpi"><b>{P(a['recall@5%'])}</b><span>ошибок находится при проверке всего 5% анкет</span></div>
<div class="kpi"><b>в {r.checks_for_90pct / sy.checks_for_90pct:.0f} раз</b><span>меньше звонков семьям, чем при проверке наугад</span></div>
<div class="kpi"><b>{P1(auto_share)}</b><span>анкет исправлены сами по Инструкции</span></div>
<div class="kpi"><b>+{f2.loc[4, 'gain_learned_pct']:.0f}%</b><span>ошибок находится больше после обучения на ответах</span></div></div>
<p class="lead">Программа не заменяет интервьюера. Она подсказывает, куда смотреть, объясняет почему и учится на его решениях.</p>""")


css = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--accent:#2a78d6;--warm:#eb6834;--line:#e6e5e0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.slide{position:relative;width:100%;max-width:1400px;min-height:100vh;margin:0 auto;padding:4vh 4.5vw;display:flex;flex-direction:column;justify-content:center;border-bottom:1px solid var(--line)}
.pg{position:absolute;right:2.2vw;bottom:2vh;color:var(--muted);font-size:.8rem}
h1{font-size:2.1rem;line-height:1.15;margin:.2em 0}h1.big{font-size:2.6rem;max-width:22em}h2{font-size:1.7rem;margin:0 0 .6em}h3{font-size:1rem;color:var(--muted);margin:.6em 0 .3em}
.kicker{color:var(--accent);font-weight:600;margin:0;font-size:.95rem}.muted{color:var(--muted);font-size:.85rem}
.lead{font-size:1.25rem;max-width:44em;margin:.6em 0}.big2{font-size:1.4rem;font-weight:600;color:var(--accent);margin-top:1em}
.team{border-left:4px solid var(--accent);padding:.3em .9em;margin:1em 0;font-size:1rem}
img{max-width:100%;display:block}.shot{border:1px solid var(--line);border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,.06)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:3vw;align-items:start}.cols.c55{grid-template-columns:1.15fr 1fr;align-items:center}.cols.c64{grid-template-columns:1.5fr 1fr;align-items:center}.cols.c46{grid-template-columns:1.35fr 1fr;align-items:center}
.chart{max-height:390px;max-width:100%;margin:0 auto .6em;width:auto}
ul,ol{padding-left:1.15em;margin:.3em 0}li{margin:.32em 0}.lg li{margin:.5em 0}
.qbox{background:#eef3fb;border-radius:10px;padding:1em 1.2em;font-size:1.1rem;margin-bottom:1em}
.flow{display:flex;flex-direction:column;gap:.2em}.box{border:1px solid var(--line);background:#fff;border-radius:8px;padding:.55em .9em;font-size:.95rem}
.row3{display:grid;grid-template-columns:repeat(3,1fr);gap:.6em}.row2{display:grid;grid-template-columns:1fr 1fr;gap:.6em}
.l1{border-top:4px solid #2a78d6}.l2{border-top:4px solid #eb6834}.l3{border-top:4px solid #1baf7a}.ok{border-left:4px solid #1baf7a}
.hl{background:#eef3fb}.arr{text-align:center;color:var(--muted);line-height:1}
.card{border:1px solid var(--line);border-radius:8px;padding:.8em 1.1em;background:#fff;font-size:.92rem}.hd{display:flex;gap:.5em;flex-wrap:wrap;align-items:baseline}
.risk{margin-left:auto;background:#fdeee8;color:#9a3412;padding:.1em .6em;border-radius:99px;font-weight:600}
.tag{font-size:.72em;padding:.05em .45em;border-radius:4px;background:#eef3fb;color:#1d4f91}.chk{border-top:1px dashed var(--line);margin-top:.4em;padding-top:.3em}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:.7em;margin-bottom:.8em}.kpis.k4{grid-template-columns:repeat(4,1fr)}
.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:.6em .8em}.kpi b{display:block;font-size:1.9rem;color:var(--accent)}.kpi span{font-size:.85rem;color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.95rem;margin:.3em 0 .8em;background:#fff}th,td{border-bottom:1px solid var(--line);padding:.45em .6em;text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.85rem}.hlr td{background:#eef3fb;font-weight:600}table.wide td:last-child{text-align:right;white-space:nowrap}
code{background:#f1f1ec;padding:.05em .3em;border-radius:4px}
.qs{margin:.5em 0 .3em}.qs li{margin:.45em 0}.ans{margin:0 0 1em}.ans li{font-size:1.1rem;margin:.55em 0}
@media(max-width:900px){.cols,.cols.c55,.cols.c64,.row3,.row2,.kpis,.kpis.k4{grid-template-columns:1fr}.slide{padding:4vh 16px}}
@media print{@page{size:1400px 800px;margin:0}.slide{height:800px;min-height:0;page-break-after:always;border:0;overflow:hidden}}
"""
out = ROOT / "presentation" / "final.html"
out.write_text(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ИИ-контроль качества Д-004 — финальная защита</title><style>{css}</style></head><body>{''.join(slides)}</body></html>""", encoding="utf-8")
print(out.name, len(slides), "slides")
