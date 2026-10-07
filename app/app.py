import gzip
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import CODE_RU  # noqa: E402

RES = ROOT / "results"
FEEDBACK = RES / "feedback.csv"
VERDICTS = {
    "error": "Ошибка подтверждена — анкета исправлена",
    "ok": "Всё верно — ложная тревога",
    "no_contact": "Не удалось связаться с семьёй",
}
TAG_COLOR = {"правило": "#2a78d6", "выброс": "#eb6834", "ИИ-модель": "#1baf7a", "правило + ИИ-модель": "#4a3aa7"}

st.set_page_config(page_title="Контроль качества Д-004", page_icon=":material/fact_check:", layout="wide")


@st.cache_data
def load():
    with gzip.open(RES / "cards.json.gz", "rt", encoding="utf-8") as f:
        cards = pd.DataFrame(json.load(f))
    sig = pd.read_csv(RES / "signals.csv.gz", dtype={"NOMER": str, "detail": str})
    sc = pd.read_csv(RES / "risk_scores.csv.gz", dtype={"NOMER": str, "TE": str, "K": str})
    auto = pd.read_csv(RES / "auto_corrections.csv", dtype={"NOMER": str}, encoding="utf-8-sig")
    ev = pd.read_csv(RES / "t3_evaluation.csv", index_col=0)
    pr = pd.read_csv(RES / "t4_precision.csv", index_col=0)
    codes = sig[sig.layer != "auto"].groupby(["NOMER", "q"])["code"].agg(lambda s: ",".join(sorted(set(s))))
    cards = cards.join(codes, on=["NOMER", "q"])
    cards["rank"] = range(1, len(cards) + 1)
    return cards, sig, sc, auto, ev, pr


def load_feedback():
    if FEEDBACK.exists():
        return pd.read_csv(FEEDBACK, dtype={"NOMER": str})
    return pd.DataFrame(columns=["time", "NOMER", "q", "verdict", "comment", "risk", "codes"])


cards, sig, sc, auto, ev, pr = load()
fb = load_feedback()
last = fb.drop_duplicates(["NOMER", "q"], keep="last").set_index(["NOMER", "q"])["verdict"] if len(fb) else pd.Series(dtype=str)
n_total = len(sc)

# ---------------- sidebar
st.sidebar.title(":material/fact_check: Контроль качества Д-004")
st.sidebar.caption("Журнал учёта ежеквартальных расходов и доходов домашних хозяйств, 2024 (синтетические данные)")
budget = st.sidebar.slider("Сколько анкет можем перепроверить", 1, 20, 5, format="%d%%",
                           help="Доля всех анкет квартала. Очередь берётся сверху по риску.")
regions = st.sidebar.multiselect("Регион", sorted(cards.region.unique()), placeholder="Все регионы")
place = st.sidebar.radio("Тип населённого пункта", ["все", "город", "село"], horizontal=True)
quarters = st.sidebar.multiselect("Квартал", [1, 2, 3, 4], placeholder="Все кварталы")
code_f = st.sidebar.multiselect("Тип сигнала", sorted({c for s in cards.code.dropna() for c in s.split(",")}),
                                format_func=lambda c: CODE_RU.get(c, c), placeholder="Все сигналы")
status_f = st.sidebar.radio("Статус", ["все", "не проверены", "проверены"], horizontal=True)
search = st.sidebar.text_input("Номер домохозяйства", placeholder="например, 2024001784")

q = cards[cards["rank"] <= budget / 100 * n_total].copy()
if regions:
    q = q[q.region.isin(regions)]
if place != "все":
    q = q[q.urban == (place == "город")]
if quarters:
    q = q[q.q.isin(quarters)]
if code_f:
    q = q[q.code.fillna("").apply(lambda s: any(c in s.split(",") for c in code_f))]
q["status"] = [VERDICTS.get(last.get((n, k)), "—") if len(last) else "—" for n, k in zip(q.NOMER, q.q)]
if status_f == "не проверены":
    q = q[q.status == "—"]
elif status_f == "проверены":
    q = q[q.status != "—"]
if search:
    q = cards[cards.NOMER.str.contains(search.strip())].assign(status="—")

tab_q, tab_auto, tab_sum, tab_quality = st.tabs([":material/format_list_numbered: Очередь проверки", ":material/rule: Автоисправления", ":material/bar_chart: Сводка", ":material/verified: Качество системы"])

# ---------------- queue
with tab_q:
    in_budget = int(budget / 100 * n_total)
    checked = fb.drop_duplicates(["NOMER", "q"], keep="last") if len(fb) else fb
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Анкет в квартальных данных", f"{n_total:,}".replace(",", " "))
    c2.metric("В очереди при выбранном бюджете", f"{in_budget:,}".replace(",", " "), f"{budget}% анкет", delta_color="off")
    c3.metric("Проверено", len(checked))
    c4.metric("Подтверждено ошибок", int((checked.verdict == "error").sum()) if len(checked) else 0)

    left, right = st.columns([6, 5], gap="large")
    with left:
        st.subheader(f"Очередь: {len(q)} анкет")
        view = q[["rank", "NOMER", "q", "region", "risk", "status"]].rename(columns={
            "rank": "№", "NOMER": "Домохозяйство", "q": "Кв.", "region": "Регион", "risk": "Риск", "status": "Статус"})
        view["Риск"] = view["Риск"].clip(upper=99).round().astype(int)
        sel = st.dataframe(view, hide_index=True, width="stretch", height=560,
                           on_select="rerun", selection_mode="single-row",
                           column_config={
                               "№": st.column_config.NumberColumn("№", width=45),
                               "Домохозяйство": st.column_config.TextColumn("Домохозяйство", width=105),
                               "Кв.": st.column_config.TextColumn("Кв.", width=40),
                               "Регион": st.column_config.TextColumn("Регион", width=130),
                               "Риск": st.column_config.ProgressColumn("Риск", min_value=0, max_value=100, format="%d", width=80),
                               "Статус": st.column_config.TextColumn("Статус", width=100)})
        rows = sel.selection.rows if sel and sel.selection else []
    with right:
        if not len(q):
            st.info("По выбранным фильтрам анкет нет.")
        else:
            u = q.iloc[rows[0]] if rows else q.iloc[0]
            st.subheader(f"Домохозяйство {u.NOMER} · {u.q} кв. 2024")
            money = lambda x: f"{x:,.0f}".replace(",", " ") + " тг"
            st.caption(f"{u.region}, {'город' if u.urban else 'село'}, {u.persons} чел. · доход {money(u.income)} · "
                       f"расходы {money(u.spending)}")
            st.markdown(f"**Риск: {min(99, round(u.risk))}/100**")
            st.progress(min(99, round(u.risk)) / 100)
            st.markdown("#### Что не так")
            for tag, text, _ in u.lines:
                color = TAG_COLOR.get(tag, "#52514e")
                st.markdown(f"<span style='background:{color};color:white;padding:2px 8px;border-radius:4px;"
                            f"font-size:.8em'>{tag}</span> {text}", unsafe_allow_html=True)
            st.markdown("#### Что уточнить у семьи")
            for c in dict.fromkeys(c for *_, c in u.lines):
                st.markdown(f"- {c}")
            st.markdown("#### Решение интервьюера")
            with st.form(f"fb_{u.NOMER}_{u.q}", clear_on_submit=True):
                verdict = st.radio("Результат проверки", list(VERDICTS), format_func=VERDICTS.get, horizontal=False)
                comment = st.text_input("Комментарий (что исправлено)")
                if st.form_submit_button("Сохранить", type="primary"):
                    row = pd.DataFrame([{"time": datetime.now().isoformat(timespec="seconds"), "NOMER": u.NOMER, "q": u.q,
                                         "verdict": verdict, "comment": comment, "risk": u.risk, "codes": u.code}])
                    row.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)
                    st.success("Сохранено. Эти ответы используются для дообучения модели.")
                    st.rerun()

# ---------------- auto-corrections
with tab_auto:
    st.subheader("Однозначные нарушения, исправленные автоматически")
    st.caption("Только случаи, где правильный ответ следует из Инструкции к Д-004, поэтому интервьюеру звонить не нужно. "
               "П. 8: после ответа «Нет» на вопрос-фильтр следующие вопросы пропускаются, лишние ответы удаляются. "
               "П. 7: продажа сельхозпродукции из раздела 4 учитывается в разделе 3 как доход от самозанятости. "
               "Сумма «в том числе» не может быть больше целого.")
    a1, a2 = st.columns(2)
    a1.metric("Анкет исправлено", f"{auto[['NOMER', 'q']].drop_duplicates().shape[0]:,}".replace(",", " "))
    a2.metric("Доля от всех анкет", f"{auto[['NOMER', 'q']].drop_duplicates().shape[0] / n_total:.1%}")
    auto["Исправление"] = auto["code"].map(lambda c: CODE_RU.get(c, c))
    kinds = auto.drop_duplicates(["NOMER", "q", "code"])["Исправление"].value_counts().rename("Анкет")
    for col, (name, n) in zip(st.columns(len(kinds)), kinds.items()):
        col.metric(name, f"{n:,}".replace(",", " "), f"{n / n_total:.1%} анкет", delta_color="off")
    kind = st.selectbox("Тип исправления", ["Все"] + list(kinds.index))
    shown = auto if kind == "Все" else auto[auto["Исправление"] == kind]
    st.dataframe(shown.rename(columns={"NOMER": "Домохозяйство", "q": "Кв.", "text": "Что исправлено"})
                 [["Домохозяйство", "Кв.", "Исправление", "Что исправлено"]], hide_index=True, width="stretch", height=360)

# ---------------- summary
with tab_sum:
    st.subheader("Какие проблемы находит система")
    s = sig[sig.layer != "auto"].drop_duplicates(["NOMER", "q", "code"])
    by_code = s.code.map(lambda c: CODE_RU.get(c, c)).value_counts().rename("Анкет")
    st.bar_chart(by_code, color="#2a78d6", horizontal=True, height=520)
    st.subheader("Доля анкет в очереди по регионам")
    reg = cards.groupby("region").size() / sc.assign(region=sc.TE).groupby("TE").size().rename(
        lambda t: cards.loc[cards.TE == t, "region"].iloc[0] if (cards.TE == t).any() else t)
    st.bar_chart((reg.dropna() * 100).sort_values().rename("% анкет"), color="#2a78d6", horizontal=True, height=520)

# ---------------- quality
with tab_quality:
    st.subheader("Эксперимент: в чистые анкеты внесены ошибки известных типов")
    st.caption("Доля найденных ошибок при проверке 2%, 5% и 10% анкет с наибольшим риском (или исправленных автоматически). "
               "Среднее по 3 повторам на синтетических данных 2024 года.")
    a = ev.loc["ALL"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Найдено при проверке 2%", f"{a['recall@2%']:.0%}")
    k2.metric("при 5%", f"{a['recall@5%']:.0%}")
    k3.metric("при 10%", f"{a['recall@10%']:.0%}")
    k4.metric("Точность топ-2%", f"{pr.loc[0.02, 'precision']:.0%}")
    t = ev.drop("ALL")[["description", "recall@2%", "recall@5%", "recall@10%"]].rename(columns={
        "description": "Тип ошибки", "recall@2%": "2%", "recall@5%": "5%", "recall@10%": "10%"})
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={c: st.column_config.ProgressColumn(c, min_value=0, max_value=1, format="percent")
                                for c in ["2%", "5%", "10%"]})

    st.subheader("Обучение на отметках интервьюеров")
    st.caption("Каждая отметка в карточке анкеты — это пример для модели. Когда накопится 400 отметок, очередь следующего "
               "квартала можно ранжировать с учётом того, какие сигналы чаще подтверждаются (src/08_feedback.py). "
               "В эксперименте это дало +5% найденных ошибок в 3 квартале и +14% в 4 квартале при проверке 2% анкет.")
    done = checked[checked.verdict != "no_contact"] if len(checked) else checked
    st.progress(min(len(done) / 400, 1.0), text=f"Отметок для обучения: {len(done)} из 400")
    if len(done):
        e = done.assign(code=done.codes.fillna("").str.split(",")).explode("code")
        e = e[e.code != ""]
        rate = (e.groupby("code").agg(n=("verdict", "size"), conf=("verdict", lambda v: (v == "error").mean()))
                  .sort_values("conf").reset_index())
        rate["code"] = rate.code.map(lambda c: CODE_RU.get(c, c))
        st.dataframe(rate.rename(columns={"code": "Сигнал", "n": "Проверено анкет", "conf": "Подтверждено ошибок"}),
                     hide_index=True, width="stretch",
                     column_config={"Подтверждено ошибок": st.column_config.ProgressColumn(
                         "Подтверждено ошибок", min_value=0, max_value=1, format="percent")})
