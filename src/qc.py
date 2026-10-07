import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from config import KODNU12, INCOME, INCOME_RU, TABLE_RU, SEED

KEY = ["NOMER", "q"]
# Section 4 of the D-004 form: filter question -> questions that must be skipped after "No" (kv_vopr9, 2024 codes).
# Source: Instruction for form D 004, item 8 (order No. 14 of 19.12.2019 as amended by order No. 11 of 25.06.2024)
SKIP_RULES = {
    "91": ("4.1 (есть ли доступ к земельному участку)", {"92", "93", "94"}, "4.2–4.4"),
    "92": ("4.2 (были ли расходы на участок)", {"93"}, "4.3"),
    "94": ("4.4 (выращивали ли что-либо на участке)", {"96"}, "4.5"),
    "97": ("4.6 (содержите ли скот, птицу, пчёл)", {"98", "99", "913", "914"}, "4.7–4.11"),
    "913": ("4.10 (получали ли продукцию от животных)", {"914"}, "4.11"),
}
SKIP_Q = list(SKIP_RULES)
OUT_P = (0.3, 0.8)
INC_COLS = [c for c in INCOME.values() if c != "of_which_agri_sales"]
SOCIAL = ["stipend", "targeted_social_aid", "housing_aid", "child_benefits", "state_social_benefit",
          "special_state_benefit", "other_state_transfers", "social_transfers_in_kind"]
SP_RU = {1: "непродовольственные товары", 2: "коммунальные услуги", 3: "связь", 4: "образование",
         5: "здравоохранение", 6: "услуги, отдых и прочее", 7: "транспорт"}
HI, LO = ("намного выше", "намного ниже"), ("намного больше", "намного меньше")
FEATURE_RU = {
    "log_income_pc": ("доход на человека", HI), "log_spending_pc": ("расходы на человека", HI),
    "wage_share": ("доля зарплаты в доходе", HI), "pension_share": ("доля пенсий в доходе", HI),
    "social_share": ("доля соцпособий в доходе", HI), "self_share": ("доля самозанятости в доходе", HI),
    "debt_in": ("полученные кредиты и займы", LO), "debt_out": ("выплаты по долгам", LO),
    "savings_used": ("использованные сбережения", LO), "persons": ("размер семьи", LO),
    "child_share": ("доля детей", HI), "elderly_share": ("доля пожилых", HI),
    "log_items": ("число записей о покупках", LO),
    **{f"sp{t}_share": (f"доля расходов на {SP_RU[t]}", HI) for t in range(1, 8)},
}


def tg(x):
    return f"{x:,.0f}".replace(",", " ") + " тг"


def approx(x):
    if x >= 1e6:
        return f"около {x/1e6:.1f} млн тг".replace(".", ",")
    if x >= 1e4:
        return f"около {x/1e3:,.0f} тыс. тг".replace(",", " ")
    return f"около {x:,.0f} тг".replace(",", " ")


def dec(x):
    return f"{x:.1f}".replace(".", ",")


def ratio_txt(x):
    return times(x) if x >= 10 else f"{dec(x)} раза"


def years(a):
    a = int(a)
    if a % 10 == 1 and a % 100 != 11:
        return f"{a} год"
    return f"{a} года" if a % 10 in (2, 3, 4) and a % 100 not in (12, 13, 14) else f"{a} лет"


def plural(n, forms):
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} {forms[0]}"
    return f"{n} {forms[1]}" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else f"{n} {forms[2]}"


def clean_name(x):
    return " ".join(str(x).replace("помщь", "помощь").split())


def times(r):
    n = int(round(r))
    return f"{n} раза" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else f"{n} раз"


def robust_z(x, med=None, mad=None):
    med = np.nanmedian(x) if med is None else med
    mad = np.nanmedian(np.abs(x - med)) if mad is None else mad
    return (x - med) / (1.4826 * mad if mad else np.nan)


# ---------------------------------------------------------------- profile
def build_profile(raw):
    hh, m, it, v, ro = raw["hh"], raw["members"], raw["items"], raw["v12"], raw["roster"]
    inc = m.groupby(KEY)[INC_COLS].sum()
    inc["income"] = inc.sum(axis=1)
    inc["social"] = inc[SOCIAL].sum(axis=1)
    sp = it.pivot_table(index=KEY, columns="table", values="STOIMK", aggfunc="sum").fillna(0)
    sp.columns = [f"sp{t}" for t in sp.columns]
    for t in range(1, 8):
        if f"sp{t}" not in sp:
            sp[f"sp{t}"] = 0.0
    sp["spending"] = sp.sum(axis=1)
    sp["n_items"] = it.groupby(KEY).size()
    a = v[v["NOMVNEW"] == "20"].assign(item=lambda d: d["KODNU"].str.strip().str.rstrip(",").map(KODNU12),
                                       val=lambda d: pd.to_numeric(d["STOIMK"], errors="coerce"))
    a = a.pivot_table(index=KEY, columns="item", values="val", aggfunc="sum")
    for c in KODNU12.values():
        if c not in a:
            a[c] = 0.0
    ans = v[v["NOMVNEW"].isin(["21", "22"])].groupby(KEY).size().rename("n_answers")
    rs = ro.groupby("NOMER").agg(persons=("NOMP", "size"), children=("age", lambda s: (s < 18).sum()),
                                 elderly=("age", lambda s: (s >= 63).sum()))
    p = (hh.set_index(KEY).join(inc).join(sp).join(a).join(ans).reset_index()
           .merge(rs, on="NOMER", how="left"))
    num = [c for c in p.columns if c not in ("NOMER", "TE", "K", "q")]
    p[num] = p[num].fillna(0)
    p["urban"] = (p["K"] == "1").astype(int)
    p["persons"] = p["persons"].clip(lower=1)
    return p


def features(p):
    f = pd.DataFrame(index=p.index)
    inc = p["income"].replace(0, np.nan)
    f["log_income_pc"] = np.log1p(p["income"] / p["persons"])
    f["log_spending_pc"] = np.log1p(p["spending"] / p["persons"])
    f["wage_share"] = (p["wage"] / inc).fillna(0)
    f["pension_share"] = (p["pension"] / inc).fillna(0)
    f["social_share"] = (p["social"] / inc).fillna(0)
    f["self_share"] = (p["self_employed"] / inc).fillna(0)
    spd = p["spending"].replace(0, np.nan)
    for t in range(1, 8):
        f[f"sp{t}_share"] = (p[f"sp{t}"] / spd).fillna(0)
    f["debt_in"] = np.log1p(p["credit_received"] + p["informal_borrowed"])
    f["debt_out"] = np.log1p(p["credit_repaid"] + p["informal_repaid"])
    f["savings_used"] = np.log1p(p["savings_used"])
    f["persons"] = p["persons"]
    f["child_share"] = p["children"] / p["persons"]
    f["elderly_share"] = p["elderly"] / p["persons"]
    f["log_items"] = np.log1p(p["n_items"])
    f["urban"] = p["urban"]
    return f


# ---------------------------------------------------------------- layer 1: rules
def layer_rules(raw, p):
    m, it, v = raw["members"], raw["items"], raw["v12"]
    out = []

    def add(df, code, prob, text, check, nomp=None, layer="rules"):
        if len(df):
            out.append(pd.DataFrame({"NOMER": df["NOMER"].values, "q": df["q"].values, "layer": layer,
                                     "code": code, "p": prob, "text": text(df),
                                     "check": check(df) if callable(check) else check,
                                     "nomp": df[nomp].values if nomp else np.nan}))

    fin = p["credit_received"] + p["informal_borrowed"] + p["savings_used"] + p["debt_returned_to_hh"]
    g = p[(p["income"] > 0) & (p["spending"] > 3 * p["income"]) & (fin == 0)]
    add(g, "R01_gap", 0.6, lambda d: [f"Расходы {tg(s)} в {ratio_txt(s/i)} больше дохода семьи ({tg(i)}), "
                                     f"а кредиты, долги и сбережения не указаны" for s, i in zip(d.spending, d.income)],
        "Уточните, на какие деньги сделаны крупные покупки (кредит, сбережения, помощь родственников), и сверьте их суммы с бумажной анкетой.")
    g = p[(p["income"] == 0) & (p["spending"] > 0)]
    other_q = p[p["income"] > 0].groupby("NOMER")["income"].median()
    add(g, "R02_no_income", 0.75, lambda d: [f"Доход не указан ни у одного члена семьи, хотя расходы составили {tg(s)}"
                                           + (f". В других кварталах 2024 года доход этой семьи указан ({approx(other_q[n])} за квартал) — похоже, доход пропущен при вводе"
                                              if n in other_q.index else "")
                                           for n, s in zip(d.NOMER, d.spending)],
        "Уточните у семьи все источники дохода за этот квартал.")
    g = p[(p["spending"] == 0) & (p["income"] > 0)]
    add(g, "R03_no_spending", 0.9, lambda d: [f"Не записано ни одной покупки или услуги, хотя доход семьи — {tg(i)}"
                                             for i in d.income], "Проверьте, не пропущены ли разделы расходов при вводе анкеты.")
    g = p[(p["income"] == 0) & (p["spending"] == 0)]
    add(g, "R13_empty", 0.5, lambda d: ["Анкета пустая: не указаны ни доходы, ни расходы"] * len(d),
        "Проверьте, проведено ли интервью и введены ли данные.")
    mm = m[(m["pension"] > 0) & (m["age"].between(0, 39))]
    add(mm, "R04_pension_young", 0.9, lambda d: [f"Пенсия по возрасту {tg(x)} у члена семьи №{n} (возраст — {years(a)}): до пенсионного возраста ещё далеко"
                                                for x, n, a in zip(d.pension, d.NOMP, d.age)],
        "Проверьте графу: возможно, это зарплата, пенсия по инвалидности или пособие, записанные не в ту графу.", nomp="NOMP")
    mm = m[(m["stipend"] > 0) & (m["age"] > 45)]
    add(mm, "R05_stipend_old", 0.4, lambda d: [f"Стипендия {tg(x)} у члена семьи №{n} (возраст — {years(a)})"
                                              for x, n, a in zip(d.stipend, d.NOMP, d.age)],
        "Проверьте, к какому члену семьи относится стипендия.", nomp="NOMP")
    mm = m[(m["wage"] > 0) & (m["wage"] < 30000)]
    typ = m.loc[m["wage"] > 0, "wage"].median()
    add(mm, "R06_wage_tiny", 0.9, lambda d: [f"Зарплата члена семьи №{n} за квартал — всего {tg(x)} (обычно {approx(typ)})"
                                            + (". Похоже, сумма записана в тысячах тенге" if x < 5000 else "")
                                            for x, n in zip(d.wage, d.NOMP)],
        lambda d: [f"Проверьте единицы: возможно, должно быть {tg(x * 1000)}." if x < 5000 else
                   "Уточните, не записана ли зарплата за один месяц вместо квартала или за неполный период работы."
                   for x in d.wage], nomp="NOMP")
    mm = m[(m["age"] < 0) | (m["age"] > 105)]
    add(mm, "R07_age", 0.95, lambda d: [f"Возраст члена семьи №{n} указан как {a:.0f} — такого значения быть не может"
                                        for n, a in zip(d.NOMP, d.age)],
        "Уточните возраст по списку членов семьи.", nomp="NOMP")
    kids = p.loc[p["children"] > 0, ["NOMER"]].drop_duplicates()
    cb = m[m["child_benefits"] > 0].groupby(KEY)["child_benefits"].sum().reset_index()
    cb = cb[~cb["NOMER"].isin(kids["NOMER"])]
    add(cb, "R08_child_benefit", 0.4, lambda d: [f"Указано пособие на детей ({tg(x)}), но в составе семьи нет детей до 18 лет"
                                                for x in d.child_benefits], "Сверьте состав семьи и вид пособия: возможно, ребёнок не внесён в список членов семьи.")
    amt = p.loc[(p[list(KODNU12.values())].sum(axis=1) == 0) & (p["n_answers"] > 0)]
    add(amt, "R09_orphan_answers", 0.9, lambda d: ["Есть ответы на вопросы 2.1–2.2 (цель и источник кредита), но ни одной суммы кредита или долга не указано"] * len(d),
        "Уточните сумму кредита или долга; если кредита не было — уберите ответы 2.1–2.2.")
    both = v[v["KOD"].notna() & v["KODNU"].notna()].drop_duplicates(KEY)
    add(both, "R10_kod_kodnu", 0.95, lambda d: ["В разделе 2 одна строка одновременно содержит и сумму, и ответ на вопрос 2.1/2.2"] * len(d),
        "Разнесите сумму и ответ по разным строкам.")
    bad = it[it["STOIMK"].isna() | (it["STOIMK"] <= 0)].drop_duplicates(KEY)
    add(bad, "R11_nonpositive", 0.95, lambda d: ["У покупки не указана стоимость, или она нулевая либо отрицательная"] * len(d),
        "Исправьте сумму.")
    dup = (it.groupby(KEY + ["table", "KODNU", "STOIMK"]).size().rename("n").reset_index())
    dup = dup[dup["n"] >= 3].sort_values("n").drop_duplicates(KEY, keep="last")
    names = raw["names"].set_index(["table", "kodnu"])["name"]
    add(dup, "R12_duplicates", 0.6, lambda d: [f"Покупка «{clean_name(names.get((t, k), k))}» на {tg(s)} записана {n} раза с одной и той же суммой"
                                              for t, k, s, n in zip(d.table, d.KODNU, d.STOIMK, d.n)],
        "Проверьте, не задвоены ли записи при вводе.")
    # R15: debt repayments that nothing could have paid for
    dout = p["credit_repaid"] + p["informal_repaid"]
    src = p["income"] + p["savings_used"] + p["credit_received"] + p["informal_borrowed"] + p["debt_returned_to_hh"]
    g = p[(dout > 0) & (dout > 1.5 * src)].assign(dout=dout, src=src)
    add(g, "R15_debt_unfunded", 0.8, lambda d: [
        f"Выплаты по кредитам и долгам — {tg(o)}, а доход, сбережения и новые займы вместе — {tg(s_)}"
        for o, s_ in zip(d.dout, d.src)],
        ("Уточните, из каких денег погашался долг. Если помогли родственники — раздел 3, строка «Материальная помощь в денежной форме»; "
         "если дали в долг — раздел 2, «Деньги, полученные в долг» (источник 7 — частное лицо); "
         "если из накоплений — раздел 2, «Сумма использованных сбережений»."))

    # R16: skip logic of section 4 printed on the D-004 form (2024 numbering of kv_vopr9)
    v9 = raw.get("v9")
    if v9 is not None and len(v9):
        ans = v9[v9["NOMVOPR"].isin(SKIP_Q)].drop_duplicates(KEY + ["NOMVOPR"]).pivot_table(
            index=KEY, columns="NOMVOPR", values="NOMOTV", aggfunc="first")
        present = v9.groupby(KEY)["NOMVOPR"].agg(set)
        rows = []
        for q_code, (q_txt, forbidden, f_txt) in SKIP_RULES.items():
            if q_code not in ans:
                continue
            no = ans.index[ans[q_code] == "2"]
            for u in no:
                bad = present.get(u, set()) & forbidden
                if bad:
                    rows.append((u[0], u[1], f"В вопросе {q_txt} ответ «Нет», но заполнены вопросы {f_txt}, "
                                              f"которые по бланку Д-004 в этом случае пропускаются"))
        if rows:
            add(pd.DataFrame(rows, columns=["NOMER", "q", "msg"]).drop_duplicates(KEY), "R16_skip_logic", 1.0,
                lambda d: list(d.msg), "Исправлено автоматически: ответы, которые по бланку должны быть пропущены, удалены.",
                layer="auto")

    # R17: codes outside the lists printed on the form / in the code books
    known = set(zip(raw["names"]["table"], raw["names"]["kodnu"]))
    bad_code = it[[(t, k) not in known for t, k in zip(it["table"], it["KODNU"])]]
    if "CELPOK" in it:
        bad_code = pd.concat([bad_code, it[(it["table"] == 1) & ~it["CELPOK"].isin(["1", "2", "9"])]])
    if "MESTPOK" in it:
        bad_code = pd.concat([bad_code, it[(it["table"] == 1) & ~it["MESTPOK"].isin(["1", "2", "3", "4", "9"])]])
    bad_code = pd.concat([bad_code[KEY], raw["hh"].loc[~raw["hh"]["K"].isin(["1", "2"]), KEY],
                          v[v["NOMVNEW"].isin(["21", "22"]) & ~v["KOD"].isin(list("123456789"))][KEY]]).drop_duplicates()
    add(bad_code, "R17_invalid_code", 0.9, lambda d: ["В анкете код не указан или его нет в бланке или справочнике Д-004 "
                                                      "(товар, цель или место покупки, тип населённого пункта или ответ 2.1–2.2)"] * len(d),
        "Сверьте коды с бланком и справочником.")

    # R18: D-004 income section vs D-008 household roster
    ro = raw["roster"]
    mm = m.merge(ro[["NOMER", "NOMP", "age", "sex"]], on=["NOMER", "NOMP"], how="left", suffixes=("", "_r"))
    miss = mm[mm["age_r"].isna() & mm["sex_r"].isna()]
    add(miss, "R18_roster", 0.8, lambda d: [f"Член семьи №{n} есть в разделе доходов, но его нет в составе семьи (форма Д-008)"
                                            for n in d.NOMP], "Сверьте список членов семьи в анкетах Д-004 и Д-008.", nomp="NOMP")
    mism = mm[mm["sex_r"].notna() & ((mm["sex"] != mm["sex_r"]) | ((mm["age"] - mm["age_r"]).abs() > 1))]
    add(mism, "R18_roster", 0.8, lambda d: [f"Пол или возраст члена семьи №{n} в Д-004 не совпадает с Д-008"
                                            for n in d.NOMP], "Сверьте пол и возраст в анкетах Д-004 и Д-008.", nomp="NOMP")

    # R20: income is recorded only for members aged 15+ (Instruction D 004, item 7)
    mm = m[(m["age"].between(0, 14)) & (m[INC_COLS].sum(axis=1) > 0)].assign(inc=lambda d: d[INC_COLS].sum(axis=1))
    add(mm, "R20_income_under15", 0.8, lambda d: [f"Доход {tg(x)} записан у члена семьи №{n} (возраст — {years(a)}), "
                                                  f"а по инструкции доходы записываются только с 15 лет"
                                                  for x, n, a in zip(d.inc, d.NOMP, d.age)],
        "Проверьте, к какому члену семьи относится доход.", nomp="NOMP")

    # R21: amounts must be whole tenge (Instruction D 004, item 5)
    frac = it[it["STOIMK"].notna() & (it["STOIMK"] % 1 != 0)].drop_duplicates(KEY)
    add(frac, "R21_not_whole", 0.9, lambda d: ["Сумма покупки записана не в целых тенге"] * len(d),
        "По инструкции суммы записываются в целых числах — проверьте, не перепутаны ли тенге и тиыны или разделитель.")

    # R22-R24: section 3 (income) vs section 4 (own production). Instruction D 004, item 7: sales of own
    # agricultural produce (4.5, 4.11, 4.12) are income from self-employment, "of which agricultural sales".
    s4 = raw.get("s4")
    if s4 is not None and len(s4):
        hh_agri = m.groupby(KEY)["of_which_agri_sales"].sum().rename("agri")
        x = (p[KEY + ["income"]].merge(s4.assign(sold=s4[["crop", "animal", "goods"]].sum(axis=1)), on=KEY, how="left")
               .merge(hh_agri.reset_index(), on=KEY, how="left").fillna({"sold": 0, "agri": 0}))
        g = x[(x.sold > 0) & (x.agri == 0)]
        what = lambda r: ", ".join(n for n, c in (("урожай (4.5)", "crop"), ("продукция животноводства (4.11)", "animal"),
                                                  ("изделия собственного производства (4.12)", "goods")) if r[c] > 0)
        add(g, "R22_agri_sales", 1.0, lambda d: [
            f"В разделе 4 продано на {tg(v)}: {what(r)}, а в разделе 3 доход от продажи сельхозпродукции не указан"
            for v, (_, r) in zip(d.sold, d.iterrows())],
            "Исправлено автоматически: по п. 7 Инструкции сумма продаж учтена в разделе 3 как доход от самозанятости, "
            "в том числе от продажи сельхозпродукции.", layer="auto")
        g = x[(x.agri > 0) & (x.sold == 0) & (x.income > 0) & (x.agri >= 0.2 * x.income)]
        add(g, "R23_agri_no_sales", 0.4, lambda d: [
            f"В разделе 3 указан доход от продажи сельхозпродукции {tg(a)} ({a / i:.0%} дохода семьи), "
            f"а в разделе 4 никаких продаж нет" for a, i in zip(d.agri, d.income)],
            "Уточните, что именно семья продавала, и заполните вопросы 4.5, 4.11 или 4.12 раздела 4.")
    mm = m[m["of_which_agri_sales"] > m["self_employed"]]
    add(mm, "R24_agri_gt_self", 1.0, lambda d: [
        f"У члена семьи №{n} доход «в том числе от продажи сельхозпродукции» ({tg(a)}) больше всего дохода "
        f"от самозанятости ({tg(s_)})" for n, a, s_ in zip(d.NOMP, d.of_which_agri_sales, d.self_employed)],
        "Исправлено автоматически: доход от самозанятости принят не меньше суммы продаж сельхозпродукции "
        "(часть не может быть больше целого).", nomp="NOMP", layer="auto")

    # R19: income jumps against the same household's other quarters
    g_ = p[["NOMER", "q", "income"]]
    o = g_.merge(g_, on="NOMER", suffixes=("", "_o"))
    o = o[(o["q"] != o["q_o"]) & (o["income_o"] > 0)].groupby(KEY)["income_o"].median().rename("oth").reset_index()
    j = g_.merge(o, on=KEY)
    j = j[(j["income"] > 0) & ((j["income"] * 10 < j["oth"]) | (j["income"] > 10 * j["oth"]))]
    add(j, "R19_income_jump", 0.5, lambda d: [
        f"Доход семьи в этом квартале — {tg(i)}, а в остальных кварталах 2024 года обычно {approx(o_)}: "
        f"в {times(max(i, o_) / min(i, o_))} {'меньше' if i < o_ else 'больше'}" for i, o_ in zip(d.income, d.oth)],
        "Уточните, не пропущен ли или не задвоен доход за этот квартал.")

    d2 = it.assign(dup=it.duplicated(KEY + ["table", "KODNU", "STOIMK"], keep=False)).groupby(KEY) \
        .agg(n=("dup", "size"), share=("dup", "mean")).reset_index()
    d2 = d2[(d2["n"] >= 6) & (d2["share"] >= .8)]
    add(d2, "R14_double_entry", 0.9, lambda d: [f"{s_:.0%} записей о покупках повторяются один в один (всего {n} записей)"
                                               for s_, n in zip(d.share, d.n)],
        "Похоже, раздел расходов введён дважды. Сверьте с бумажной анкетой.")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- layer 2: outliers
LBINS = np.linspace(0, 20, 201)   # log-tenge bins for per-item price densities


def item_features(x, raw):
    """Context of one purchase: what the household is like and how it prices its other purchases."""
    m = raw["members"]
    inc = m.groupby(KEY)[INC_COLS].sum().sum(axis=1).rename("inc")
    pers = raw["roster"].groupby("NOMER").size().rename("pers")
    hh = raw["hh"].set_index(KEY)[["TE", "K"]]
    c = x[KEY].join(inc, on=KEY).join(pers, on="NOMER").join(hh, on=KEY)
    r = (x["l"] - x["med"]) / x["mad"]
    tot, cnt = r.groupby([x["NOMER"], x["q"]]).transform("sum"), r.groupby([x["NOMER"], x["q"]]).transform("size")
    return pd.DataFrame({
        "hh_level": ((tot - r) / (cnt - 1).clip(lower=1)).fillna(0),          # leave-one-out household price level
        "linc": np.log1p(c["inc"].fillna(0) / c["pers"].fillna(1)), "pers": c["pers"].fillna(1),
        "urban": (c["K"] == "1").astype(int), "te": pd.to_numeric(c["TE"], errors="coerce").fillna(0),
        "q": x["q"], "table": x["table"], "med": x["med"], "mad": x["mad"], "n_items": cnt,
        "celpok": pd.to_numeric(x.get("CELPOK"), errors="coerce").fillna(0) if "CELPOK" in x else 0,
        "mest": pd.to_numeric(x.get("MESTPOK"), errors="coerce").fillna(0) if "MESTPOK" in x else 0,
    }, index=x.index)
def layer_outliers(raw, ref=None, zcut=5.0, top_share=0.08):
    """ref: reference distributions from clean data, so injected errors do not shift them"""
    it, m = raw["items"], raw["members"]
    names = raw["names"].set_index(["table", "kodnu"])["name"]
    x = it[it["STOIMK"] > 0].copy()
    x["l"] = np.log(x["STOIMK"])
    fit = ref is None
    if fit:
        ref = {"items": {k: np.sort(v.values) for k, v in x.groupby(["table", "KODNU"])["l"] if len(v) >= 30},
               "inc": {}}
        ref["dens"] = {k: np.histogram(v, bins=LBINS)[0] + 0.5 for k, v in ref["items"].items()}
    x = x[[k in ref["items"] for k in zip(x["table"], x["KODNU"])]].copy()
    tail = np.empty(len(x))
    med = np.empty(len(x))
    mad = np.empty(len(x))
    for k, gi in x.groupby(["table", "KODNU"]).indices.items():
        arr = ref["items"][k]
        vals = x["l"].values[gi]
        hi = np.searchsorted(arr, vals, side="right") / (len(arr) + 1)
        lo = np.searchsorted(arr, vals, side="left") / (len(arr) + 1)
        tail[gi] = np.maximum(-np.log10(np.clip(1 - hi, 1e-6, 1)), -np.log10(np.clip(lo, 1e-6, 1)))
        md = np.median(arr)
        med[gi] = md
        mad[gi] = 1.4826 * np.median(np.abs(arr - md)) or np.nan
    x["tail"], x["med"] = tail, med
    x["mad"] = pd.Series(mad, index=x.index).fillna(np.nanmedian(mad))
    x["ratio"] = np.exp(x["l"] - x["med"])
    x["z"] = (np.abs(x["l"] - x["med"]) / x["mad"]).clip(0, 30)

    # item-level model 1: "decimal shift" — is the amount divided by 10 much more typical for this item?
    lr = np.zeros(len(x))
    for k, gi in x.groupby(["table", "KODNU"]).indices.items():
        d = ref["dens"][k] / ref["dens"][k].sum()
        v = x["l"].values[gi]
        b0 = np.clip(np.digitize(v, LBINS) - 1, 0, len(d) - 1)
        b1 = np.clip(np.digitize(v - np.log(10), LBINS) - 1, 0, len(d) - 1)
        lr[gi] = np.log(d[b1] / d[b0])
    x["lr"] = lr

    # item-level model 2: gradient boosting predicts the usual price of THIS purchase from the household context
    Xi = item_features(x, raw)
    yi = ((x["l"] - x["med"]) / x["mad"]).clip(-10, 10)
    if fit:
        smp = np.random.default_rng(SEED).choice(len(x), min(400_000, len(x)), replace=False)
        ref["item_model"] = HistGradientBoostingRegressor(max_iter=200, learning_rate=.1, random_state=SEED).fit(
            Xi.iloc[smp], yi.iloc[smp])
    x["cr"] = yi - ref["item_model"].predict(Xi)
    x["score"] = x["tail"] + 0.3 * x["lr"].clip(lower=0) + 0.5 * x["cr"].clip(0, 20) + 0.01 * np.abs(x["l"] - x["med"])
    hh_sp = x.groupby(KEY)["STOIMK"].transform("sum")
    x["impact"] = ((x["STOIMK"] - np.exp(x["med"])).clip(lower=0) / hh_sp).clip(0, 1)
    f = x.sort_values("score", ascending=False).drop_duplicates(KEY)
    f["pct"] = f["score"].rank(pct=True) * len(f) / raw["hh"].shape[0] + (1 - len(f) / raw["hh"].shape[0])
    f = f[(f["pct"] >= 1 - top_share) & ((f["ratio"] >= 3) | (f["ratio"] <= 1 / 3))]
    share_of = lambda t: (f"больше, чем в {np.floor(1000 * (1 - 10 ** -t)) / 10:.1f}% записей по этой позиции".replace(".", ",")
                          if t < 3 else "больше, чем в 99,9% записей по этой позиции")
    nm = [clean_name(names.get((t, k), k)) for t, k in zip(f.table, f.KODNU)]
    zero = (f["ratio"] > 1) & (f["lr"] > 1.5)
    txt = [f"«{n}»: {tg(s)} — в {times(r)} больше обычного (обычно {approx(np.exp(md))}); {share_of(tl)}"
           + (f". Без одного нуля было бы {tg(s / 10)} — это типичная сумма для этой позиции" if zz else "")
           if r > 1 else f"«{n}»: {tg(s)} — в {times(1/r)} меньше обычного (обычно {approx(np.exp(md))})"
           for n, s, r, md, tl, zz in zip(nm, f.STOIMK, f.ratio, f.med, f["tail"], zero)]
    chk = [f"Похоже на лишний ноль: уточните, не {tg(s / 10)} ли на самом деле." if zz else
           "Проверьте сумму: нет ли лишнего нуля, не записана ли сумма сразу за несколько покупок?" if r > 1 else
           "Проверьте сумму: не пропущен ли ноль, не записана ли цена одной единицы вместо общей суммы?"
           for r, s, zz in zip(f.ratio, f.STOIMK, zero)]
    sig = [pd.DataFrame({"NOMER": f.NOMER.values, "q": f.q.values, "layer": "outliers", "code": "O1_item_price",
                         "p": np.clip(OUT_P[0] + (OUT_P[1] - OUT_P[0]) * (f.pct.values - (1 - top_share)) / top_share, *OUT_P)
                         * (0.6 + 0.4 * np.minimum(1, f.impact.values / 0.3)), "text": txt,
                         "check": chk, "nomp": np.nan, "detail": nm})]
    for col in ("wage", "pension"):
        y = m[m[col] > 0].copy()
        y["l"] = np.log(y[col])
        if col not in ref["inc"]:
            ref["inc"][col] = (y["l"].median(), np.median(np.abs(y["l"] - y["l"].median())))
        md, mad = ref["inc"][col]
        y["z"] = (y["l"] - md) / (1.4826 * mad)
        y = y[y["z"].abs() > zcut].sort_values("z", key=abs, ascending=False).drop_duplicates(KEY)
        sig.append(pd.DataFrame({"NOMER": y.NOMER.values, "q": y.q.values, "layer": "outliers", "code": f"O2_{col}",
                                 "p": np.clip(.5 + (y.z.abs().values - zcut) / 8, .5, .95),
                                 "text": [f"{INCOME_RU[col].capitalize()} члена семьи №{n} за квартал — {tg(v_)}, "
                                          f"а обычно {approx(np.exp(md))}" for n, v_ in zip(y.NOMP, y[col])],
                                 "check": "Проверьте сумму и единицы измерения (тенге или тысячи тенге).",
                                 "nomp": y.NOMP.values.astype(float)}))
    return pd.concat(sig, ignore_index=True), ref


# ---------------------------------------------------------------- layer 3: ML
M1_PREFIX = "Редкое сочетание показателей: "


def m2_text(s, e, n):
    if s >= e:
        return f"Расходы {tg(s)} — примерно в {times(s / max(e, 1))} больше, чем обычно у похожих семей ({approx(e)})"
    return (f"Расходы всего {tg(s)} ({plural(n, ('запись', 'записи', 'записей'))}) — примерно в {times(e / max(s, 1))} меньше, "
            f"чем обычно у похожих семей ({approx(e)})")


def m2_check(s, e):
    return ("Сверьте крупные покупки с бумажной анкетой: нет ли лишних нулей или задвоенных записей?" if s >= e
            else "Проверьте, не пропущены ли целые разделы расходов при вводе.")
def layer_ml(p, top_share=0.02, iso_model=None):
    p = p[(p["income"] > 0) | (p["spending"] > 0)].reset_index(drop=True)
    f = features(p)
    if iso_model is None:
        iso_model = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(f)
    s = -iso_model.score_samples(f)
    pct = pd.Series(s).rank(pct=True).values
    def z_col(c):
        v = c.values.astype(float)
        mad = np.median(np.abs(v - np.median(v)))
        if mad > 0:
            return robust_z(v)
        return (v - v.mean()) / v.std() if v.std() > 0 else 0 * v
    zf = f.apply(z_col)
    zf = zf.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-50, 50)
    flag = pct >= 1 - top_share
    rows = []
    for i in np.where(flag)[0]:
        zi = zf.iloc[i].drop("urban")
        top = [c for c in zi.abs().sort_values(ascending=False).index[:2] if abs(zi[c]) >= 2]
        parts = [f"{FEATURE_RU[c][0]} — {FEATURE_RU[c][1][0] if zi[c] > 0 else FEATURE_RU[c][1][1]}, чем у большинства семей"
                 for c in top] or ["несколько показателей одновременно отличаются от типичных"]
        rows.append((p.NOMER.iat[i], p.q.iat[i], "ml", "M1_isolation", .4 + .4 * (pct[i] - (1 - top_share)) / top_share,
                     M1_PREFIX + "; ".join(parts),
                     "Такое сочетание ответов встречается редко — просмотрите анкету целиком.", np.nan, "|".join(parts)))
    sig = pd.DataFrame(rows, columns=["NOMER", "q", "layer", "code", "p", "text", "check", "nomp", "detail"])

    X = pd.DataFrame({"li": np.log1p(p["income"]), "persons": p["persons"], "children": p["children"],
                      "elderly": p["elderly"], "urban": p["urban"], "te": p["TE"].astype(int), "q": p["q"],
                      "loan": (p["credit_received"] + p["informal_borrowed"] > 0).astype(int),
                      "sav": (p["savings_used"] > 0).astype(int)})
    y = np.log1p(p["spending"])
    hgb = HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=SEED,
                                        categorical_features=[X.columns.get_loc("te")])
    pred = cross_val_predict(hgb, X, y, cv=5)
    z = robust_z((y - pred).values)
    idx = np.where(np.abs(z) > 3.5)[0]
    exp_sp = np.expm1(pred)
    sig2 = pd.DataFrame({"NOMER": p.NOMER.values[idx], "q": p.q.values[idx], "layer": "ml", "code": "M2_expected_spending",
                         "p": np.clip(.4 + (np.abs(z[idx]) - 3.5) / 10, .4, .85),
                         "text": [m2_text(s, e, n) for s, e, n in zip(p.spending.values[idx], exp_sp[idx], p.n_items.values[idx])],
                         "check": [m2_check(s, e) for s, e in zip(p.spending.values[idx], exp_sp[idx])],
                         "nomp": np.nan, "detail": [f"{s}|{e}" for s, e in zip(p.spending.values[idx], exp_sp[idx])]})
    return pd.concat([sig, sig2], ignore_index=True), iso_model, None


# ---------------------------------------------------------------- scoring
def score(p, signals):
    """Risk for the interviewer queue. Deterministic auto-corrections (layer "auto") are logged, not queued."""
    auto = signals[signals["layer"] == "auto"].groupby(KEY).size().rename("auto_fixes")
    signals = signals[signals["layer"] != "auto"]
    s = signals.groupby(KEY)["p"].apply(lambda x: 1 - np.prod(1 - x.values)).rename("risk") * 100
    out = p[KEY + ["TE", "K", "persons", "income", "spending"]].merge(s.reset_index(), on=KEY, how="left")
    out["risk"] = out["risk"].fillna(0)
    layers = signals.groupby(KEY)["layer"].agg(lambda x: ",".join(sorted(set(x)))).rename("layers")
    out = out.merge(layers.reset_index(), on=KEY, how="left").merge(auto.reset_index(), on=KEY, how="left")
    out["auto_fixes"] = out["auto_fixes"].fillna(0).astype(int)
    return out.sort_values("risk", ascending=False, kind="mergesort").reset_index(drop=True)


def run(raw, ref=None, iso_model=None):
    p = build_profile(raw)
    s1 = layer_rules(raw, p)
    s2, ref = layer_outliers(raw, ref)
    s3, iso_model, _ = layer_ml(p, iso_model=iso_model)
    sig = pd.concat([s1, s2, s3], ignore_index=True)
    return p, sig, score(p, sig), ref, iso_model


# ---------------------------------------------------------------- explanation card
LAYER_RU = {"rules": "правило", "outliers": "выброс", "ml": "ИИ-модель", "auto": "автоисправление"}


def card_lines(su, prow, big=None):
    """su: signals of one questionnaire; prow: its profile row; big: (item name, amount) of the largest purchase.
    Returns [(tag, text, check)] with duplicates merged."""
    su = su[su.layer != "auto"].sort_values("p", ascending=False)
    codes = set(su.code)
    lines = []
    # one cause, one line: tiny wage found by both the rule and the outlier layer
    r06 = set(su.loc[su.code == "R06_wage_tiny", "nomp"].dropna())
    su = su[~((su.code == "O2_wage") & su.nomp.isin(r06))]
    o1_items = set(su.loc[su.code == "O1_item_price", "detail"])
    big_txt = (f" Крупнейшая запись — «{clean_name(big[0])}» на {tg(big[1])}."
               if big and clean_name(big[0]) not in o1_items else "")
    if {"R01_gap", "M2_expected_spending"} <= codes:
        s_, e_ = map(float, su.loc[su.code == "M2_expected_spending", "detail"].iat[0].split("|"))
        i_ = prow["income"]
        lines.append(("правило + ИИ-модель",
                      f"Расходы {tg(s_)}: в {ratio_txt(s_ / i_)} больше дохода семьи ({tg(i_)}) и примерно в "
                      f"{times(s_ / e_)} больше, чем обычно у похожих семей ({approx(e_)}). Кредиты, долги и сбережения "
                      f"не указаны.{big_txt}",
                      "Уточните, на какие деньги сделаны крупные покупки (кредит, сбережения, помощь родственников), "
                      "и сверьте их суммы с бумажной анкетой."))
        su = su[~su.code.isin(["R01_gap", "M2_expected_spending"])]
        big_txt = ""
    no_income = bool(codes & {"R02_no_income", "R13_empty"})
    spend_explained = bool(codes & {"R01_gap", "M2_expected_spending"})
    for _, r in su.iterrows():
        text = r.text
        if r.code == "M1_isolation":
            parts = [x for x in str(r.detail).split("|")
                     if not (no_income and x.startswith("доход на человека"))
                     and not (spend_explained and x.startswith("расходы на человека"))]
            if not parts:
                continue
            text = M1_PREFIX + "; ".join(parts)
        if r.code in ("R01_gap", "M2_expected_spending") and big_txt and prow["spending"] > prow.get("income", 0):
            text = text + "." + big_txt
            big_txt = ""
        elif not text.endswith("."):
            text += "."
        lines.append((LAYER_RU[r.layer], text, r.check))
    return lines


def risk_label(r):
    return f"риск {min(99, round(r))}/100"
