import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATES = [ROOT / "data", ROOT.parent.parent / "synthetic_microdata_2021-2024_20260921"]
DATA_DIR = Path(os.environ.get("DATATHON_DATA") or next(
    (c for c in _CANDIDATES if (c / "MANIFEST.csv").exists()), _CANDIDATES[0]))
OUT = ROOT / "data_processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
for p in (OUT, RES, FIG):
    p.mkdir(exist_ok=True)

YEAR = 2024
QUARTERS = ["1kv", "2kv", "3kv", "4kv"]
SEED = 2026

KODNU12 = {
    "59": "credit_received", "2.1.1.4.1": "credit_repaid", "60": "savings_used",
    "61": "informal_borrowed", "2.1.1.4.2": "informal_repaid", "62": "debt_returned_to_hh",
}
INCOME = {
    1: "wage", 2: "self_employed", 3: "of_which_agri_sales", 4: "pension", 5: "stipend",
    6: "targeted_social_aid", 7: "housing_aid", 8: "child_benefits", 9: "state_social_benefit",
    10: "special_state_benefit", 11: "other_state_transfers", 12: "social_transfers_in_kind",
    13: "material_aid_cash", 14: "material_aid_in_kind", 15: "other_transfers", 16: "alimony",
    17: "sale_real_estate", 18: "sale_personal_property", 19: "property_income",
    20: "rent_housing", 21: "rent_land", 22: "other_receipts", 23: "subsidies",
}
INCOME_RU = {
    "wage": "зарплата", "self_employed": "доход от самозанятости", "pension": "пенсия по возрасту",
    "stipend": "стипендия", "child_benefits": "пособие на детей",
}
TABLE_RU = {1: "непродовольственные товары", 2: "коммунальные услуги", 3: "связь", 4: "образование",
            5: "здравоохранение", 6: "услуги, отдых, прочие расходы", 7: "транспорт"}
REGION = {
    "10": "Абайская", "11": "Акмолинская", "15": "Актюбинская", "19": "Алматинская",
    "23": "Атырауская", "27": "ЗКО", "31": "Жамбылская", "33": "Жетісу",
    "35": "Карагандинская", "39": "Костанайская", "43": "Кызылординская",
    "47": "Мангистауская", "55": "Павлодарская", "59": "СКО", "61": "Туркестанская",
    "62": "Улытау", "63": "ВКО", "71": "г. Астана", "75": "г. Алматы", "79": "г. Шымкент",
}


def read(path, **kw):
    import pandas as pd
    return pd.read_csv(DATA_DIR / path, dtype=str, **kw)

# Short Russian names of the checks (web app, figures)
CODE_RU = {
    "R01_gap": "Расходы намного больше дохода", "R02_no_income": "Не указан доход", "R03_no_spending": "Нет расходов",
    "R04_pension_young": "Пенсия у молодого", "R05_stipend_old": "Стипендия у пожилого", "R06_wage_tiny": "Слишком маленькая зарплата",
    "R07_age": "Недопустимый возраст", "R08_child_benefit": "Пособие на детей без детей", "R09_orphan_answers": "Ответы о кредите без суммы",
    "R10_kod_kodnu": "Сумма и ответ в одной строке", "R11_nonpositive": "Нулевая или отрицательная сумма",
    "R12_duplicates": "Повторы одной покупки", "R13_empty": "Пустая анкета", "R14_double_entry": "Двойной ввод расходов",
    "R15_debt_unfunded": "Долг без источника денег", "R16_skip_logic": "Нарушение переходов бланка (п. 8)",
    "R17_invalid_code": "Недопустимый код", "R18_roster": "Расхождение с Д-008", "R19_income_jump": "Скачок дохода между кварталами",
    "R20_income_under15": "Доход у ребёнка до 15 лет", "R21_not_whole": "Сумма не в целых тенге",
    "R22_agri_sales": "Продажи из раздела 4 не учтены в доходах (п. 7)", "R23_agri_no_sales": "Доход от продажи сельхозпродукции без продаж",
    "R24_agri_gt_self": "«В том числе» больше целого",
    "O1_item_price": "Необычная цена покупки", "O2_wage": "Необычная зарплата", "O2_pension": "Необычная пенсия",
    "M1_isolation": "ИИ: редкое сочетание ответов", "M2_expected_spending": "ИИ: расходы не как у похожих семей",
}
