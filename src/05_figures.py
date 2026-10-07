import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import RES, FIG, CODE_RU

BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#b9b8b2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GREY,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})

ev = pd.read_csv(RES / "t3_evaluation.csv", index_col=0)
e = ev.drop("ALL").sort_values("recall@5%")
fig, ax = plt.subplots(figsize=(8, 3.9))
AQUA = "#1baf7a"
ax.barh(e.description, e["recall@5%"], height=.62,
        color=[AQUA if a > .9 else ORANGE if v < .5 else BLUE for v, a in zip(e["recall@5%"], e["auto_fixed"])])
for i, v in enumerate(e["recall@5%"]):
    ax.text(v + .01, i, f"{v:.0%}", va="center", fontsize=10, color=INK)
    if e["auto_fixed"].iloc[i] > .9:
        ax.text(.02, i, "исправлено автоматически", va="center", fontsize=8.5, color="white", fontweight="bold")
ax.axvline(.05, color=MUTED, ls="--", lw=1)
ax.text(.06, -.75, "случайная проверка 5%", fontsize=9, color=MUTED)
ax.set_xlim(0, 1.12); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax.grid(axis="x", color="#e6e5e0"); ax.set_axisbelow(True)
ax.set_title("Доля внесённых ошибок: исправлено автоматически или найдено в топ-5% по риску",
             loc="left", fontsize=11.5, color=INK)
fig.savefig(FIG / "f1_evaluation.png"); plt.close(fig)

# f2: how many checks are needed and how much distortion remains
im = pd.read_csv(RES / "t5_impact.csv")
STYLE = {"random": ("случайная проверка", GREY, "-"), "rules": ("только правила", ORANGE, "-"),
         "system": ("вся система", BLUE, "-")}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for m, (lab, col, ls) in STYLE.items():
    g = im[im.method == m]
    a1.plot(g.budget, g.recall, color=col, lw=2.4 if m == "system" else 1.8, label=lab)
    a2.plot(g.budget, g.spending_abs_err_pct, color=col, lw=2.4 if m == "system" else 1.8, label=lab)
for a in (a1, a2):
    a.axvline(.05, color=MUTED, ls="--", lw=1); a.set_xticks([0, .05, .1, .15, .2])
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    a.grid(color="#e6e5e0"); a.set_axisbelow(True); a.set_xlabel("доля перепроверенных анкет", color=MUTED)
a1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0)); a1.set_ylim(0, 1)
a1.set_title("Найдено внесённых ошибок", loc="left", fontsize=11.5, color=INK)
a2.set_title("Искажение расходов домохозяйств, % от суммы", loc="left", fontsize=11.5, color=INK)
a2.set_ylim(0, None); a1.legend(frameon=False, fontsize=9.5, loc="center right")
fig.tight_layout(); fig.savefig(FIG / "f2_impact.png"); plt.close(fig)

# f3: closed loop — learning from interviewers' verdicts (budget 2% per quarter)
fl = pd.read_csv(RES / "t7_feedback_loop.csv")
fl = fl[fl.budget == .02]
rc = pd.read_csv(RES / "t8_rule_confirmation.csv").sort_values("confirmed")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.25]})
x = fl.q.values
a1.bar(x - .2, fl.found_static, .38, color=GREY, label="без обучения")
a1.bar(x + .2, fl.found_learned, .38, color=BLUE, label="с обучением на отметках")
for xi, s0, s1 in zip(x, fl.found_static, fl.found_learned):
    if s1 > s0 * 1.01:
        a1.text(xi + .2, s1 + 3, f"+{s1 / s0 - 1:.0%}", ha="center", fontsize=10, color=BLUE, fontweight="bold")
a1.set_xticks(x, [f"{q} кв." for q in x]); a1.set_ylim(0, fl.found_learned.max() * 1.35)
a1.set_title("Найдено ошибок при проверке 2% анкет", loc="left", fontsize=11.5, color=INK)
a1.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2); a1.grid(axis="y", color="#e6e5e0"); a1.set_axisbelow(True)
names = [CODE_RU.get(c, c) for c in rc.code]
a2.barh(names, rc.confirmed, height=.62, color=[ORANGE if v < .5 else BLUE for v in rc.confirmed])
for i, v in enumerate(rc.confirmed):
    a2.text(v + .01, i, f"{v:.0%}", va="center", fontsize=9, color=INK)
a2.set_xlim(0, 1.12); a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
a2.tick_params(axis="y", labelsize=9); a2.grid(axis="x", color="#e6e5e0"); a2.set_axisbelow(True)
a2.set_title("Доля подтверждённых ошибок по типу сигнала (симуляция)", loc="left", fontsize=11.5, color=INK)
fig.tight_layout(); fig.savefig(FIG / "f3_feedback.png"); plt.close(fig)
print("ok")
