import os
try: os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from the folder that contains data/
except NameError: pass
# =============================================================================
# verify_paper.py  (part 2 of 2) - run after run_paper.py
# Draws Fig. 1 of the paper (code relations), prints every number quoted in the
# manuscript and supplement, and writes output/VERIFY_REPORT.txt (PASS/CHECK).
# Takes about 1 min. Send output/VERIFY_REPORT.txt back for checking.
# =============================================================================
import os, pickle, time as _time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
_T0 = _time.time(); OUT = "output"
globals().update(pickle.load(open(os.path.join(OUT, "checkpoint_for_verify.pkl"), "rb")))
BLUE, ORANGE, INK, MUTED, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#b5b4ae"; COL = {"flex": BLUE, "dts": ORANGE}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def save(f, n): f.savefig(os.path.join(OUT, n + ".png"), dpi=600, bbox_inches="tight"); f.savefig(os.path.join(OUT, n + ".pdf"), bbox_inches="tight")
def mean_score(k, r):
    d = sets[k]; y = d[TARGET[k]].values; p = np.full(len(y), np.nan)
    for tr, te in folds[k]["grouped"][r]: p[te] = y[tr].mean()
    return r2_score(y, p), np.mean(p / y)


# =============================================================================
# STEP 12 (new) - Fig. 1 code relations, extra statistics, verification report
# =============================================================================
print("\n===== STEP 12: extra statistics and verification =====")
REPORT = []
def chk(label, val, target, tol):
    try: val = float(val)
    except Exception: val = np.nan
    ok = (not np.isnan(val)) and abs(val - target) <= tol
    REPORT.append(f"{'PASS ' if ok else 'CHECK'} | {label:62s} | code {val:10.3f} | paper {target:9.3f} | tol {tol}")

# ---- Fig. 1 (paper): code and guideline relations, predicted vs measured ----
try:
    fig, ax = plt.subplots(2, 2, figsize=(7, 6.6))
    for a, (name, k, p) in zip(ax.ravel(), [REL[0], REL[1], REL[2], REL[3]]):
        yv = (fl["f_flex"] if k == "flex" else dt["f_dts"]).values; pv = np.asarray(p, float)
        lim = [0, max(yv.max(), pv.max()) * 1.05]
        a.scatter(yv, pv, s=10, color=COL[k], alpha=0.55); a.plot(lim, lim, color=INK, lw=0.8)
        a.plot(lim, [v * 1.2 for v in lim], ls="--", color=MUTED, lw=0.6); a.plot(lim, [v * 0.8 for v in lim], ls="--", color=MUTED, lw=0.6)
        a.set_xlim(lim); a.set_ylim(lim); a.set_title(name, fontsize=8); a.set_xlabel("Measured (MPa)"); a.set_ylabel("Predicted (MPa)")
    fig.tight_layout(); save(fig, "Fig1_paper_code_relations")
except Exception as _e:
    REPORT.append('ERROR | section "Fig. 1 (paper): code and guideline relations, predicted vs measured" failed: ' + repr(_e))

# ---- Data description (Methods, Table S2) ----
try:
    fx, ts = sets["flex"], sets["dts"]
    chk("flexural records", len(fx), 982, 0); chk("flexural studies", fx["source_id"].nunique(), 89, 0)
    chk("tensile records", len(ts), 243, 0); chk("tensile studies", ts["source_id"].nunique(), 38, 0)
    chk("flex steel-fiber records", fx["is_steel"].sum(), 648, 0)
    chk("flex plain (Vf = 0) records", (fx["vf_pct"] == 0).sum(), 241, 0)
    chk("flex PE records", fx["is_pe"].sum(), 60, 0); chk("flex PVA records", fx["is_pva"].sum(), 24, 0)
    chk("flex steam/heat cured", fx["cure_steam"].sum(), 270, 0); chk("flex autoclaved", fx["cure_auto"].sum(), 31, 0)
    chk("flex 40x40x160 prisms", ((fx["prism_depth"] == 40) & (fx["prism_length"] == 160)).sum(), 633, 0)
    vcf = fx["source_id"].value_counts(); chk("flex median records/study", vcf.median(), 8, 0); chk("flex max records/study", vcf.max(), 71, 0)
    chk("tensile median records/study", ts["source_id"].value_counts().median(), 4, 0)
    chk("tensile steel records", ts["is_steel"].sum(), 128, 0); chk("tensile PE records", ts["is_pe"].sum(), 102, 0)
    sv = fx.loc[fx["is_steel"] == 1, "vf_pct"]
    chk("median steel-fiber volume (%)", sv.median(), 1.99, 0.01)
    chk("steel Vf 5th percentile (%)", sv.quantile(0.05), 0.79, 0.01); chk("steel Vf 95th percentile (%)", sv.quantile(0.95), 2.98, 0.01)
    chk("tensile min strength (MPa)", ts["f_dts"].min(), 4.4, 0.05); chk("tensile max strength (MPa)", ts["f_dts"].max(), 43.2, 0.05)
except Exception as _e:
    REPORT.append('ERROR | section "Data description (Methods, Table S2)" failed: ' + repr(_e))

# ---- Code relations (Table 1, Table S3) ----
try:
    paperT1 = [(-3.70, 0.34, 0.57), (0.30, 1.18, 0.50), (-0.58, 0.73, 0.37), (-1.09, 0.58, 0.35), (-0.25, 0.83, 0.38)]
    for (i, row_), (r2p, bp, cp) in zip(T2A.iterrows(), paperT1):
        chk(f"T1 {row_['relation'][:28]} R2", row_["R2"], r2p, 0.01); chk(f"T1 {row_['relation'][:28]} ratio", row_["mean pred/meas"], bp, 0.01); chk(f"T1 {row_['relation'][:28]} CoV", row_["CoV"], cp, 0.01)
    for (i, row_), tgt in zip(T2B.iterrows(), [12.8, 6.2, 0.0]): chk(f"S3 {row_['provision'][:40]} %", row_["violating %"], tgt, 0.05)
    chk("S3 NF records with fc >= 150", T2B.iloc[2]["n"], 96, 0)
except Exception as _e:
    REPORT.append('ERROR | section "Code relations (Table 1, Table S3)" failed: ' + repr(_e))

# ---- Protocol effect (Table 2) ----
try:
    chk("exposure flex random", exposure[("flex", "random")], 0.993, 0.001); chk("exposure dts random", exposure[("dts", "random")], 0.975, 0.001)
    g2 = R.groupby(["target", "model", "protocol"])[["R2", "RMSE", "bias"]].agg(["mean", "std"])
    chk("CatBoost flex R2 random", g2.loc[("flex", "CatBoost", "random"), ("R2", "mean")], 0.87, 0.005)
    chk("CatBoost flex R2 grouped", g2.loc[("flex", "CatBoost", "grouped"), ("R2", "mean")], 0.39, 0.005)
    chk("CatBoost flex R2 grouped SD", g2.loc[("flex", "CatBoost", "grouped"), ("R2", "std")], 0.04, 0.005)
    chk("CatBoost flex RMSE grouped", g2.loc[("flex", "CatBoost", "grouped"), ("RMSE", "mean")], 7.40, 0.01)
    chk("CatBoost dts R2 random", g2.loc[("dts", "CatBoost", "random"), ("R2", "mean")], 0.80, 0.005)
    chk("CatBoost dts R2 grouped", g2.loc[("dts", "CatBoost", "grouped"), ("R2", "mean")], -0.03, 0.005)
    chk("CatBoost dts R2 grouped SD", g2.loc[("dts", "CatBoost", "grouped"), ("R2", "std")], 0.12, 0.005)
    chk("XGBoost dts R2 grouped SD", g2.loc[("dts", "XGBoost", "grouped"), ("R2", "std")], 0.96, 0.005)
    chk("StudyMean flex R2 random", g2.loc[("flex", "StudyMean", "random"), ("R2", "mean")], 0.57, 0.005)
    fam = ["CatBoost", "XGBoost", "LightGBM", "RF", "MLP"]
    loss = [(g2.loc[("flex", m, "random"), ("R2", "mean")] - g2.loc[("flex", m, "grouped"), ("R2", "mean")]) / g2.loc[("flex", m, "random"), ("R2", "mean")] * 100 for m in fam]
    chk("flex R2 loss, min over families (%)", min(loss), 55, 0.5); chk("flex R2 loss, max over families (%)", max(loss), 71, 0.5)
    sdg = [g2.loc[("flex", m, "grouped"), ("R2", "std")] for m in fam]
    chk("flex grouped SD, min over families", min(sdg), 0.02, 0.005); chk("flex grouped SD, max over families", max(sdg), 0.08, 0.005)
    bf = [g2.loc[("flex", m, "grouped"), ("bias", "mean")] for m in fam]; bt = [g2.loc[("dts", m, "grouped"), ("bias", "mean")] for m in fam]
    br = [g2.loc[(k, m, "random"), ("bias", "mean")] for k in ["flex", "dts"] for m in fam]
    chk("bias grouped flex min", min(bf), 1.09, 0.005); chk("bias grouped flex max", max(bf), 1.12, 0.005)
    chk("bias grouped dts min", min(bt), 1.12, 0.005); chk("bias grouped dts max", max(bt), 1.22, 0.005)
    chk("bias random min (all)", min(br), 1.03, 0.005); chk("bias random max (all)", max(br), 1.06, 0.005)
    chk("all dts grouped R2 negative (1 = yes)", float(all(g2.loc[("dts", m, "grouped"), ("R2", "mean")] < 0 for m in fam)), 1, 0)
    chk("XGBoost leads random flex (1 = yes)", float(max(fam, key=lambda m: g2.loc[("flex", m, "random"), ("R2", "mean")]) == "XGBoost"), 1, 0)
    chk("CatBoost leads grouped flex (1 = yes)", float(max(fam, key=lambda m: g2.loc[("flex", m, "grouped"), ("R2", "mean")]) == "CatBoost"), 1, 0)
    gb = B.groupby(["target", "model", "protocol"])[["R2", "RMSE"]].mean()
    chk("B3 flex R2 random", gb.loc[("flex", "B3 fc+RI+depth", "random"), "R2"], 0.39, 0.005)
    chk("B3 flex R2 grouped", gb.loc[("flex", "B3 fc+RI+depth", "grouped"), "R2"], 0.30, 0.005)
    chk("B3 flex RMSE grouped", gb.loc[("flex", "B3 fc+RI+depth", "grouped"), "RMSE"], 7.96, 0.01)
    chk("CatBoost minus B3, random (R2)", g2.loc[("flex", "CatBoost", "random"), ("R2", "mean")] - gb.loc[("flex", "B3 fc+RI+depth", "random"), "R2"], 0.48, 0.01)
    chk("CatBoost minus B3, grouped (R2)", g2.loc[("flex", "CatBoost", "grouped"), ("R2", "mean")] - gb.loc[("flex", "B3 fc+RI+depth", "grouped"), "R2"], 0.09, 0.01)
except Exception as _e:
    REPORT.append('ERROR | section "Protocol effect (Table 2)" failed: ' + repr(_e))

# ---- Nested tuning (Table S4) and chosen configurations ----
try:
    t9 = ALL.groupby(["target", "config", "protocol"])["R2"].mean()
    chk("tuning flex random fixed", t9[("flex", "fixed", "random")], 0.869, 0.001); chk("tuning flex random tuned", t9[("flex", "tuned (nested)", "random")], 0.889, 0.001)
    chk("tuning flex grouped fixed", t9[("flex", "fixed", "grouped")], 0.403, 0.001); chk("tuning flex grouped tuned", t9[("flex", "tuned (nested)", "grouped")], 0.381, 0.001)
    chk("tuning dts grouped fixed", t9[("dts", "fixed", "grouped")], 0.04, 0.005); chk("tuning dts grouped tuned", t9[("dts", "tuned (nested)", "grouped")], 0.08, 0.005)
    best["cfg_txt"] = best["cfg"].map(lambda c: str(GRID[c]))
    print("\nMost frequent tuned configuration by target/protocol:")
    print(best.groupby(["target", "protocol"])["cfg_txt"].agg(lambda s: s.value_counts().index[0]))
except Exception as _e:
    REPORT.append('ERROR | section "Nested tuning (Table S4) and chosen configurations" failed: ' + repr(_e))

# ---- Variance share (Fig. 3) ----
try:
    for v, tgt in [("prism_depth", 1.00), ("prism_length", 1.00), ("l_d", 0.90), ("sp_b", 0.87), ("RI", 0.86), ("sand_b", 0.81), ("sf_b", 0.78), ("fc_28", 0.71), ("w_b", 0.70), ("vf_pct", 0.55), ("curing_temp", 0.56), ("f_flex", 0.66)]:
        chk(f"variance share flex {v}", VS.loc[v, "flex"], tgt, 0.005)
    for v, tgt in [("sand_b", 0.99), ("curing_temp", 0.84), ("vf_pct", 0.45), ("f_dts", 0.75), ("sp_b", 0.89), ("w_b", 0.83)]:
        chk(f"variance share dts {v}", VS.loc[v, "dts"], tgt, 0.005)
except Exception as _e:
    REPORT.append('ERROR | section "Variance share (Fig. 3)" failed: ' + repr(_e))

# ---- Learning curve (Table S5) ----
try:
    lcm = LC.groupby("studies")["R2"].agg(["mean", "std"]); ks_ = sorted(lcm.index)
    chk("learning curve R2, 10 studies", lcm.loc[ks_[0], "mean"], 0.15, 0.005); chk("learning curve SD, 10 studies", lcm.loc[ks_[0], "std"], 0.27, 0.005)
    chk("learning curve R2, 40 studies", lcm.loc[40, "mean"], 0.39, 0.01); chk("learning curve R2, 69 studies", lcm.loc[ks_[-1], "mean"], 0.42, 0.005)
except Exception as _e:
    REPORT.append('ERROR | section "Learning curve (Table S5)" failed: ' + repr(_e))

# ---- Selection and ablation (Figs. S4-S5) ----
try:
    sm = SEL.groupby(["target", "set"])[["R2", "bias"]].agg(["mean", "std"])
    print("\nSelected inputs:", sel)
    chk("flex selected-set size", len(sel["flex"]), 6, 0); chk("dts selected-set size", len(sel["dts"]), 0, 0)
    chk("flex selected R2", sm.loc[("flex", "selected"), ("R2", "mean")], 0.425, 0.001); chk("flex selected R2 SD", sm.loc[("flex", "selected"), ("R2", "std")], 0.039, 0.001)
    chk("flex complete R2 (reps 1-9)", sm.loc[("flex", "complete"), ("R2", "mean")], 0.393, 0.001)
    chk("flex selected bias", sm.loc[("flex", "selected"), ("bias", "mean")], 1.100, 0.001); chk("flex complete bias", sm.loc[("flex", "complete"), ("bias", "mean")], 1.119, 0.001)
    chk("dts fc only R2", sm.loc[("dts", "fc only"), ("R2", "mean")], -0.13, 0.005); chk("dts fc + Vf R2", sm.loc[("dts", "fc + Vf"), ("R2", "mean")], -0.31, 0.005)
    chk("dts constant predictor R2 (selection folds)", mean_score("dts", 0)[0], -0.009, 0.001)
    am = ABm["mean"]
    chk("flex ablation complete R2", am[("flex", "complete")], 0.394, 0.001); chk("flex ablation minus sp_b R2", am[("flex", "minus sp_b")], 0.407, 0.001)
    for v, tgt in [("sp_b", -0.11), ("w_b", -0.10), ("sf_b", 0.06), ("vf_pct", 0.03)]:
        chk(f"dts ablation change, minus {v}", am[("dts", f"minus {v}")] - am[("dts", "complete")], tgt, 0.005)
    chk("dts ablation |change|, minus fc_28 (< 0.01)", abs(am[("dts", "minus fc_28")] - am[("dts", "complete")]), 0.0, 0.01)
except Exception as _e:
    REPORT.append('ERROR | section "Selection and ablation (Figs. S4-S5)" failed: ' + repr(_e))

# ---- SHAP (Table S6) ----
try:
    sf_ = shap_tab["flex"]
    chk("SHAP flex fc_28", sf_["fc_28"], 2.60, 0.005); chk("SHAP flex vf_pct", sf_["vf_pct"], 1.77, 0.005)
    fib = sorted([sf_["vf_pct"], sf_["l_d"], sf_["RI"]], reverse=True)
    chk("SHAP flex 2nd fiber descriptor", fib[1], 1.20, 0.005); chk("SHAP flex 3rd fiber descriptor", fib[2], 0.87, 0.005)
    chk("SHAP flex geometry total", sf_["prism_depth"] + sf_["prism_length"], 1.83, 0.01)
    chk("SHAP dts fc_28", shap_tab["dts"]["fc_28"], 0.93, 0.005); chk("SHAP dts w_b", shap_tab["dts"]["w_b"], 0.82, 0.005)
    dep = SV["flex"][:, FEAT["flex"].index("prism_depth")]; pdv = sets["flex"]["prism_depth"].values
    print("\nSHAP prism depth: max at 40 mm = %.2f, min at 100 mm = %.2f" % (np.nanmax(dep[pdv == 40]), np.nanmin(dep[pdv == 100])))
except Exception as _e:
    REPORT.append('ERROR | section "SHAP (Table S6)" failed: ' + repr(_e))

# ---- Conformal intervals (Table S7, Fig. S9) ----
try:
    cq = C.query("split >= 0").groupby(["target", "scheme"])[["q90", "coverage"]].agg(["mean", "std"])
    chk("q90 flex record-weighted", cq.loc[("flex", "record_weighted"), ("q90", "mean")], 0.54, 0.005)
    chk("q90 flex SD", cq.loc[("flex", "record_weighted"), ("q90", "std")], 0.07, 0.005)
    chk("q90 dts record-weighted", cq.loc[("dts", "record_weighted"), ("q90", "mean")], 0.49, 0.005)
    chk("q90 flex study-weighted", cq.loc[("flex", "study_weighted"), ("q90", "mean")], 0.51, 0.005)
    chk("q90 dts study-weighted", cq.loc[("dts", "study_weighted"), ("q90", "mean")], 0.52, 0.005)
    chk("coverage flex mean", cq.loc[("flex", "record_weighted"), ("coverage", "mean")], 0.89, 0.005)
    chk("coverage flex SD", cq.loc[("flex", "record_weighted"), ("coverage", "std")], 0.06, 0.005)
    cf = C.query("split >= 0 and target == 'flex' and scheme == 'record_weighted'")["coverage"]
    print("coverage flex 5th percentile / min: %.3f / %.3f" % (cf.quantile(0.05), cf.min()))
    pcov = PSc.query("scheme == 'record_weighted'").groupby(["target", "study"])["cov"].mean()
    chk("flex studies with coverage < 50%", (pcov.loc["flex"] < 0.5).sum(), 6, 0)
    chk("flex studies in coverage count", len(pcov.loc["flex"]), 89, 0)
    print("worst flex study coverage: %.3f" % pcov.loc["flex"].min())
except Exception as _e:
    REPORT.append('ERROR | section "Conformal intervals (Table S7, Fig. S9)" failed: ' + repr(_e))

# ---- Ranking (Fig. S3c) ----
try:
    spf, spt = SP.query("target == 'flex'")["rho"], SP.query("target == 'dts'")["rho"]
    chk("Spearman flex eligible studies", len(spf), 69, 0); chk("Spearman flex median", spf.median(), 0.66, 0.005)
    chk("Spearman flex share > 0.5 (%)", (spf > 0.5).mean() * 100, 64, 0.5); chk("Spearman flex share < 0 (%)", (spf < 0).mean() * 100, 10, 0.5)
    chk("Spearman dts eligible studies", len(spt), 16, 0); chk("Spearman dts median", spt.median(), 0.59, 0.005)
except Exception as _e:
    REPORT.append('ERROR | section "Ranking (Fig. S3c)" failed: ' + repr(_e))

# ---- Few-shot (Table 3, Fig. 4) ----
try:
    fsm = per_rep.groupby(["target", "k"])[["RMSE", "R2", "bias"]].mean()
    for kk, (rm, r2t, bt_) in {0: (7.23, 0.41, 1.106), 1: (5.87, 0.61, None), 5: (3.94, 0.82, 1.039)}.items():
        chk(f"few-shot flex RMSE k={kk}", fsm.loc[("flex", kk), "RMSE"], rm, 0.005); chk(f"few-shot flex R2 k={kk}", fsm.loc[("flex", kk), "R2"], r2t, 0.005)
        if bt_: chk(f"few-shot flex bias k={kk}", fsm.loc[("flex", kk), "bias"], bt_, 0.001)
    chk("few-shot dts RMSE k=0", fsm.loc[("dts", 0), "RMSE"], 3.61, 0.005); chk("few-shot dts RMSE k=5", fsm.loc[("dts", 5), "RMSE"], 2.43, 0.005)
    pss = PS4.groupby(["target", "study", "k"])["RMSE"].mean().unstack("k")
    chk("few-shot flex eligible studies", len(pss.loc["flex"]), 47, 0); chk("few-shot dts eligible studies", len(pss.loc["dts"]), 11, 0)
    chk("flex studies improved k0->k5", (pss.loc["flex"][5] < pss.loc["flex"][0]).sum(), 47, 0)
    chk("dts studies improved k0->k5", (pss.loc["dts"][5] < pss.loc["dts"][0]).sum(), 11, 0)
    chk("flex per-study RMSE P90, k=0", pss.loc["flex"][0].quantile(0.9), 11.6, 0.05); chk("flex per-study RMSE P90, k=5", pss.loc["flex"][5].quantile(0.9), 5.5, 0.05)
    chk("flex median per-study RMSE, k=0", pss.loc["flex"][0].median(), 5.04, 0.01)
    print("dts worst per-study RMSE after k=5: %.2f" % pss.loc["dts"][5].max())
    fc_only = FSc.query("set == 'fc only'").groupby(["k", "rep"]).apply(lambda x: np.sqrt(np.mean((x.y - x.p) ** 2))).groupby("k").mean()
    chk("dts fc-only few-shot RMSE k=0", fc_only[0], 4.12, 0.005); chk("dts fc-only few-shot RMSE k=5", fc_only[5], 3.93, 0.005)
except Exception as _e:
    REPORT.append('ERROR | section "Few-shot (Table 3, Fig. 4)" failed: ' + repr(_e))

# ---- External transfer (Table 4) ----
try:
    xr = X6.set_index("overlap rule")
    chk("external records, rule A", xr.iloc[0]["n"], 100, 0); chk("external records, rule B", xr.iloc[1]["n"], 75, 0)
    chk("external R2 rule A", xr.iloc[0]["R2"], -0.84, 0.005); chk("external RMSE rule A", xr.iloc[0]["RMSE"], 14.21, 0.01); chk("external ratio rule A", xr.iloc[0]["bias"], 0.75, 0.005)
    chk("external R2 rule B", xr.iloc[1]["R2"], -2.13, 0.005); chk("external RMSE rule B", xr.iloc[1]["RMSE"], 15.20, 0.01); chk("external ratio rule B", xr.iloc[1]["bias"], 0.64, 0.005)
    chk("external best R2 over rules", xr["R2"].max(), -0.83, 0.01)
    chk("external JSCE R2 (rule B)", TX.iloc[0]["R2"], -3.08, 0.005); chk("external ACI R2 (rule B)", TX.iloc[1]["R2"], -9.35, 0.005)
    chk("median flex/fc training", (fl["f_flex"] / fl["fc_28"]).median(), 0.175, 0.001)
    chk("median flex/fc external all", np.median(E["FS"] / E["CS"]), 0.248, 0.001); chk("median flex/fc external rule B", np.median(ye / fce), 0.305, 0.001)
except Exception as _e:
    REPORT.append('ERROR | section "External transfer (Table 4)" failed: ' + repr(_e))

# ---- Aspect-ratio encoding (supplement) ----
try:
    am2 = A.groupby(["target", "model"])["R2"].mean()
    for m, tgt in [("CatBoost", 0.382), ("XGBoost", 0.349), ("LightGBM", 0.323), ("RF", 0.342), ("MLP", 0.199)]:
        chk(f"encoding flex {m} R2", am2[("flex", m)], tgt, 0.001)
    chk("encoding dts MLP R2", am2[("dts", "MLP")], -0.61, 0.005)
except Exception as _e:
    REPORT.append('ERROR | section "Aspect-ratio encoding (supplement)" failed: ' + repr(_e))

# ---- Write report ----
npass = sum(r.startswith("PASS") for r in REPORT)
hdr = [f"VERIFY REPORT  |  {npass} PASS, {len(REPORT) - npass} CHECK  |  run time {(_time.time() - _T0) / 60:.1f} min", "-" * 120]
open(os.path.join(OUT, "VERIFY_REPORT.txt"), "w", encoding="utf-8").write("\n".join(hdr + REPORT) + "\n")
print("\n".join(hdr + REPORT)); print(f"\nReport written to {os.path.join(OUT, 'VERIFY_REPORT.txt')}")
