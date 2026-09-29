import os
try: os.chdir(os.path.dirname(os.path.abspath(__file__)))   # run from the folder that contains data/
except NameError: pass
# =============================================================================
# run_paper.py  (part 1 of 2: all analyses, Steps 0-11)
# Complete reproduction script for:
#   Tabani, A., and R. Biswas. "Transferability of Machine-Learning Models for UHPC
#   Strength Prediction: Cross-Study Validation, Diagnosis, and Calibration." (JMCE submission)
#
# Steps 0-11 are the published notebook (github.com/tabani1592/uhpc-ml-transferability),
# unchanged so that results match the paper. After it finishes, run verify_paper.py (part 2), which adds Fig. 1 and writes
# output/VERIFY_REPORT.txt comparing every number with the manuscript.
#
# Run:  python run_paper.py   then   python verify_paper.py   (uhpc-transfer env)
# Needs data/uhpc_v2_clean.xlsx, data/uhpc_compressive_strength_1.csv,
#       data/uhpc_flexural_strength_1.csv
# Time: about 60-75 min on a 12-thread CPU (Step 10 is the longest part).
# =============================================================================
import sys, platform, importlib, time as _time
_T0 = _time.time()
print("Python", sys.version.split()[0], "|", platform.platform())
for _m in ["pandas","numpy","scipy","sklearn","catboost","xgboost","lightgbm","shap","joblib","matplotlib"]:
    try: print(f"  {_m:11s}", importlib.import_module(_m).__version__)
    except Exception as _e: print(f"  {_m:11s} MISSING ->", _e)
import os
for _f in ["uhpc_v2_clean.xlsx","uhpc_compressive_strength_1.csv","uhpc_flexural_strength_1.csv"]:
    print("  data/"+_f, "found" if os.path.exists(os.path.join("data",_f)) else "NOT FOUND")
import matplotlib; matplotlib.use("Agg")

# ---- progress helpers (added): same job order as before, so results are unchanged ----
from joblib import Parallel as _Parallel
from tqdm.auto import tqdm as _tqdm
_BAR = "{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [spent {elapsed} | left {remaining}]"
def PPAR(tasks, desc):
    tasks = list(tasks)
    return list(_tqdm(_Parallel(n_jobs=-1, return_as="generator")(tasks), total=len(tasks), desc=desc, bar_format=_BAR))
def STEP(n, name):
    print(f"\n===== STEP {n}/11: {name}  |  total time so far {(_time.time() - _T0) / 60:.1f} min =====", flush=True)

# # Transferability of ML models for UHPC strength prediction — reproduction notebook
# 
# Reproduces every table and figure of Tabani & Biswas, *Transferability of Machine-Learning Models for UHPC Strength Prediction: Cross-Study Validation, Diagnosis and Calibration*.
# 
# **Inputs (folder `data/`)**
# - `uhpc_v2_clean.xlsx` — cleaned modelling table derived from Malik et al. (2025), Mendeley Data V2, doi:10.17632/czb7ww5pkz.2
# - `uhpc_compressive_strength_1.csv`, `uhpc_flexural_strength_1.csv` — external database of Bolbolvand et al. (2025), obtained from its authors
# 
# All outputs are written to `output/`. Run cells in order; total run time is about 60–75 min on a 12-thread CPU.

# ===== STEP 0: PATHS =====
import os, warnings; warnings.filterwarnings("ignore"); os.environ["PYTHONWARNINGS"] = "ignore"
P = "data"; OUT = "output"; os.makedirs(OUT, exist_ok=True)

# ## Step 1 — modelling sets and fold assignments (Section 3.2–3.5)
STEP(1, "modelling sets and folds")

import pandas as pd, numpy as np, pickle
from sklearn.model_selection import KFold
SEED0, NREP = 2026, 10
df = pd.read_excel(os.path.join(P, "uhpc_v2_clean.xlsx"))
df["source_id"] = df["source_id"].ffill()
ft = df["fibre_type"].astype(str).str.lower()
df["is_steel"] = ft.str.contains("steel").astype(int)
df["is_pe"]    = ft.str.contains("pe|polyeth").astype(int) & ~ft.str.contains("pp|polyprop").astype(int)
df["is_pva"]   = ft.str.contains("pva|vinyl").astype(int)
cu = df["curing"].astype(str).str.lower()
df["cure_steam"] = cu.str.contains("steam|heat").astype(int)
df["cure_auto"]  = cu.str.contains("auto").astype(int)
X_COMMON = ["fc_28","w_b","sf_b","sand_b","sp_b","vf_pct","l_d","RI","curing_temp","cure_steam","cure_auto","is_steel","is_pe","is_pva"]
FEAT = {"flex": X_COMMON + ["prism_depth","prism_length"], "dts": X_COMMON}
TARGET = {"flex": "f_flex", "dts": "f_dts"}
sets = {}
for k in ["flex","dts"]:
    d = df[df["fc_28"].notna() & df[TARGET[k]].notna()].reset_index(drop=True); sets[k] = d
    print(k, "records:", len(d), "| studies:", d["source_id"].nunique())
def grouped_folds(groups, seed, n=5):
    u = np.array(sorted(groups.unique(), key=str)); rng = np.random.RandomState(seed); rng.shuffle(u)
    fold_of = {g: i % n for i, g in enumerate(u)}; f = groups.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(n)]
folds = {}
for k, d in sets.items():
    folds[k] = {"random": [], "grouped": []}
    for r in range(NREP):
        folds[k]["random"].append(list(KFold(5, shuffle=True, random_state=SEED0 + r).split(d)))
        folds[k]["grouped"].append(grouped_folds(d["source_id"], SEED0 + r))
pickle.dump({"sets": sets, "FEAT": FEAT, "TARGET": TARGET, "folds": folds}, open(os.path.join(OUT, "setup.pkl"), "wb"))

# ## Step 2 — random vs grouped CV, five families + study-mean diagnostic (Table 3, Section 4.2)
STEP(2, "random vs grouped CV")

import time
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def make(name, seed):
    if name == "CatBoost": return CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=seed, verbose=0, thread_count=1)
    if name == "XGBoost":  return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=seed, n_jobs=1)
    if name == "LightGBM": return LGBMRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, random_state=seed, verbose=-1, n_jobs=1)
    if name == "RF":  return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=1))
    if name == "MLP": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=2000, random_state=seed))

def run_job(k, proto, r, m):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k]; X = d[FEAT[k]].values.astype(float); y = d[TARGET[k]].values; g = d["source_id"].values
    oof = np.full(len(y), np.nan); hits = 0
    for tr, te in folds[k][proto][r]:
        if m == "StudyMean":
            tm = pd.Series(y[tr]).groupby(g[tr]).mean()
            oof[te] = pd.Series(g[te]).map(tm).fillna(y[tr].mean()).values; hits += np.isin(g[te], g[tr]).sum()
        else:
            mdl = make(m, 2026 + r); mdl.fit(X[tr], y[tr]); oof[te] = mdl.predict(X[te])
    return k, proto, r, m, oof, hits / len(y)

MODELS = ["CatBoost", "XGBoost", "LightGBM", "RF", "MLP", "StudyMean"]
out = PPAR((delayed(run_job)(k, p, r, m) for k in sets for p in ["random", "grouped"] for r in range(10) for m in MODELS), "Step 2 random vs grouped CV")
rows, oof_store, exposure = [], {}, {}
for k, proto, r, m, oof, exp in out:
    y = sets[k][TARGET[k]].values
    rows.append([k, proto, m, r, r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), np.mean(oof / y)])
    if proto == "grouped": oof_store[(k, m, r)] = oof
    if m == "StudyMean" and r == 0: exposure[(k, proto)] = exp
R = pd.DataFrame(rows, columns=["target", "protocol", "model", "rep", "R2", "RMSE", "bias"])
def summ(s):
    m, sd, n = s.mean(), s.std(ddof=1), len(s); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return f"{m:.3f} ± {sd:.3f} [{m-h:.3f}, {m+h:.3f}]"
T3 = R.groupby(["target", "model", "protocol"])[["R2", "RMSE", "bias"]].agg(summ).unstack("protocol")
R.to_csv(os.path.join(OUT, "step2_all_reps.csv"), index=False); T3.to_excel(os.path.join(OUT, "step2_table3.xlsx"))
pickle.dump(oof_store, open(os.path.join(OUT, "oof_grouped.pkl"), "wb"))
print(T3["R2"]); print(exposure)

# ## Step 3 — study-level split conformal (Table 6, Fig. 11)
STEP(3, "split conformal")

def wq(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]; c = np.cumsum(w) / w.sum(); return x[np.searchsorted(c, q)]
res, per_study = [], []
for k in ["flex", "dts"]:
    d = sets[k]; y = d[TARGET[k]].values; g = d["source_id"].astype(str).values
    studies = np.unique(g); nstud = pd.Series(g).map(pd.Series(g).value_counts()).values
    for r in range(10):
        p = oof_store[(k, "CatBoost", r)]; s = np.abs(y - p) / p
        q_in = np.quantile(s, 0.90); res.append([k, r, -1, "v4_insample", q_in, np.mean(s <= q_in)])
        for j in range(20):
            rng = np.random.RandomState(1000 * r + j); rng.shuffle(studies)
            cal = np.isin(g, studies[: len(studies) // 2]); te = ~cal; n_cal = cal.sum()
            for scheme, q in [("record_weighted", np.quantile(s[cal], min(1, np.ceil((n_cal + 1) * 0.9) / n_cal))),
                              ("study_weighted",  wq(s[cal], 1 / nstud[cal], 0.90))]:
                cov = s[te] <= q; res.append([k, r, j, scheme, q, cov.mean()])
                per_study += [[k, r, j, scheme, st, c] for st, c in pd.Series(cov).groupby(g[te]).mean().items()]
C = pd.DataFrame(res, columns=["target", "rep", "split", "scheme", "q90", "coverage"])
PSc = pd.DataFrame(per_study, columns=["target", "rep", "split", "scheme", "study", "cov"])
C.to_csv(os.path.join(OUT, "step3_conformal_all.csv"), index=False); PSc.to_csv(os.path.join(OUT, "step3_conformal_per_study.csv"), index=False)
print(C.groupby(["target", "scheme"])[["q90", "coverage"]].agg(["mean", "std"]).round(3))

# ## Step 4 — few-shot calibration, 10 orderings (Table 7, Fig. 12)
STEP(4, "few-shot calibration")

KS = [0, 1, 3, 5]
def fs_job(k, r, study, cols=None):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k]; cols = cols or FEAT[k]; X = d[cols].values.astype(float); y = d[TARGET[k]].values; g = d["source_id"].astype(str).values
    idx = np.random.RandomState(7000 + r).permutation(np.where(g == study)[0]); ev = idx[5:]
    te_fold = [te for tr, te in folds[k]["grouped"][r] if np.isin(idx[0], te)][0]
    base_tr = np.setdiff1d(np.arange(len(y)), te_fold); out = []
    for kk in KS:
        tr = np.concatenate([base_tr, idx[:kk]])
        m = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026 + r, verbose=0, thread_count=1).fit(X[tr], y[tr])
        out.append((k, r, study, kk, ev, m.predict(X[ev])))
    return out
jobs = []
for k in ["flex", "dts"]:
    vc = sets[k]["source_id"].astype(str).value_counts(); jobs += [(k, r, s) for r in range(10) for s in vc[vc >= 8].index]
res4 = PPAR((delayed(fs_job)(*j) for j in jobs), "Step 4 few-shot")
rows, stud = [], []
for block in res4:
    for k, r, s, kk, ev, p in block:
        yt = sets[k][TARGET[k]].values[ev]
        rows.append(pd.DataFrame({"target": k, "rep": r, "study": s, "k": kk, "y": yt, "p": p}))
        stud.append([k, r, s, kk, np.sqrt(np.mean((yt - p) ** 2)), np.mean(p / yt)])
F = pd.concat(rows); PS4 = pd.DataFrame(stud, columns=["target", "rep", "study", "k", "RMSE", "bias"])
def pooled(x):
    e = x["y"] - x["p"]
    return pd.Series({"RMSE": np.sqrt(np.mean(e**2)), "R2": 1 - np.sum(e**2) / np.sum((x["y"] - x["y"].mean())**2), "bias": np.mean(x["p"] / x["y"])})
per_rep = F.groupby(["target", "k", "rep"]).apply(pooled).reset_index()
per_rep.to_csv(os.path.join(OUT, "step4_fewshot_per_rep.csv"), index=False); PS4.to_csv(os.path.join(OUT, "step4_fewshot_per_study.csv"), index=False)
print(per_rep.groupby(["target", "k"])[["RMSE", "R2", "bias"]].agg(["mean", "std"]).round(3))

# ## Step 5 — transparent baselines B1–B4 (Table 3)
STEP(5, "transparent baselines")

from sklearn.linear_model import LinearRegression
BASE = {"flex": {"B1 fc": ["fc_28"], "B2 fc+Vf": ["fc_28", "vf_pct"], "B3 fc+RI+depth": ["fc_28", "RI", "prism_depth"], "B4 power law": "pow"},
        "dts":  {"B1 fc": ["fc_28"], "B2 fc+Vf": ["fc_28", "vf_pct"], "B3 fc+RI+PE": ["fc_28", "RI", "is_pe"], "B4 power law": "pow"}}
rows = []
for k, d in sets.items():
    y = d[TARGET[k]].values
    for name, cols in BASE[k].items():
        for proto in ["random", "grouped"]:
            for r, fl in enumerate(folds[k][proto]):
                oof = np.full(len(y), np.nan)
                for tr, te in fl:
                    if cols == "pow":
                        Xp = np.column_stack([np.log(d["fc_28"]), d["RI"]]); m = LinearRegression().fit(Xp[tr], np.log(y[tr])); oof[te] = np.exp(m.predict(Xp[te]))
                    else:
                        X = d[cols].values.astype(float); oof[te] = make_pipeline(SimpleImputer(strategy="median"), LinearRegression()).fit(X[tr], y[tr]).predict(X[te])
                rows.append([k, name, proto, r, r2_score(y, oof), np.sqrt(mean_squared_error(y, oof)), np.mean(oof / y)])
B = pd.DataFrame(rows, columns=["target", "model", "protocol", "rep", "R2", "RMSE", "bias"])
B.to_csv(os.path.join(OUT, "step5_baselines_all.csv"), index=False)
print(B.groupby(["target", "model", "protocol"])["R2"].agg(summ).unstack("protocol"))

# ## Step 6 — sensitivity: (a) aspect-ratio encoding (Section 4.10); (b) external transfer and overlap rules (Table 8)
STEP(6, "encoding and external transfer")

from sklearn.neighbors import NearestNeighbors
def run_c(k, r, m):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k].copy(); d["has_fibre"] = (d["vf_pct"] > 0).astype(int); d.loc[d["has_fibre"] == 0, "l_d"] = 0
    X = d[FEAT[k] + ["has_fibre"]].values.astype(float); y = d[TARGET[k]].values; oof = np.full(len(y), np.nan)
    for tr, te in folds[k]["grouped"][r]: oof[te] = make(m, 2026 + r).fit(X[tr], y[tr]).predict(X[te])
    return k, m, r, r2_score(y, oof)
A = pd.DataFrame(PPAR((delayed(run_c)(k, r, m) for k in sets for r in range(10) for m in MODELS[:5]), "Step 6 aspect-ratio encoding"), columns=["target", "model", "rep", "R2"])
A.groupby(["target", "model"])["R2"].agg(["mean", "std"]).to_excel(os.path.join(OUT, "step6a_aspect_ratio.xlsx"))

V = ["C","SF","QP","FA","SL","MK","S","QS","W","SP","Age","L","D","BV","PPV","GV","SSV"]
cs = pd.read_csv(os.path.join(P, "uhpc_compressive_strength_1.csv"))[V + ["CS"]].dropna(); fs = pd.read_csv(os.path.join(P, "uhpc_flexural_strength_1.csv"))[V + ["FS"]].dropna()
cs[V] = cs[V].round(2); fs[V] = fs[V].round(2)
E = fs.merge(cs.groupby(V, as_index=False)["CS"].mean(), on=V).query("Age == 28").groupby(V, as_index=False)[["FS", "CS"]].mean()
b = E[["C","SF","FA","SL","MK"]].sum(axis=1)
ext = pd.DataFrame({"fc_28": E["CS"], "w_b": E["W"]/b, "sf_b": E["SF"]/b, "sand_b": (E["S"]+E["QS"])/b, "sp_b": E["SP"]/b, "vf_pct": E[["BV","PPV","GV","SSV"]].sum(axis=1)})
ext["l_d"] = np.where(ext["vf_pct"] > 0, E["L"] / E["D"], np.nan); ext["RI"] = ext["vf_pct"] * ext["l_d"].fillna(0) / 100
ext["curing_temp"] = np.nan; ext["cure_steam"] = 0; ext["cure_auto"] = 0
ext["is_steel"] = (E["SSV"] > 0).astype(int); ext["is_pe"] = 0; ext["is_pva"] = 0; ext["prism_depth"] = 100; ext["prism_length"] = 400
y_ext = E["FS"].values; tr = sets["flex"]
p_ext = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026, verbose=0).fit(tr[FEAT["flex"]].values.astype(float), tr["f_flex"].values).predict(ext[FEAT["flex"]].values.astype(float))
T = tr[["cement","silica_fume","water","fc_28","f_flex"]].values
ruleB = np.array([np.any((np.abs(T[:,0]-c) <= 1) & (np.abs(T[:,1]-s) <= 1) & (np.abs(T[:,2]-w) <= 1) & ((np.abs(T[:,3]-fc) <= 1) | (np.abs(T[:,4]-ff) <= 1)))
                  for c, s, w, fc, ff in zip(E["C"], E["SF"], E["W"], E["CS"], y_ext)])
nc = ["w_b","sf_b","sand_b","sp_b","vf_pct","fc_28"]; mu, sd = tr[nc].mean(), tr[nc].std()
Zt = ((tr[nc] - mu) / sd).fillna(0).values; Ze = ((ext[nc] - mu) / sd).fillna(0).values
nn = NearestNeighbors(n_neighbors=2).fit(Zt); thr = np.quantile(nn.kneighbors(Zt)[0][:, 1], 0.05); ruleC = nn.kneighbors(Ze, n_neighbors=1)[0][:, 0] < thr
rows = []
for name, keep in [("A: no removal", np.ones(len(E), bool)), ("B: v4 rule", ~ruleB), ("C: NN distance < P5", ~ruleC), ("B or C", ~(ruleB | ruleC))]:
    rows.append([name, keep.sum(), r2_score(y_ext[keep], p_ext[keep]), np.sqrt(mean_squared_error(y_ext[keep], p_ext[keep])), np.mean(p_ext[keep] / y_ext[keep])])
X6 = pd.DataFrame(rows, columns=["overlap rule", "n", "R2", "RMSE", "bias"]); X6.to_excel(os.path.join(OUT, "step6b_external_overlap.xlsx"), index=False); print(X6.round(3))

# ## Step 7 — variance share, Spearman ranking and SHAP (Fig. 5, Table 5, Figs 8–10)
STEP(7, "variance share, ranking, SHAP")

import shap
from scipy.stats import spearmanr
def share(d, v):
    x = d[[v, "source_id"]].dropna()
    if x[v].var() == 0: return np.nan
    m = x.groupby("source_id")[v].transform("mean"); return ((m - x[v].mean())**2).sum() / ((x[v] - x[v].mean())**2).sum()
VV = ["fc_28","w_b","sf_b","sand_b","sp_b","vf_pct","l_d","RI","curing_temp","prism_depth","prism_length"]
VS = pd.DataFrame({k: {v: share(sets[k], v) for v in VV + [TARGET[k]]} for k in sets}); VS.to_excel(os.path.join(OUT, "step8_variance_share.xlsx"))
rows = []
for k in sets:
    d = sets[k]; y = d[TARGET[k]].values; g = d["source_id"].astype(str).values
    for r in range(10):
        p = oof_store[(k, "CatBoost", r)]
        for s in pd.unique(g):
            i = g == s
            if i.sum() >= 5 and np.ptp(y[i]) > 0: rows.append([k, r, s, spearmanr(y[i], p[i]).correlation])
SP = pd.DataFrame(rows, columns=["target", "rep", "study", "rho"]).groupby(["target", "study"])["rho"].mean().reset_index()
SP.to_csv(os.path.join(OUT, "step8_spearman_per_study.csv"), index=False); print(SP.groupby("target")["rho"].median())
shap_tab, SV = {}, {}
for k in sets:
    d = sets[k]; X = d[FEAT[k]].astype(float)
    m = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026, verbose=0).fit(X.values, d[TARGET[k]].values)
    SV[k] = shap.TreeExplainer(m).shap_values(X.values); shap_tab[k] = pd.Series(np.abs(SV[k]).mean(0), index=FEAT[k])
pd.DataFrame(shap_tab).round(2).to_excel(os.path.join(OUT, "step8_table5_shap.xlsx")); print(pd.DataFrame(shap_tab).round(2))

# ## Step 8 — learning curve, nested forward selection, ablation, compact few-shot (Tables 4, Section 4.5, Figs 6–7)
STEP(8, "learning curve, selection, ablation")

def gcv(k, cols, r):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k]; X = d[cols].values.astype(float); y = d[TARGET[k]].values; p = np.full(len(y), np.nan)
    for tr, te in folds[k]["grouped"][r]:
        if np.all(np.nanstd(X[tr], axis=0) == 0): p[te] = y[tr].mean(); continue
        p[te] = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026 + r, verbose=0, thread_count=1).fit(X[tr], y[tr]).predict(X[te])
    return r2_score(y, p), np.mean(p / y)
def mean_score(k, r):
    d = sets[k]; y = d[TARGET[k]].values; p = np.full(len(y), np.nan)
    for tr, te in folds[k]["grouped"][r]: p[te] = y[tr].mean()
    return r2_score(y, p), np.mean(p / y)
PAR = lambda gen, desc="Step 8": PPAR(gen, desc)
sel = {}
for k in sets:                                   # select on repeat 0, starting from the constant predictor
    usable = [c for c in FEAT[k] if sets[k][c].nunique(dropna=True) > 1]; chosen, best = [], mean_score(k, 0)[0]
    while True:
        cand = [c for c in usable if c not in chosen]
        if not cand: break
        sc = [s[0] for s in PAR(delayed(gcv)(k, chosen + [c], 0) for c in cand)]; j = int(np.argmax(sc))
        if sc[j] - best <= 0.005: break
        chosen.append(cand[j]); best = sc[j]
    sel[k] = chosen; print(k, "selected:", chosen or "none")
rows = []                                        # report on repeats 1-9
for k, sets_k in {"flex": {"complete": FEAT["flex"], "selected": sel["flex"]},
                  "dts": {"complete": FEAT["dts"], "selected": sel["dts"], "fc only": ["fc_28"], "fc + Vf": ["fc_28", "vf_pct"]}}.items():
    for n, c in sets_k.items():
        for r in range(1, 10): rows.append([k, n, r, *(gcv(k, c, r) if c else mean_score(k, r))])
SEL = pd.DataFrame(rows, columns=["target", "set", "rep", "R2", "bias"]); print(SEL.groupby(["target", "set"])[["R2", "bias"]].agg(["mean", "std"]).round(3))
abl = [(k, "complete", FEAT[k], r) for k in sets for r in range(10)] + [(k, "minus sp_b", [c for c in FEAT[k] if c != "sp_b"], r) for k in sets for r in range(10)] + \
      [("dts", f"minus {v}", [c for c in FEAT["dts"] if c != v], r) for v in FEAT["dts"] if v != "sp_b" for r in range(10)]
AB = pd.DataFrame([[k, n, r, o[0]] for (k, n, c, r), o in zip(abl, PAR(delayed(gcv)(k, c, r) for k, n, c, r in abl))], columns=["target", "set", "rep", "R2"])
ABm = AB.groupby(["target", "set"])["R2"].agg(["mean", "std"]); print(ABm.round(3))
d = sets["flex"]; g = d["source_id"].astype(str).values; X = d[FEAT["flex"]].values.astype(float); y = d["f_flex"].values
st = np.array(sorted(pd.unique(g))); np.random.RandomState(99).shuffle(st); test_st, pool = st[:20], st[20:]; te = np.isin(g, test_st)
def lc(n, r):
    import warnings; warnings.filterwarnings("ignore")
    trm = np.isin(g, np.random.RandomState(500 + 10 * n + r).choice(pool, n, replace=False))
    p = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026 + r, verbose=0, thread_count=1).fit(X[trm], y[trm]).predict(X[te])
    return n, r, trm.sum(), r2_score(y[te], p), np.sqrt(mean_squared_error(y[te], p))
LC = pd.DataFrame(PAR(delayed(lc)(n, r) for n in [10, 20, 40, len(pool)] for r in range(10)), columns=["studies", "rep", "records", "R2", "RMSE"])
print(LC.groupby("studies")[["records", "R2", "RMSE"]].agg(["mean", "std"]).round(3))
vc = sets["dts"]["source_id"].astype(str).value_counts(); elig = vc[vc >= 8].index
FSc = []
for nm, cols in [("complete", FEAT["dts"]), ("fc only", ["fc_28"])]:
    for blk in PAR(delayed(fs_job)("dts", r, s, cols) for r in range(10) for s in elig):
        for k_, r, s, kk, ev, pp in blk: FSc.append(pd.DataFrame({"set": nm, "k": kk, "rep": r, "y": sets["dts"]["f_dts"].values[ev], "p": pp}))
FSc = pd.concat(FSc)
print(FSc.groupby(["set", "k", "rep"]).apply(lambda x: np.sqrt(np.mean((x.y - x.p)**2))).groupby(["set", "k"]).agg(["mean", "std"]).round(3))
with pd.ExcelWriter(os.path.join(OUT, "step9_results.xlsx")) as w:
    SEL.to_excel(w, sheet_name="selection"); ABm.to_excel(w, sheet_name="ablation"); LC.to_excel(w, sheet_name="learning_curve")

# ## Step 9 — figures
STEP(9, "figures")

import matplotlib.pyplot as plt
BLUE, ORANGE, INK, MUTED, GREY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#b5b4ae"
COL = {"flex": BLUE, "dts": ORANGE}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
def save(f, n): f.savefig(os.path.join(OUT, n + ".png"), dpi=600, bbox_inches="tight"); f.savefig(os.path.join(OUT, n + ".pdf"), bbox_inches="tight")
# Fig 4
order = ["CatBoost", "XGBoost", "LightGBM", "RF", "MLP", "StudyMean"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for a, k in zip(ax, ["flex", "dts"]):
    gm = R.query("target == @k").groupby(["model", "protocol"])["R2"].agg(["mean", "std"]); x = np.arange(6)
    a.bar(x - 0.2, [gm.loc[(m, "random"), "mean"] for m in order], 0.38, yerr=[gm.loc[(m, "random"), "std"] for m in order], color=GREY, capsize=2, label="Random")
    a.bar(x + 0.2, [gm.loc[(m, "grouped"), "mean"] for m in order], 0.38, yerr=[gm.loc[(m, "grouped"), "std"] for m in order], color=COL[k], capsize=2, label="Study-grouped")
    a.axhline(0, color=MUTED, lw=0.8); a.set_xticks(x); a.set_xticklabels(order, fontsize=8); a.set_ylabel("R²"); a.legend(frameon=False)
fig.tight_layout(); save(fig, "Fig4_families_random_vs_grouped")
# Fig 11
cov = PSc.query("scheme == 'record_weighted'").groupby(["target", "study"])["cov"].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
for a, k in zip(ax, ["flex", "dts"]):
    a.hist(cov.query("target == @k")["cov"], bins=np.linspace(0, 1, 11), color=COL[k], edgecolor="white"); a.axvline(0.9, color=INK, ls="--")
    a.set_xlabel("Coverage within study"); a.set_ylabel("Number of studies")
fig.tight_layout(); save(fig, "Fig11_conformal_per_study")
# Fig 12
PSm = PS4.groupby(["target", "study", "k"], as_index=False)["RMSE"].mean()
fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
for a, k in zip(ax, ["flex", "dts"]):
    data = [PSm.query("target == @k and k == @kk")["RMSE"].values for kk in KS]
    bp = a.boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
    for bx in bp["boxes"]: bx.set(facecolor=COL[k], alpha=0.3)
    for i, dd in enumerate(data): a.scatter(i + 1 + np.random.RandomState(1).uniform(-0.12, 0.12, len(dd)), dd, s=12, color=COL[k], zorder=3)
    a.set_xticklabels(KS); a.set_xlabel("Records transferred, k"); a.set_ylabel("Per-study RMSE (MPa)")
fig.tight_layout(); save(fig, "Fig12_fewshot_per_study")
# Fig 13
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2)); keep = ~ruleB
ax[0].scatter(y_ext[keep], p_ext[keep], s=14, color=BLUE); lim = [0, max(y_ext[keep].max(), p_ext[keep].max()) * 1.05]; ax[0].plot(lim, lim, color=INK, lw=0.8)
ax[0].set_xlabel("Measured (MPa)"); ax[0].set_ylabel("Predicted (MPa)")
lcm = LC.groupby("studies")["R2"].agg(["mean", "std"]); ax[1].errorbar(lcm.index, lcm["mean"], yerr=lcm["std"], color=BLUE, marker="o", capsize=3, ls="none")
ax[1].set_xlabel("Training studies"); ax[1].set_ylabel("R² on fixed test set")
spf = SP.query("target == 'flex'")["rho"]; ax[2].hist(spf, bins=np.linspace(-1, 1, 21), color=BLUE, edgecolor="white"); ax[2].axvline(spf.median(), color=INK, ls="--")
ax[2].set_xlabel("Within-study Spearman ρ"); ax[2].set_ylabel("Number of studies")
fig.tight_layout(); save(fig, "Fig13_external_learning_ranking")
# SHAP figures
for k, n in [("flex", "Fig8_shap_summary_flex"), ("dts", "Fig10_shap_summary_dts")]:
    plt.figure(); shap.summary_plot(SV[k], sets[k][FEAT[k]].astype(float), show=False, max_display=12); save(plt.gcf(), n); plt.close()

# Fig 3: CatBoost predicted vs measured, random vs grouped (repeat 0)
fig, ax = plt.subplots(2, 2, figsize=(7, 6.6))
for i, k in enumerate(["flex", "dts"]):
    d = sets[k]; X = d[FEAT[k]].values.astype(float); y = d[TARGET[k]].values; pr = np.full(len(y), np.nan)
    for tr, te in folds[k]["random"][0]: pr[te] = CatBoostRegressor(iterations=400, learning_rate=0.05, depth=5, random_seed=2026, verbose=0).fit(X[tr], y[tr]).predict(X[te])
    for j, p in enumerate([pr, oof_store[(k, "CatBoost", 0)]]):
        a = ax[i, j]; lim = [0, max(y.max(), p.max()) * 1.05]; a.scatter(y, p, s=10, color=COL[k], alpha=0.55)
        a.plot(lim, lim, color=INK, lw=0.8); a.plot(lim, [v * 1.2 for v in lim], ls="--", color=MUTED, lw=0.6); a.plot(lim, [v * 0.8 for v in lim], ls="--", color=MUTED, lw=0.6)
        a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("Measured (MPa)"); a.set_ylabel("Predicted (MPa)")
fig.tight_layout(); save(fig, "Fig3_catboost_random_vs_grouped")
# Fig 5: between-study variance share (prism dimensions flexural only)
VS2 = VS.copy(); VS2.loc["response"] = [VS2.loc["f_flex", "flex"], VS2.loc["f_dts", "dts"]]; VS2 = VS2.drop(["f_flex", "f_dts"]); VS2.loc[["prism_depth", "prism_length"], "dts"] = np.nan
idx = VS2.sort_values("flex").index; y0 = np.arange(len(idx)); fig, a = plt.subplots(figsize=(6.4, 4.4))
for off, k in [(-0.2, "flex"), (0.2, "dts")]: a.barh(y0 + off, VS2.loc[idx, k], 0.38, color=COL[k], label=k)
a.set_yticks(y0); a.set_yticklabels(idx); a.set_xlim(0, 1.02); a.axvline(0.5, ls="--", color=MUTED, lw=0.6); a.set_xlabel("Share of variance between studies"); a.legend(frameon=False)
fig.tight_layout(); save(fig, "Fig5_variance_share")
# Fig 6-7: tensile selection and ablation
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
t = SEL.query("target == 'dts'").groupby("set")["R2"].agg(["mean", "std"]).reindex(["complete", "selected", "fc only", "fc + Vf"])
ax[0].bar(range(4), t["mean"], yerr=t["std"], color=[ORANGE, ORANGE, GREY, GREY], capsize=3); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Complete", "Constant", "f′c", "f′c + Vf"]); ax[0].axhline(0, color=MUTED)
T_fs = FSc.groupby(["set", "k", "rep"]).apply(lambda x: np.sqrt(np.mean((x.y - x.p)**2))).groupby(["set", "k"]).agg(["mean", "std"])
for nm, ls in [("complete", "-"), ("fc only", "--")]: ax[1].errorbar(KS, T_fs.loc[nm, "mean"], yerr=T_fs.loc[nm, "std"], color=ORANGE, ls=ls, marker="o", capsize=3, label=nm)
ax[1].set_xlabel("Records transferred, k"); ax[1].set_ylabel("RMSE (MPa)"); ax[1].legend(frameon=False); fig.tight_layout(); save(fig, "Fig6_tensile_selection")
loo = (ABm.loc["dts", "mean"] - ABm.loc[("dts", "complete"), "mean"]).drop("complete").sort_values()
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3)); a2 = ABm.loc["dts"].reindex(["complete", "minus sp_b"])
ax[0].bar([0, 1], a2["mean"], yerr=a2["std"], color=ORANGE, capsize=3); ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Complete", "Without SP/binder"]); ax[0].axhline(0, color=MUTED)
ax[1].barh(range(len(loo)), loo.values, color=[ORANGE if v > 0 else GREY for v in loo.values]); ax[1].set_yticks(range(len(loo))); ax[1].set_yticklabels([s.replace("minus ", "") for s in loo.index], fontsize=8)
ax[1].axvline(0, color=MUTED); ax[1].set_xlabel("Change in grouped R² when removed"); fig.tight_layout(); save(fig, "Fig7_tensile_ablation")
# Fig 9: SHAP dependence (flexural)
Xf = sets["flex"][FEAT["flex"]].astype(float); fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
for a, v in zip(ax, ["fc_28", "RI", "prism_depth"]):
    a.scatter(Xf[v], SV["flex"][:, FEAT["flex"].index(v)], s=8, color=BLUE, alpha=0.5); a.axhline(0, color=MUTED, lw=0.6); a.set_xlabel(v); a.set_ylabel("SHAP value (MPa)")
fig.tight_layout(); save(fig, "Fig9_shap_dependence_flex")
# Fig 14: validation hierarchy (CatBoost)
from sklearn.metrics import r2_score as _r2
keep = ~ruleB; yy, pp = y_ext[keep], p_ext[keep]; rng = np.random.RandomState(0); bs = []
for _ in range(1000):
    i = rng.randint(0, len(yy), len(yy)); bs.append([_r2(yy[i], pp[i]), np.sqrt(np.mean((yy[i] - pp[i])**2)), np.mean(pp[i] / yy[i])])
bs = np.array(bs); s2 = R.query("model == 'CatBoost'"); s4 = per_rep.query("k == 5")
def row(t):
    sel_ = {p: s2[(s2["target"] == t) & (s2["protocol"] == p)] for p in ["random", "grouped"]}; s4t = s4[s4["target"] == t]
    out = [[(sel_[p][c].mean(), sel_[p][c].std()) for c in ["R2", "RMSE", "bias"]] for p in ["random", "grouped"]]
    out.append([(s4t[c].mean(), s4t[c].std()) for c in ["R2", "RMSE", "bias"]])
    out.append([(bs[:, j].mean(), bs[:, j].std()) for j in range(3)] if t == "flex" else [(np.nan, np.nan)] * 3)
    return np.array(out)
res14 = {"flex": row("flex"), "dts": row("dts")}
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
for j, lab in enumerate(["R²", "RMSE (MPa)", "Mean predicted / measured"]):
    for t, off in [("flex", -0.08), ("dts", 0.08)]:
        ax[j].errorbar(np.arange(4) + off, res14[t][:, j, 0], yerr=res14[t][:, j, 1], color=COL[t], ls="none", marker="o", ms=8, capsize=3, label=t)
    ax[j].axvline(2.5, color="#c9c8c2", lw=0.8); ax[j].set_xticks(range(4)); ax[j].set_xticklabels(["Random", "Grouped", "Few-shot k=5", "External"], fontsize=8); ax[j].set_ylabel(lab)
ax[0].axhline(0, ls="--", color=MUTED, lw=0.8); ax[2].axhline(1, ls="--", color=MUTED, lw=0.8); ax[0].legend(frameon=False)
fig.tight_layout(); save(fig, "Fig14_validation_hierarchy")

# ## Step 10 — nested hyperparameter tuning sensitivity (Table 9, Section 4.10)
STEP(10, "nested tuning (longest step)")

import itertools
from tqdm.auto import tqdm
GRID = [dict(depth=d, learning_rate=lr, iterations=it, l2_leaf_reg=l2) for d, lr, it, l2 in itertools.product([4, 6, 8], [0.03, 0.1], [300, 800], [1, 5])]
NREP_T = 3
def inner_splits(idx, groups, proto, seed):
    if proto == "random":
        return [(idx[a], idx[b]) for a, b in KFold(3, shuffle=True, random_state=seed).split(idx)]
    u = np.array(sorted(pd.unique(groups[idx]), key=str)); rng = np.random.RandomState(seed); rng.shuffle(u)
    f = pd.Series(groups[idx]).map({g: i % 3 for i, g in enumerate(u)}).values
    return [(idx[f != i], idx[f == i]) for i in range(3)]
def score_cfg(k, proto, r, fo, ci):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k]; X = d[FEAT[k]].values.astype(float); y = d[TARGET[k]].values; g = d["source_id"].astype(str).values
    tr, _ = folds[k][proto][r][fo]; sc = []
    for itr, ite in inner_splits(tr, g, proto, 900 + 10 * r + fo):
        m = CatBoostRegressor(**GRID[ci], random_seed=2026 + r, verbose=0, thread_count=1).fit(X[itr], y[itr])
        sc.append(np.sqrt(mean_squared_error(y[ite], m.predict(X[ite]))))
    return k, proto, r, fo, ci, np.mean(sc)
def refit(k, proto, r, fo, ci):
    import warnings; warnings.filterwarnings("ignore")
    d = sets[k]; X = d[FEAT[k]].values.astype(float); y = d[TARGET[k]].values; tr, te = folds[k][proto][r][fo]
    return k, proto, r, te, CatBoostRegressor(**GRID[ci], random_seed=2026 + r, verbose=0, thread_count=1).fit(X[tr], y[tr]).predict(X[te])
def run_with_bar(func, jobs, desc):
    gen = Parallel(n_jobs=-1, return_as="generator_unordered")(delayed(func)(*j) for j in jobs)
    return [res for res in tqdm(gen, total=len(jobs), desc=desc, unit="job",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [spent {elapsed} | left {remaining} | {rate_fmt}]")]
jobs = [(k, p, r, fo, ci) for k in sets for p in ["random", "grouped"] for r in range(NREP_T) for fo in range(5) for ci in range(len(GRID))]
S1 = pd.DataFrame(run_with_bar(score_cfg, jobs, "Stage 1/2 inner search"), columns=["target", "protocol", "rep", "fold", "cfg", "inner_rmse"])
best = S1.loc[S1.groupby(["target", "protocol", "rep", "fold"])["inner_rmse"].idxmin()]
out = run_with_bar(refit, [(b.target, b.protocol, b.rep, b.fold, b.cfg) for b in best.itertuples()], "Stage 2/2 refit best")
pred = {}
for k, p, r, te, pr in out: pred.setdefault((k, p, r), np.full(len(sets[k]), np.nan))[te] = pr
TU = pd.DataFrame([[k, p, r, "tuned (nested)", r2_score(sets[k][TARGET[k]].values, pr), np.sqrt(mean_squared_error(sets[k][TARGET[k]].values, pr))]
                   for (k, p, r), pr in pred.items()], columns=["target", "protocol", "rep", "config", "R2", "RMSE"])
ALL = pd.concat([R.query("model == 'CatBoost' and rep < @NREP_T").assign(config="fixed")[TU.columns], TU])
T9 = ALL.groupby(["target", "config", "protocol"])[["R2", "RMSE"]].agg(["mean", "std"]).round(3)
T9.to_excel(os.path.join(OUT, "step10_tuning_sensitivity.xlsx")); print(T9)

# ## Step 11 — code relations (Tables 2A/2B) and code relations on the external set (Table 8)
STEP(11, "code relations")

fl, dt = sets["flex"], sets["dts"]
FB100 = 30.9; FB40 = 30.9 * 28.9 / 24.4       # JSCE (2006) reference UFC; fc,avg = 194 MPa, ft,avg = 11.3 MPa
def jsce_flex(fc, depth): return np.interp(np.nan_to_num(depth, nan=40), [40, 100], [FB40, FB100]) / 194 * fc
steam_t = (dt["cure_steam"] + dt["cure_auto"]).clip(0, 1)
REL = [("ACI 318 / AASHTO, 0.62 sqrt(fc)", "flex", 0.62 * np.sqrt(fl["fc_28"])),
       ("JSCE 2006 ratio, size-adjusted", "flex", jsce_flex(fl["fc_28"].values, fl["prism_depth"].values)),
       ("Graybeal, 0.556 or 0.689 sqrt(fc)", "dts", np.where(steam_t == 1, 0.689, 0.556) * np.sqrt(dt["fc_28"])),
       ("fib MC2010, 2.12 ln(1+0.1 fcm)", "dts", 2.12 * np.log(1 + 0.1 * dt["fc_28"])),
       ("JSCE 2006 ratio, 11.3/194", "dts", 11.3 / 194 * dt["fc_28"])]
rows = []
for name, k, p in REL:
    y = (fl["f_flex"] if k == "flex" else dt["f_dts"]).values; p = np.asarray(p, float); rr = p / y
    rows.append([name, k, len(y), r2_score(y, p), rr.mean(), rr.std() / rr.mean()])
T2A = pd.DataFrame(rows, columns=["relation", "target", "n", "R2", "mean pred/meas", "CoV"])
hi = dt["fc_28"] >= 150
T2B = pd.DataFrame([["T/CECS 10107, 14/120 x fc", "flex", len(fl), 100 * np.mean(fl["f_flex"] < 14 / 120 * fl["fc_28"])],
                    ["NF P18-470, fctk,el >= 6.0 MPa", "dts", len(dt), 100 * np.mean(dt["f_dts"] < 6.0)],
                    ["NF P18-470, 6.0 MPa, fc >= 150 MPa", "dts", int(hi.sum()), 100 * np.mean(dt.loc[hi, "f_dts"] < 6.0)]],
                   columns=["provision", "target", "n", "violating %"])
Ek = E[~ruleB]; ye, fce = Ek["FS"].values, Ek["CS"].values
TX = pd.DataFrame([[n, len(ye), r2_score(ye, p), np.sqrt(mean_squared_error(ye, p)), np.mean(p / ye)]
                   for n, p in [("JSCE ratio, 30.9/194", FB100 / 194 * fce), ("ACI 318, 0.62 sqrt(fc)", 0.62 * np.sqrt(fce))]],
                  columns=["predictor", "n", "R2", "RMSE", "mean pred/meas"])
print("median flex/fc ratio  training:", round((fl["f_flex"] / fl["fc_28"]).median(), 3), "| external rule B:", round(np.median(ye / fce), 3))
with pd.ExcelWriter(os.path.join(OUT, "step11_code_relations.xlsx")) as w:
    T2A.to_excel(w, sheet_name="Table2A", index=False); T2B.to_excel(w, sheet_name="Table2B", index=False); TX.to_excel(w, sheet_name="external_codes", index=False)
print(T2A.round(3)); print(T2B.round(1)); print(TX.round(3))


# ===== CHECKPOINT for verify_paper.py =====
REL = [(n, k, np.asarray(p, float)) for n, k, p in REL]
_keep = ["sets","FEAT","TARGET","folds","R","B","VS","LC","SEL","sel","ABm","shap_tab","SV","C","PSc","SP","per_rep","PS4",
         "FSc","X6","TX","T2A","T2B","A","ALL","best","GRID","exposure","E","ruleB","ye","fce","fl","dt","REL","KS"]
pickle.dump({n: globals()[n] for n in _keep}, open(os.path.join(OUT, "checkpoint_for_verify.pkl"), "wb"))
print(f"\nAll analyses finished in {(_time.time() - _T0) / 60:.1f} min. Now run:  python verify_paper.py")
