#!/usr/bin/env python3
"""
evaluate.py  —  septimo eslabon: metricas de publicacion.

Lee las predicciones que guardo train.py (runs/<protocolo>_<fold>_pred.npz) y calcula:
  - ROC-AUC por fold + AUC MACRO (media de folds)   -> metrica DG principal (Fawcett 2006)
  - AUC agrupado (pool de todas las predicciones) + IC por bootstrap (honesto, suele ser ancho)
  - AUC POR CAPTURA (votacion) + IC por bootstrap de CAPTURAS  -> metrica principal
  - Sensibilidad, especificidad, exactitud, F1 en un umbral
  - ECE (error de calibracion esperado, definicion de fiabilidad)
  - Matriz de confusion

LODO: cada fold tiene ambas clases -> AUC por fold + macro.
LOSO: con sesion=dia cada fold tiene ambas clases; aun asi el pool es robusto.

Uso:
  python evaluate.py --runs runs/ --protocol lodo
  opciones: --threshold 0.5 --boot 2000 --bins 10 --out reporte.json

Requiere: numpy
"""
import argparse, glob, json, os
import numpy as np


def auc_score(y, s):
    """ROC-AUC (Mann-Whitney con rangos promedio)."""
    y = np.asarray(y); s = np.asarray(s)
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); sv = s[order]; i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def bootstrap_auc(y, s, n_boot=2000, seed=0):
    """Bootstrap a NIVEL DE VENTANA. OJO: si las ventanas estan correlacionadas
    (vienen de pocas capturas), este IC sale demasiado angosto. Ver capture_bootstrap_auc."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc_score(y[idx], s[idx])
        if not np.isnan(a):
            vals.append(a)
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def capture_bootstrap_auc(y, s, cap, n_boot=2000, seed=0, vote=True):
    """Bootstrap por GRUPOS remuestreando CAPTURAS completas (no ventanas sueltas).
    Es el IC honesto cuando las ~529 ventanas de una misma captura estan correlacionadas.
      vote=True  -> AUC sobre el VOTO por captura (1 score promedio por captura).
      vote=False -> AUC a nivel ventana, pero remuestreando capturas enteras.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s); cap = np.asarray(cap)
    caps = np.unique(cap)
    idx_by = {c: np.where(cap == c)[0] for c in caps}
    cy = {c: int(round(y[idx_by[c]].mean())) for c in caps}     # etiqueta de la captura
    cs = {c: float(s[idx_by[c]].mean()) for c in caps}          # score promedio
    K = len(caps)
    vals = []
    for _ in range(n_boot):
        pick = caps[rng.integers(0, K, K)]
        if vote:
            yb = np.array([cy[c] for c in pick])
            sb = np.array([cs[c] for c in pick])
        else:
            yb = np.concatenate([y[idx_by[c]] for c in pick])
            sb = np.concatenate([s[idx_by[c]] for c in pick])
        a = auc_score(yb, sb)
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def best_threshold_youden(y, s):
    """Umbral que maximiza Youden J = sens + espec - 1 (mejor punto de operacion)."""
    y = np.asarray(y); s = np.asarray(s)
    thrs = np.unique(s)
    best_j, best_t = -1.0, 0.5
    P = (y == 1).sum(); N = (y == 0).sum()
    for t in thrs:
        pred = s >= t
        tp = float((pred & (y == 1)).sum()); tn = float((~pred & (y == 0)).sum())
        sens = tp / P if P else 0.0; spec = tn / N if N else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t, best_j


def ece_score(y, p, n_bins=10):
    """ECE de FIABILIDAD (Naeini 2015): bin por P(falla) predicha; en cada bin compara
    la probabilidad media predicha con la FRECUENCIA OBSERVADA de falla.

    FIX: antes se comparaba contra la exactitud del umbral 0.5, lo que inflaba el ECE
    de forma sistematica para p<0.5 (un modelo perfectamente calibrado daba ECE alto).
    Ahora coincide con figuras_resultados.py y es la definicion estandar."""
    y = np.asarray(y); p = np.asarray(p); N = len(y)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean(); freq = y[m].mean()         # frecuencia observada de positivos
        e += m.sum() / N * abs(freq - conf)
    return float(e)


def clf_metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    FP = int(((pred == 1) & (y == 0)).sum()); FN = int(((pred == 0) & (y == 1)).sum())
    sens = TP / (TP + FN) if (TP + FN) else float("nan")
    spec = TN / (TN + FP) if (TN + FP) else float("nan")
    acc = (TP + TN) / len(y)
    bal_acc = (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else float("nan")
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, sens=sens, spec=spec, acc=acc, bal_acc=bal_acc, f1=f1)


def fold_detail(yf, sf, capf, thr, boot=2000, seed=0):
    """Metricas completas de UN fold (en LODO = un fluido; en LOSO = un dia) al umbral
    global thr, con IC95% de AUC por bootstrap de las CAPTURAS de ese fold."""
    yf = np.asarray(yf); sf = np.asarray(sf); capf = np.asarray(capf)
    auc_win = auc_score(yf, sf)
    capsf = np.unique(capf)
    cyf = np.array([int(round(yf[capf == c].mean())) for c in capsf])
    csf = np.array([sf[capf == c].mean() for c in capsf])
    auc_capf = auc_score(cyf, csf)
    _, lo, hi = capture_bootstrap_auc(yf, sf, capf, boot, seed=seed, vote=True)
    m = clf_metrics(yf, sf, thr)
    P = int((yf == 1).sum()); N = int((yf == 0).sum())
    return dict(auc=auc_win, auc_captura=auc_capf, ci_lo=lo, ci_hi=hi,
                f1=m["f1"], sens=m["sens"], spec=m["spec"],
                n_win=int(len(yf)), n_cap=int(len(capsf)), n_falla=P, n_sano=N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--protocol", choices=["lodo", "loso"], default="lodo")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.runs, f"{a.protocol}_*_pred.npz")))
    if not paths:
        raise SystemExit(f"ERROR: no hay predicciones {a.protocol}_*_pred.npz en {a.runs}/")

    fold_aucs = []; per_fold = []; ally = []; alls = []; allcap = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        y = d["y"]; s = d["score"]; meta = json.loads(str(d["meta"]))
        ally.append(y); alls.append(s)
        allcap.append(d["capture"] if "capture" in d.files else np.array([str(meta["held_out"])]*len(y)))
        fa = auc_score(y, s)
        per_fold.append((meta["held_out"], fa, len(y)))
        if not np.isnan(fa):
            fold_aucs.append(fa)

    y = np.concatenate(ally); s = np.concatenate(alls); cap = np.concatenate(allcap)

    # ---- VOTACION POR CAPTURA: promedia las ventanas de cada captura -> 1 voto ----
    caps = np.unique(cap)
    cap_y = np.array([int(round(y[cap == c].mean())) for c in caps])   # etiqueta de la captura
    cap_s = np.array([s[cap == c].mean() for c in caps])               # score promedio
    auc_cap = auc_score(cap_y, cap_s)

    macro = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
    macro_sd = float(np.std(fold_aucs)) if fold_aucs else float("nan")
    pooled = auc_score(y, s)
    bmean, blo, bhi = bootstrap_auc(y, s, a.boot)                          # nivel ventana (optimista)
    # IC HONESTO: remuestrea CAPTURAS completas, sobre el voto por captura
    capmean, caplo, caphi = capture_bootstrap_auc(y, s, cap, a.boot, vote=True)
    cm = clf_metrics(y, s, a.threshold)
    # umbral optimo por Youden + sus metricas
    t_opt, j_opt = best_threshold_youden(y, s)
    cm_opt = clf_metrics(y, s, t_opt)
    ece = ece_score(y, s, a.bins)

    # ---- METRICAS POR FLUIDO (LODO) o POR DIA (LOSO): cada fold es uno ----
    # SIN FUGA DE UMBRAL: para evaluar el fluido i, el umbral de Youden se elige usando
    # SOLO los OTROS fluidos (leave-one-out tambien para el umbral). Asi el umbral nunca
    # ve el fluido de test, eliminando la fuga que inflaba F1/Sens/Espec.
    etiqueta = "fluido" if a.protocol == "lodo" else "dia"
    per_fluid = []
    thr_por_fluido = {}
    for (ho, fa, n), yf, sf, capf in zip(per_fold, ally, alls, allcap):
        if np.isnan(auc_score(yf, sf)):          # fold de una sola clase -> no aplica
            continue
        # umbral elegido con TODOS los fluidos MENOS este (sin fuga)
        y_otros = np.concatenate([yy for (hh, _, _), yy in zip(per_fold, ally) if hh != ho])
        s_otros = np.concatenate([ss for (hh, _, _), ss in zip(per_fold, alls) if hh != ho])
        t_loo, _ = best_threshold_youden(y_otros, s_otros)
        thr_por_fluido[str(ho)] = float(t_loo)
        det = fold_detail(yf, sf, capf, t_loo, a.boot)   # se aplica el umbral SIN fuga
        det["fluido"] = str(ho)
        det["thr_usado"] = float(t_loo)
        per_fluid.append(det)

    print(f"\n===== evaluate: protocolo {a.protocol.upper()} =====")
    print(f"Folds: {len(paths)}   ventanas totales: {len(y)}  (desb={int((y==1).sum())}, sano={int((y==0).sum())})")
    print("\nAUC por fold:")
    for ho, fa, n in per_fold:
        print(f"   {str(ho):<14} AUC={'n/d (1 clase)' if np.isnan(fa) else f'{fa:.3f}':<14} (n={n})")
    if fold_aucs:
        print(f"\nAUC MACRO (media de folds) : {macro:.3f} ± {macro_sd:.3f}")
    print(f"AUC agrupado (pool)        : {pooled:.3f}")
    print(f"AUC POR CAPTURA (votacion) : {auc_cap:.3f}   <- METRICA PRINCIPAL ({len(caps)} capturas)")
    print(f"   IC 95% por CAPTURA      : [{caplo:.3f}, {caphi:.3f}]  (honesto: remuestrea capturas enteras)")
    print(f"   IC 95% por ventana      : [{blo:.3f}, {bhi:.3f}]  (optimista, NO usar como principal)")
    print(f"\nEn umbral {a.threshold}:")
    print(f"   Sensibilidad : {cm['sens']:.3f}   Especificidad : {cm['spec']:.3f}")
    print(f"   Exactitud    : {cm['acc']:.3f}   Bal.Acc : {cm['bal_acc']:.3f}   F1 : {cm['f1']:.3f}")
    print(f"\nEn umbral OPTIMO (Youden={t_opt:.3f}, J={j_opt:.3f}):")
    print(f"   Sensibilidad : {cm_opt['sens']:.3f}   Especificidad : {cm_opt['spec']:.3f}")
    print(f"   Exactitud    : {cm_opt['acc']:.3f}   Bal.Acc : {cm_opt['bal_acc']:.3f}   F1 : {cm_opt['f1']:.3f}")
    print(f"\n   ECE (calibracion, fiabilidad) : {ece:.3f}")
    print(f"\nMatriz de confusion (umbral {a.threshold}):")
    print(f"                 pred sano   pred desb")
    print(f"   real sano        {cm['TN']:<10} {cm['FP']}")
    print(f"   real desb        {cm['FN']:<10} {cm['TP']}")

    # ---- tabla por fluido/dia (corrige Oblig. 2: AUC, F1, sens, spec, IC95% por fluido) ----
    if per_fluid:
        print(f"\n===== METRICAS POR {etiqueta.upper()} (umbral SIN FUGA: elegido con los otros {etiqueta}s) =====")
        print(f"   {etiqueta:<8} {'thr':>5} {'AUC':>6} {'AUCcap':>7} {'IC95% (cap)':>17} {'F1':>6} {'Sens':>6} {'Spec':>6}  capturas")
        for d in per_fluid:
            print(f"   {d['fluido']:<8} {d['thr_usado']:5.3f} {d['auc']:6.3f} {d['auc_captura']:7.3f}  "
                  f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] {d['f1']:6.3f} {d['sens']:6.3f} {d['spec']:6.3f}   {d['n_cap']}")
        csv_path = os.path.join(a.runs, f"tabla_por_{etiqueta}_{a.protocol}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(f"{etiqueta},umbral_sin_fuga,AUC,AUC_captura,IC95_lo,IC95_hi,F1,sensibilidad,especificidad,n_capturas,n_ventanas\n")
            for d in per_fluid:
                f.write(f"{d['fluido']},{d['thr_usado']:.4f},{d['auc']:.4f},{d['auc_captura']:.4f},{d['ci_lo']:.4f},{d['ci_hi']:.4f},"
                        f"{d['f1']:.4f},{d['sens']:.4f},{d['spec']:.4f},{d['n_cap']},{d['n_win']}\n")
        print(f"\nTabla por {etiqueta} guardada: {csv_path}")

    report = dict(protocol=a.protocol, n_folds=len(paths), n_windows=int(len(y)),
                  n_captures=int(len(caps)),
                  auc_macro=macro, auc_macro_sd=macro_sd, auc_pooled=float(pooled),
                  auc_por_captura=float(auc_cap),
                  auc_ci95_captura=[caplo, caphi], auc_ci95_window=[blo, bhi],
                  auc_ci95_cluster=[caplo, caphi],   # alias para compatibilidad con make_tables.py
                  threshold=a.threshold, threshold_opt=t_opt, youden_j=j_opt, ece=ece,
                  metrics_at_threshold={k: cm[k] for k in ("sens", "spec", "acc", "bal_acc", "f1", "TP", "TN", "FP", "FN")},
                  metrics_at_opt={k: cm_opt[k] for k in ("sens", "spec", "acc", "bal_acc", "f1", "TP", "TN", "FP", "FN")},
                  per_fold=[{"held_out": str(ho), "auc": (None if np.isnan(fa) else fa), "n": n}
                            for ho, fa, n in per_fold],
                  per_fluid=[{"fluido": d["fluido"], "auc": d["auc"], "auc_captura": d["auc_captura"],
                              "ci95": [d["ci_lo"], d["ci_hi"]], "f1": d["f1"], "sens": d["sens"],
                              "spec": d["spec"], "n_capturas": d["n_cap"], "n_ventanas": d["n_win"]}
                             for d in per_fluid],
                  threshold_por_fluido=thr_por_fluido)
    out = a.out or os.path.join(a.runs, f"reporte_{a.protocol}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReporte guardado: {out}")


if __name__ == "__main__":
    main()
