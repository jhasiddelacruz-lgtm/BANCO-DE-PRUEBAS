#!/usr/bin/env python3

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

from model import BioreactorDG
from augmentation import augment_batch
from build_dataset import load_dataset, lodo_folds, loso_folds, zscore_fit, zscore_apply


def auc_score(y_true, y_score):
    """ROC-AUC por Mann-Whitney (con rangos promedio para empates)."""
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n_pos = int((y_true == 1).sum()); n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sv = y_score[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    R_pos = ranks[y_true == 1].sum()
    return (R_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def lambda_at(progress, lambda_max=1.0, gamma=10.0):
    """Calendario sigmoidal de Ganin (2016): 0 -> lambda_max."""
    return lambda_max * (2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0)


def stratified_val_mask(y, frac=0.15, seed=0):
    """Marca ~frac de cada clase como validacion (a NIVEL VENTANA).
    OJO: deja ventanas de una misma captura en train y val a la vez -> fuga.
    Conservada solo por compatibilidad; usar group_val_mask (por sesion)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        k = max(1, int(round(len(idx) * frac)))
        mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def group_val_mask(y, groups, frac=0.15, seed=0):
    """Validacion por GRUPO (sesion/captura), NO por ventana: aparta sesiones
    completas para early stopping -> sin fuga. Intenta que queden ambas clases
    en validacion; si una clase tiene una sola sesion, la deja toda en train."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    groups = np.asarray(groups)
    for c in np.unique(y):
        # sesiones que contienen la clase c
        sess_c = np.unique(groups[y == c])
        if len(sess_c) <= 1:
            continue                                  # 1 sola sesion -> no la saco del train
        k = max(1, int(round(len(sess_c) * frac)))
        k = min(k, len(sess_c) - 1)                   # deja al menos 1 sesion en train
        pick = rng.choice(sess_c, size=k, replace=False)
        for s in pick:
            mask[groups == s] = True
    return mask


def train_one_fold(F, y, fluid, session, train_idx, test_idx, held_out, args, device):
    # z-score ajustado solo con el train del fold
    mean, std = zscore_fit(F[train_idx])
    Xtr = zscore_apply(F[train_idx], mean, std)
    Xte = zscore_apply(F[test_idx], mean, std)
    ytr = y[train_idx].astype(np.float32)
    yte = y[test_idx].astype(np.float32)

    # dominios = fluidos del train, mapeados a enteros
    fl_tr = fluid[train_idx]
    doms = sorted(set(fl_tr.tolist()))
    dom_map = {d: i for i, d in enumerate(doms)}
    dtr = np.array([dom_map[f] for f in fl_tr], dtype=np.int64)
    n_domains = len(doms)
    use_dann = n_domains >= 2                      # sin >=2 fluidos no hay adversario

    # split de validacion por SESION (early stopping sin fuga de ventanas)
    sess_tr = session[train_idx]
    vmask = group_val_mask(ytr, sess_tr, args.val_frac, seed=args.seed)
    if not vmask.any():                            # respaldo si no se pudo agrupar
        vmask = stratified_val_mask(ytr, args.val_frac, seed=args.seed)
    tr = ~vmask
    Xt = torch.tensor(Xtr[tr], device=device); yt = torch.tensor(ytr[tr], device=device)
    dt = torch.tensor(dtr[tr], device=device)
    Xv = torch.tensor(Xtr[vmask], device=device); yv = ytr[vmask]

    # MODELO: por defecto COMPLETO; los switches solo lo cambian si pasas --sin-*
    model = BioreactorDG(in_ch=args.in_ch, n_domains=max(2, n_domains),
                         use_ibn=not args.sin_ibn, use_se=not args.sin_se,
                         use_lstm=not args.sin_lstm, use_chdrop=not args.sin_chdrop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # pos_weight para el desbalance falla/sano, calculado SOLO con el train del fold
    n_pos = float((ytr[tr] == 1).sum()); n_neg = float((ytr[tr] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()

    best_auc, best_state, wait = -1.0, None, 0
    n = Xt.size(0)
    for epoch in range(args.epochs):
        model.train()
        lam = float(lambda_at(epoch / max(1, args.epochs - 1), args.lambda_max)) if use_dann else 0.0
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.batch):
            b = perm[s:s + args.batch]
            xb, yb, db = Xt[b], yt[b], dt[b]
            xb_aug, yb_soft = augment_batch(xb, yb, db, alpha=args.alpha,
                                            mask_width=args.mask_width, mask_p=args.mask_p,
                                            use_mixup=not args.sin_mixup)
            cls, dom = model(xb_aug, lambd=lam)
            loss = bce(cls, yb_soft)
            if use_dann:
                loss = loss + ce(dom, db)
            opt.zero_grad(); loss.backward(); opt.step()

        # validacion
        model.eval()
        with torch.no_grad():
            vlogit, _ = model(Xv, lambd=0.0)
            vscore = torch.sigmoid(vlogit).cpu().numpy()
        vauc = auc_score(yv, vscore)
        if np.isnan(vauc):
            vauc = -1.0
        if vauc > best_auc:
            best_auc, best_state, wait = vauc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # evaluacion del fold de prueba
    model.eval()
    with torch.no_grad():
        tlogit, _ = model(torch.tensor(Xte, device=device), lambd=0.0)
        tscore = torch.sigmoid(tlogit).cpu().numpy()
    tauc = auc_score(yte, tscore)

    return dict(held_out=held_out, n_train=len(train_idx), n_test=len(test_idx),
                n_domains=n_domains, use_dann=use_dann, val_auc=float(best_auc),
                test_auc=float(tauc), test_y=yte, test_score=tscore,
                mean=mean, std=std, dom_map=dom_map, state=model.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--protocol", choices=["lodo", "loso"], default="lodo")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--lambda-max", dest="lambda_max", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.15)
    ap.add_argument("--mask-width", dest="mask_width", type=int, default=20)
    ap.add_argument("--mask-p", dest="mask_p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    # ---- flags de ablacion (por defecto NINGUNO activo -> modelo completo) ----
    ap.add_argument("--sin-ibn", dest="sin_ibn", action="store_true", help="quita IBN (usa BatchNorm)")
    ap.add_argument("--sin-se", dest="sin_se", action="store_true", help="quita la atencion SE")
    ap.add_argument("--sin-lstm", dest="sin_lstm", action="store_true", help="LSTM -> promedio temporal")
    ap.add_argument("--sin-chdrop", dest="sin_chdrop", action="store_true", help="quita Channel Dropout")
    ap.add_argument("--sin-mixup", dest="sin_mixup", action="store_true", help="quita el mixup Dirichlet")
    ap.add_argument("--in-ch", dest="in_ch", type=int, default=5, help="canales de entrada (def. 5)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.dataset)
    F = ds["F"].astype(np.float32); y = ds["y"].astype(np.int64)
    fluid = ds["fluid"]; session = ds["session"]; capture = ds["capture"]
    folds = (lodo_folds(fluid) if args.protocol == "lodo" else loso_folds(session))

    # traza de configuracion (que ablacion se esta corriendo)
    abl = [n for n, on in [("sin_ibn", args.sin_ibn), ("sin_se", args.sin_se),
                           ("sin_lstm", args.sin_lstm), ("sin_chdrop", args.sin_chdrop),
                           ("sin_mixup", args.sin_mixup)] if on]
    cfg = "COMPLETO" if not abl and args.in_ch == 5 else (", ".join(abl) + (f", in_ch={args.in_ch}" if args.in_ch != 5 else ""))
    print(f"Entrenando protocolo {args.protocol.upper()} en {device}  (dataset {F.shape})")
    print(f"Configuracion del modelo: {cfg}")
    results = []
    for held_out, tr_idx, te_idx in folds:
        r = train_one_fold(F, y, fluid, session, tr_idx, te_idx, held_out, args, device)
        results.append(r)
        # guardar modelo + z-score + predicciones del fold (+ capture para votar)
        torch.save(r["state"], os.path.join(args.out_dir, f"{args.protocol}_{held_out}.pt"))
        np.savez(os.path.join(args.out_dir, f"{args.protocol}_{held_out}_pred.npz"),
                 y=r["test_y"], score=r["test_score"], mean=r["mean"], std=r["std"],
                 capture=capture[te_idx],
                 meta=json.dumps({"held_out": str(held_out), "val_auc": r["val_auc"],
                                  "test_auc": r["test_auc"], "n_domains": r["n_domains"]}))
        tag = "" if r["use_dann"] else "  (modo single-source: 1 fluido en train, sin adversario)"
        print(f"  fold {str(held_out):<12} test_AUC={r['test_auc']:.3f}  val_AUC={r['val_auc']:.3f}"
              f"  (train={r['n_train']}, test={r['n_test']}){tag}")

    aucs = [r["test_auc"] for r in results if not np.isnan(r["test_auc"])]
    if aucs:
        print(f"\n{args.protocol.upper()} AUC medio (por fold): {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
              f"({len(aucs)} folds con ambas clases)")

    # AUC GLOBAL agrupando (pool) las predicciones de TODOS los folds.
    # Imprescindible en LOSO, donde cada fold de test es de una sola clase
    # y el AUC por fold no se puede calcular.
    y_pool = np.concatenate([r["test_y"] for r in results])
    s_pool = np.concatenate([r["test_score"] for r in results])
    auc_pool = auc_score(y_pool, s_pool)
    print(f"{args.protocol.upper()} AUC GLOBAL (pooled, {len(y_pool)} ventanas): "
          f"{auc_pool:.3f}   <- usar este como metrica principal en LOSO")
    if not aucs:
        print("  (en LOSO el AUC por fold sale nan por diseno; el pooled es el valido)")

    print(f"Modelos y predicciones guardados en: {args.out_dir}/")


if __name__ == "__main__":
    main()
