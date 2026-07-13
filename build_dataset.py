#!/usr/bin/env python3

import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import argparse, glob, json, os, sys
import numpy as np

CH = ["ax_raw", "ay_raw", "az_raw", "acs_raw", "vfd_torque_Nm"]


# ---------------------------------------------------------------- ensamblado
def assemble(paths):
    Fs, ys, fluid, session, freq, fill, capture = [], [], [], [], [], [], []
    freqs = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if "F" not in d.files or "meta" not in d.files:
            print(f"  (omito {os.path.basename(p)}: no es un .npz de features)")
            continue
        F = d["F"]; y = d["y"]; meta = json.loads(str(d["meta"]))
        if freqs is None:
            freqs = d["freqs"]
        n = F.shape[0]
        cap = os.path.splitext(os.path.basename(p))[0]   # id unico de captura (= archivo)
        Fs.append(F); ys.append(y)
        fluid   += [meta.get("fluid", "?")] * n
        session += [meta.get("session", "?")] * n
        freq    += [meta.get("vfd_freq_hz", 0)] * n
        fill    += [meta.get("fill_pct", 0)] * n
        capture += [cap] * n                              # para votar por captura

    if not Fs:
        sys.exit("ERROR: ningun .npz valido (faltan claves 'F'/'meta'). "
                 "Vuelve a correr feature_extraction.py / procesar_todo.py.")

    shapes = {f.shape[1:] for f in Fs}
    if len(shapes) > 1:
        sys.exit("ERROR: los features tienen formas de ventana distintas "
                 f"(tiempo/canales): {sorted(shapes)}. "
                 "Reprocesa todo con los mismos parametros.")

    F = np.concatenate(Fs, axis=0)
    y = np.concatenate(ys, axis=0)
    return (F, y, np.array(fluid), np.array(session),
            np.array(freq, dtype=float), np.array(fill, dtype=float),
            np.array(capture), freqs)


# ---------------------------------------------------------------- particiones
def lodo_folds(fluid):
    """Genera (fluido_dejado_fuera, idx_train, idx_test) por cada fluido."""
    for fl in sorted(set(fluid.tolist())):
        test = np.where(fluid == fl)[0]
        train = np.where(fluid != fl)[0]
        yield fl, train, test


def loso_folds(session):
    """Genera (sesion_dejada_fuera, idx_train, idx_test) por cada sesion."""
    for s in sorted(set(session.tolist())):
        test = np.where(session == s)[0]
        train = np.where(session != s)[0]
        yield s, train, test


# ---------------------------------------------------------------- z-score (por canal)
def zscore_fit(F_train):
    """Ajusta media/desv por CANAL usando solo el train. Devuelve (mean[5], std[5])."""
    mean = F_train.mean(axis=(0, 1))
    std = F_train.std(axis=(0, 1)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def zscore_apply(F, mean, std):
    return ((F - mean) / std).astype(np.float32)


def load_dataset(path):
    d = np.load(path, allow_pickle=True)
    out = dict(F=d["F"], y=d["y"], fluid=d["fluid"], session=d["session"],
               freq=d["freq"], fill=d["fill"], freqs=d["freqs"])
    out["capture"] = d["capture"] if "capture" in d.files else d["session"]  # compatibilidad
    return out


# ---------------------------------------------------------------- reporte
def _balance(y):
    p = int(np.sum(y == 1)); n = int(np.sum(y == 0)); u = int(np.sum(y == -1))
    return p, n, u


def report(F, y, fluid, session):
    print(f"\n===== build_dataset =====")
    print(f"Ventanas totales   : {F.shape[0]}   tensor {F.shape}")
    p, n, u = _balance(y)
    print(f"Balance de clases  : falla={p}  sano={n}" + (f"  SIN_ETIQUETA={u}" if u else ""))
    if u:
        print("  AVISO: hay ventanas sin etiqueta (-1). Asigna --label en parse_session.")

    fls = sorted(set(fluid.tolist()))
    print(f"\nFluidos (dominios) : {len(fls)} -> {fls}")
    for fl in fls:
        m = fluid == fl
        pp, nn, _ = _balance(y[m])
        print(f"   {fl:<14} ventanas={int(m.sum()):<6} (falla={pp}, sano={nn})")
    if len(fls) < 2:
        print("  AVISO: LODO necesita >=2 fluidos. Con 1 fluido no hay generalizacion de dominio.")

    ses = sorted(set(session.tolist()))
    print(f"\nSesiones           : {len(ses)}")
    for s in ses:
        m = session == s
        pp, nn, _ = _balance(y[m])
        print(f"   {s:<14} ventanas={int(m.sum()):<6} (falla={pp}, sano={nn})")

    print(f"\nFolds LODO (deja-un-fluido-fuera):")
    for fl, tr, te in lodo_folds(fluid):
        pp, nn, _ = _balance(y[te])
        deg = " <-- DEGENERADO (una sola clase en test)" if (pp == 0 or nn == 0) else ""
        print(f"   test={fl:<12} train={len(tr):<6} test={len(te):<6} (falla={pp}, sano={nn}){deg}")

    print(f"\nFolds LOSO (deja-una-sesion-fuera): {len(ses)} folds")


def baseline_subtract(F, y, fluid, session, eps=1e-6, ref_fluid=None):
    """MEJORA: resta la linea base SANA a TODAS las ventanas.

    Dos modos:
    - ref_fluid=None (por defecto): base SANA por cada (dia+fluido). Calibra cada
      fluido. Da el AUC mas alto pero requiere sano de cada fluido.
    - ref_fluid='cmc100' (u otro): UNA sola base, la del SANO del fluido de
      referencia (promediada sobre sus dias), aplicada a TODOS los fluidos. Simula
      'calibro UNA vez y detecto en fluidos NO calibrados' -> prueba el aporte real
      de generalizacion sin recalibrar por fluido.
    """
    Fout = F.astype(np.float64).copy()
    sin_base = 0
    if ref_fluid is not None:
        m_ref = (fluid == ref_fluid) & (y == 0)
        if not m_ref.any():
            print(f"  AVISO: no hay sano del fluido de referencia '{ref_fluid}'. Uso base por fluido.")
            ref_fluid = None
        else:
            base_global = F[m_ref].mean(axis=0, keepdims=True)
            Fout = np.log((F + eps) / (base_global + eps))
            print(f"  Calibracion UNICA con el sano de '{ref_fluid}' aplicada a TODOS los fluidos.")
            return Fout.astype(np.float32)
    # Agrupa por DIA+FLUIDO (no por el nombre completo del archivo).
    # Asi el sano y la falla del MISMO dia+fluido comparten la misma base sana.
    # Extrae el dia (D1/D2/D3) del nombre de sesion; si no hay, usa la sesion entera.
    import re as _re
    def dia_de(sess):
        m = _re.search(r"_(D\d+)_", str(sess))
        return m.group(1) if m else str(sess)
    dia = np.array([dia_de(s) for s in session])
    grupo = np.array([f"{d}|{fl}" for d, fl in zip(dia, fluid)])
    for g in np.unique(grupo):
        m = (grupo == g)
        if not m.any():
            continue
        m_sano = m & (y == 0)
        if m_sano.any():
            base = F[m_sano].mean(axis=0, keepdims=True)
        else:
            base = F[m].mean(axis=0, keepdims=True); sin_base += 1
        Fout[m] = np.log((F[m] + eps) / (base + eps))
    if sin_base:
        print(f"  AVISO baseline: {sin_base} grupos sin ventanas sanas; usaron su media.")
    return Fout.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="carpeta o glob de featXX.npz")
    ap.add_argument("--out", required=True, help="dataset.npz de salida")
    ap.add_argument("--baseline", action="store_true",
                    help="MEJORA: normaliza restando la linea base SANA de cada dia+fluido "
                         "(mata el confound de dia/fluido; deja solo la firma de falla)")
    ap.add_argument("--baseline-ref", default=None,
                    help="UNA sola calibracion: usa el sano de ESTE fluido (ej. cmc100) para "
                         "TODOS. Prueba 'calibro una vez y detecto en fluidos no calibrados'.")
    a = ap.parse_args()

    paths = (sorted(glob.glob(os.path.join(a.features, "*.npz")))
             if os.path.isdir(a.features) else sorted(glob.glob(a.features)))
    if not paths:
        sys.exit("ERROR: no se encontraron .npz de features.")
    print(f"Ensamblando {len(paths)} sesiones...")

    F, y, fluid, session, freq, fill, capture, freqs = assemble(paths)
    if a.baseline or a.baseline_ref:
        if a.baseline_ref:
            print(f"Aplicando UNA calibracion unica (sano de '{a.baseline_ref}')...")
        else:
            print("Aplicando normalizacion por LINEA BASE SANA (dia+fluido)...")
        F = baseline_subtract(F, y, fluid, session, ref_fluid=a.baseline_ref)
        print("  -> features ahora son log-desviacion respecto al sano de referencia.")
    np.savez_compressed(a.out, F=F, y=y, fluid=fluid, session=session,
                        freq=freq, fill=fill, capture=capture, freqs=freqs)
    report(F, y, fluid, session)
    print(f"\nDataset guardado: {a.out}")
    print("Nota: el z-score se aplica por fold en train.py (zscore_fit sobre el train).")


if __name__ == "__main__":
    main()
