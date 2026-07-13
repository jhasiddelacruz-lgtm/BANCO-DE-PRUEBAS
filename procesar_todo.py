#!/usr/bin/env python3
"""
procesar_todo.py - corre el pipeline (parse_session + feature_extraction) sobre
TODAS las capturas .csv de una carpeta, infiriendo fluido y etiqueta del nombre.

Convencion de nombres (real):  [rep]<condicion>_<dia>_cmc<gramos>gr_<freq>.csv
  ej.  01sano_D1_cmc75gr_40.csv ,  0190grados_D3_cmc125gr_40.csv
  - etiqueta: nombre contiene 'sano' -> 0 (sano) ; si no -> 1 (falla: roto/hueco)
  - fluido:   regex cmc(\\d+) -> cmc50/75/100/125/150 ; si no, 'agua' ; si no, 'cmc'
  - sesion:   (FIX opcion A) el DIA (D1/D2/D3), para que LOSO deje un DIA entero fuera
              y mida robustez inter-dia de verdad. El id de captura por-archivo se
              conserva aparte (build_dataset usa el nombre del feat como 'capture').

Despues de correr esto:
  python build_dataset.py --features feats/ --out dataset.npz
  python train.py --dataset dataset.npz --protocol lodo --out-dir runs --epochs 80
  python evaluate.py --runs runs --protocol lodo

Uso:
  python procesar_todo.py --dir . --freq 40
"""
import sys as _sys
try:
    _sys.stdout.reconfigure(encoding='utf-8')
    _sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import os, glob, subprocess, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="carpeta con las capturas .csv")
    ap.add_argument("--freq", type=float, default=40)
    ap.add_argument("--parsed", default="parsed")
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--hz-min", dest="hz_min", type=float, default=0.0,
                    help="passthrough a parse_session: descarta arranque (vfd_hz<hz_min). "
                         "0 = no filtra (por defecto; tus datos ya arrancan estables).")
    a = ap.parse_args()
    os.makedirs(a.parsed, exist_ok=True); os.makedirs(a.feats, exist_ok=True)

    csvs = sorted(f for f in glob.glob(os.path.join(a.dir, "*.csv")) if "_key" not in f.lower())
    if not csvs:
        sys.exit("No se encontraron capturas .csv en " + a.dir)
    print(f"Encontradas {len(csvs)} capturas\n")

    # Forzar UTF-8 en los subprocesos (evita UnicodeEncodeError 'charmap' en Windows)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    import re
    ok = 0; err = []; sin_dia = []
    for csv in csvs:
        stem = os.path.splitext(os.path.basename(csv))[0]
        low = stem.lower()
        # Detecta la concentracion EXACTA de CMC (50/75/100/125/150) como dominio.
        # Asi cmc50, cmc75, cmc100, cmc125, cmc150 son fluidos DISTINTOS (clave para DG).
        m = re.search(r"cmc\s*(\d+)", low)
        if m:
            fluid = "cmc" + m.group(1)        # ej. cmc75, cmc125
        elif "agua" in low:
            fluid = "agua"
        else:
            fluid = "cmc"                      # fallback si no hay numero
            print(f"  *** AVISO: '{stem}' no trae concentracion de CMC ni 'agua'; "
                  f"se etiqueta como 'cmc' generico (COLAPSA dominios para DG). Renombra. ***")
        label = 0 if "sano" in low else 1

        # (FIX opcion A) sesion = DIA (D1/D2/D3). Fallback: nombre completo si no hay dia.
        md = re.search(r"_(D\d+)_", stem)
        if md:
            session = md.group(1)
        else:
            session = stem
            sin_dia.append(stem)

        key = csv[:-4] + "_key.csv"
        key_arg = ["--key", key] if os.path.exists(key) else []
        parsed = os.path.join(a.parsed, stem + ".npz")
        feat = os.path.join(a.feats, stem + ".npz")
        hz_arg = ["--hz-min", str(a.hz_min)] if a.hz_min > 0 else []
        print(f"-> {stem:28s} fluido={fluid:6s} label={label}  sesion={session}", end="  ")
        r1 = subprocess.run([sys.executable, "parse_session.py", "--raw", csv, *key_arg,
              "--out", parsed, "--fluid", fluid, "--freq", str(a.freq),
              "--label", str(label), "--session", session, *hz_arg],
              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
              encoding="utf-8", errors="replace", env=env)
        if r1.returncode != 0:
            print("ERROR parse"); err.append((stem, r1.stderr[-200:])); continue
        r2 = subprocess.run([sys.executable, "feature_extraction.py", "--in", parsed, "--out", feat],
              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
              encoding="utf-8", errors="replace", env=env)
        if r2.returncode != 0:
            print("ERROR features"); err.append((stem, r2.stderr[-200:])); continue
        print("OK"); ok += 1

    print(f"\nProcesadas OK: {ok}/{len(csvs)}   ->  features en {a.feats}/")
    if sin_dia:
        print(f"\nAVISO: {len(sin_dia)} archivo(s) sin patron _Dx_ usaron su nombre como sesion "
              f"(no entran al LOSO inter-dia): {sin_dia}")
    if err:
        print("\nCon error:")
        for s, e in err: print(f"  {s}: {e.strip()[:150]}")
    print("\nSiguiente paso:")
    print(f"  python build_dataset.py --features {a.feats}/ --out dataset.npz")
    print(f"  python train.py --dataset dataset.npz --protocol lodo --out-dir runs --epochs 80")
    print(f"  python evaluate.py --runs runs --protocol lodo")

if __name__ == "__main__":
    main()
