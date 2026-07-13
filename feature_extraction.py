#!/usr/bin/env python3

import argparse, json, sys
import numpy as np

CH = ["ax_raw", "ay_raw", "az_raw", "acs_raw", "vfd_torque_Nm"]  # 5 canales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help=".npz de parse_session.py")
    ap.add_argument("--out", required=True, help=".npz de ventanas espectrales")
    ap.add_argument("--n", type=int, default=2000, help="muestras por ventana")
    ap.add_argument("--hop", type=int, default=337, help="salto entre ventanas")
    ap.add_argument("--fmax", type=float, default=60.0, help="frecuencia maxima (Hz). MEJORA: 60 enfoca 1x(20)/2x(40)/3x(60) y bota ruido >100 Hz")
    ap.add_argument("--bins", type=int, default=121, help="numero de bins fijos (0..fmax). 121 en 0-60 Hz = paso 0.5 Hz")
    ap.add_argument("--detrend", action="store_true", help="restar media por ventana")
    a = ap.parse_args()

    d = np.load(a.inp, allow_pickle=True)
    X = d["X"].astype(float)                 # (N, 5) crudo
    meta = json.loads(str(d["meta"]))
    fs = meta["fs_hz"]
    Nsamp = X.shape[0]

    if Nsamp < a.n:
        sys.exit(f"ERROR: la sesion tiene {Nsamp} muestras, menos que la ventana N={a.n}.")
    if X.shape[1] != len(CH):
        sys.exit(f"ERROR: se esperaban {len(CH)} canales, hay {X.shape[1]}.")

    # rejilla de frecuencias: nativa (de la FFT) y fija (objetivo, comun a todas las sesiones)
    freqs_native = np.fft.rfftfreq(a.n, d=1.0 / fs)
    freqs_fixed = np.linspace(0.0, a.fmax, a.bins)        # 241 puntos, 0.5 Hz
    hann = np.hanning(a.n)
    cg = np.sum(hann)                                     # ganancia coherente (Hann)

    starts = range(0, Nsamp - a.n + 1, a.hop)
    feats = []
    for s in starts:
        seg = X[s:s + a.n, :].copy()                     # (N, 5)
        if a.detrend:
            seg -= seg.mean(axis=0, keepdims=True)
        seg *= hann[:, None]                             # ventana de Hann por canal
        mag = np.abs(np.fft.rfft(seg, axis=0)) * (2.0 / cg)   # amplitud unilateral
        # interpolar cada canal a la rejilla fija de 241 bins
        win = np.empty((a.bins, len(CH)), dtype=np.float32)
        for c in range(len(CH)):
            win[:, c] = np.interp(freqs_fixed, freqs_native, mag[:, c])
        feats.append(win)

    F = np.stack(feats, axis=0)                          # (n_ventanas, 241, 5)
    n_win = F.shape[0]
    label = meta.get("label", -1)
    y = np.full(n_win, label, dtype=np.int64)

    np.savez_compressed(
        a.out,
        F=F,
        y=y,
        freqs=freqs_fixed.astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
        params=json.dumps({"n": a.n, "hop": a.hop, "fmax": a.fmax,
                           "bins": a.bins, "detrend": a.detrend}, ensure_ascii=False),
    )

    overlap = 100.0 * (1 - a.hop / a.n)
    print(f"\n===== feature_extraction: {meta['session']} =====")
    print(f"Entrada            : {a.inp}  ({Nsamp} muestras @ {fs} Hz)")
    print(f"Ventana            : N={a.n} ({a.n/fs:.2f} s)  salto={a.hop}  solape={overlap:.0f}%")
    print(f"FFT                : 0-{a.fmax:.0f} Hz  ->  {a.bins} bins (paso {a.fmax/(a.bins-1):.3f} Hz)")
    print(f"Detrend            : {'si' if a.detrend else 'no (conserva DC)'}")
    print(f"Ventanas generadas : {n_win}")
    print(f"Forma del tensor F : {F.shape}  (n_ventanas x bins x canales)")
    print(f"Etiqueta           : {label}  (1=desbalance, 0=sano, -1=sin etiqueta)")
    print(f"Salida             : {a.out}")
    if meta.get("vib_corrupta_EMI"):
        print("AVISO: la vibracion venia corrupta por EMI; estas ventanas sirven para "
              "probar el pipeline, no para resultados.")


if __name__ == "__main__":
    main()
