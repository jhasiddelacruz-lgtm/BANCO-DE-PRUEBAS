#!/usr/bin/env python3
"""
parse_session.py  —  primer eslabon del pipeline de analisis.

Lee la captura cruda del firmware integrado (producida por capture_serial.py:
  <sesion>.csv       filas: t_us,ax_raw,ay_raw,az_raw,acs_raw,sct_raw,vfd_hz,vfd_torque_Nm
  <sesion>_key.csv   tiempos del keyphasor (1 pulso/vuelta)
), separa los canales, maneja que cada uno viene a distinta tasa, procesa el
keyphasor (RPM + referencia para order tracking), SELECCIONA los 5 canales del
modelo y guarda una sesion limpia en .npz con sus metadatos.

  5 canales del MODELO : ax, ay, az (vibracion) + acs (corriente) + torque (par VFD)
  Canal de RESPALDO    : sct  (no entra al modelo, se guarda aparte)

La señal se mantiene CRUDA (cuentas LSB). La normalizacion z-score se hace despues,
solo con el fold de entrenamiento (anti-fuga). El order tracking angular se deja como
gancho (se guardan los tiempos del keyphasor y las RPM) y se aplica en etapas posteriores.

Uso:
  python parse_session.py --raw sesion01.csv --key sesion01_key.csv --out sesion01.npz \
      --fluid agua --freq 60 --fill 80 --label 0 --session S01

  --label : 1 = desbalance presente, 0 = ausente   (etiqueta binaria)
  --freq  : frecuencia del VFD en Hz (condicion/dominio de operacion)
  --fill  : nivel de llenado del tanque (%)        (dominio secundario)

Requiere: numpy
"""
import argparse, json, os, sys
import numpy as np

COLS = ["t_us", "ax_raw", "ay_raw", "az_raw", "acs_raw", "sct_raw", "vfd_hz", "vfd_torque_Nm"]
MODEL_CHANNELS = ["ax_raw", "ay_raw", "az_raw", "acs_raw", "vfd_torque_Nm"]  # los 5
ADXL_SAT = 32000   # |cuentas| >= esto -> muestra saturada (firma de corrupcion por EMI)


def load_raw(path):
    """Carga las filas de datos del CSV crudo, ignorando #, INFO, KEY y la cabecera."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#" or line.startswith(("t_us", "KEY", "INFO")):
                continue
            p = line.split(",")
            if len(p) != len(COLS):
                continue
            try:
                rows.append([float(x) for x in p])
            except ValueError:
                continue
    if not rows:
        sys.exit(f"ERROR: no se encontraron filas de datos validas en {path}")
    return np.asarray(rows, dtype=float)


def load_key(path):
    """Carga los tiempos del keyphasor (us). Acepta 'KEY,t_us' o un numero por linea."""
    if not path or not os.path.exists(path):
        return np.array([], dtype=float)
    ts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#" or line.lower().startswith("t"):
                continue
            tok = line.split(",")[-1]   # toma el ultimo campo (sirve para 'KEY,12345')
            try:
                ts.append(float(tok))
            except ValueError:
                continue
    return np.asarray(ts, dtype=float)


def effective_rate(values, fs):
    """Tasa real de actualizacion de un canal sostenido = fs * (fraccion de muestras que cambian)."""
    if len(values) < 2:
        return 0.0
    changes = np.count_nonzero(np.diff(values) != 0)
    return fs * changes / (len(values) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="CSV crudo de capture_serial.py")
    ap.add_argument("--key", default=None, help="CSV _key.csv del keyphasor")
    ap.add_argument("--out", required=True, help="archivo .npz de salida")
    ap.add_argument("--fluid", default="desconocido")
    ap.add_argument("--freq", type=float, default=0.0, help="frecuencia VFD (Hz)")
    ap.add_argument("--fill", type=float, default=0.0, help="llenado del tanque (%)")
    ap.add_argument("--label", type=int, default=-1, help="1=desbalance, 0=sano, -1=sin etiqueta")
    ap.add_argument("--session", default=None, help="id de sesion (por defecto: nombre del archivo)")
    a = ap.parse_args()

    raw = load_raw(a.raw)
    key_us = load_key(a.key)

    t_us = raw[:, 0]
    t_s = (t_us - t_us[0]) / 1e6
    fs = (len(t_s) - 1) / (t_s[-1] - t_s[0])

    col = {name: i for i, name in enumerate(COLS)}
    # matriz de los 5 canales del modelo (en orden), crudos
    X = np.column_stack([raw[:, col[c]] for c in MODEL_CHANNELS])
    sct = raw[:, col["sct_raw"]]
    vfd_hz = raw[:, col["vfd_hz"]]

    # tasas reales de los canales sostenidos
    fs_acs = effective_rate(raw[:, col["acs_raw"]], fs)
    fs_vfd = effective_rate(vfd_hz, fs)

    # keyphasor -> RPM (mediana de los intervalos entre pulsos)
    rpm = 0.0
    if len(key_us) >= 2:
        dt = np.diff(np.sort(key_us)) / 1e6           # s entre pulsos
        dt = dt[dt > 0]
        if len(dt):
            rpm = 60.0 / np.median(dt)
    key_t_s = (np.sort(key_us) - t_us[0]) / 1e6 if len(key_us) else np.array([])

    # control de calidad de la vibracion (firma de EMI: muestras saturadas)
    vib = X[:, :3]
    sat_frac = np.mean(np.any(np.abs(vib) >= ADXL_SAT, axis=1))
    vib_corrupta = sat_frac > 0.02   # >2% de ventanas saturadas

    session = a.session or os.path.splitext(os.path.basename(a.raw))[0]
    meta = {
        "session": session,
        "fluid": a.fluid,
        "vfd_freq_hz": a.freq,
        "fill_pct": a.fill,
        "label": a.label,                 # 1 desbalance / 0 sano / -1 sin etiqueta
        "n_samples": int(len(t_s)),
        "fs_hz": round(float(fs), 2),
        "fs_acs_hz": round(float(fs_acs), 1),
        "fs_vfd_hz": round(float(fs_vfd), 2),
        "rpm_keyphasor": round(float(rpm), 1),
        "vfd_hz_medido": round(float(np.median(vfd_hz)), 2),
        "vib_saturada_frac": round(float(sat_frac), 4),
        "vib_corrupta_EMI": bool(vib_corrupta),
        "channels": MODEL_CHANNELS,
    }

    np.savez_compressed(
        a.out,
        t_s=t_s.astype(np.float32),
        X=X.astype(np.float32),
        sct=sct.astype(np.float32),
        key_t_s=key_t_s.astype(np.float32),
        meta=json.dumps(meta, ensure_ascii=False),
    )

    # resumen en consola
    print(f"\n===== parse_session: {session} =====")
    print(f"Salida              : {a.out}")
    print(f"Muestras            : {meta['n_samples']}   fs vibracion: {meta['fs_hz']} Hz")
    print(f"Tasa real ACS       : {meta['fs_acs_hz']} Hz   (sostenido en el stream de 1 kHz)")
    print(f"Tasa real VFD        : {meta['fs_vfd_hz']} Hz")
    print(f"Keyphasor           : {len(key_t_s)} pulsos   RPM≈ {meta['rpm_keyphasor']}")
    print(f"VFD freq medida     : {meta['vfd_hz_medido']} Hz   (etiqueta dominio: {a.freq} Hz)")
    print(f"Metadatos           : fluido={a.fluid}, llenado={a.fill}%, etiqueta={a.label}")
    print(f"Canales del modelo  : {MODEL_CHANNELS}")
    if a.label == -1:
        print("AVISO: sesion SIN etiqueta (--label). Asignala antes de entrenar.")
    if vib_corrupta:
        print(f"AVISO: vibracion CORRUPTA por EMI ({sat_frac*100:.1f}% muestras saturadas). "
              "Sirve para probar el pipeline, NO para resultados. Blindar el cable del ADXL.")
    else:
        print(f"Vibracion          : OK ({sat_frac*100:.2f}% muestras saturadas)")


if __name__ == "__main__":
    main()
