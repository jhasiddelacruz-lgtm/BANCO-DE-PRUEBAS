#!/usr/bin/env python3

import argparse, serial, time, sys
import numpy as np

HEADER = "t_us,ax_raw,ay_raw,az_raw,acs_raw,sct_raw,vfd_hz,vfd_torque_Nm\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="COM5, /dev/ttyUSB0, etc.")
    ap.add_argument("--out", default="sesion", help="nombre base de salida")
    ap.add_argument("--seconds", type=float, default=None, help="duracion (modo RAW_START)")
    ap.add_argument("--command", default="RAW_START", choices=["RAW_START", "SESSION"])
    ap.add_argument("--baud", type=int, default=921600)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(0.5)
    ser.reset_input_buffer()

    raw_path = args.out + ".csv"
    key_path = args.out + "_key.csv"
    raw_f = open(raw_path, "w"); raw_f.write(HEADER)
    key_f = open(key_path, "w"); key_f.write("t_us\n")

    print(f"INFO: enviando comando {args.command}...")
    ser.write((args.command + "\n").encode())

    t_start = time.time()
    n_raw = 0; n_key = 0
    stopped = False
    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                if args.seconds and (time.time() - t_start) > args.seconds + 3:
                    break
                continue

            if line[0].isdigit():                 # fila de datos
                raw_f.write(line + "\n"); n_raw += 1
            elif line.startswith("KEY,"):          # pulso keyphasor
                key_f.write(line[4:] + "\n"); n_key += 1
            else:                                  # INFO, / #...  metadatos
                print("  " + line)
                if "Sesion completa" in line:      # SESSION termino solo
                    stopped = True; break

            # corte por tiempo en modo RAW_START
            if args.seconds and not stopped and (time.time() - t_start) >= args.seconds:
                ser.write(b"RAW_STOP\n")
                time.sleep(0.3)
                while ser.in_waiting:
                    rest = ser.readline().decode(errors="ignore").strip()
                    if rest and rest[0].isdigit(): raw_f.write(rest + "\n"); n_raw += 1
                    elif rest.startswith("KEY,"):  key_f.write(rest[4:] + "\n"); n_key += 1
                break

    except KeyboardInterrupt:
        print("\nINFO: detenido por usuario, enviando RAW_STOP...")
        ser.write(b"RAW_STOP\n"); time.sleep(0.3)
    finally:
        raw_f.close(); key_f.close(); ser.close()

    reportar(raw_path, key_path, n_raw, n_key)

def reportar(raw_path, key_path, n_raw, n_key):
    print("\n" + "=" * 50)
    print(f"Filas de datos : {n_raw}  -> {raw_path}")
    print(f"Pulsos keyphasor: {n_key}  -> {key_path}")
    if n_raw < 2:
        print("ADVERTENCIA: muy pocos datos. Revisa firmware/cableado.")
        return

    data = np.genfromtxt(raw_path, delimiter=",", skip_header=1)
    t = data[:, 0] / 1e6
    dur = t[-1] - t[0]
    fs = n_raw / dur if dur > 0 else 0
    dt_ms = np.diff(t) * 1000.0
    gaps = int(np.sum(dt_ms > 2.0))

    print(f"Duracion       : {dur:.2f} s")
    print(f"Tasa media     : {fs:.1f} Hz (objetivo 1000)")
    print(f"Jitter (std dt): {np.std(dt_ms):.3f} ms")
    print(f"Huecos >2ms    : {gaps}")

    if n_key >= 2:
        kt = np.genfromtxt(key_path, skip_header=1) / 1e6
        rpm = 60.0 / np.diff(kt)
        print(f"RPM keyphasor  : {np.mean(rpm):.0f} (min {np.min(rpm):.0f}, max {np.max(rpm):.0f})")
    print("=" * 50)

if __name__ == "__main__":
    main()
