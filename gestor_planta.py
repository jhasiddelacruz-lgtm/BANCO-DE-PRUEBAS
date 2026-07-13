#!/usr/bin/env python3

import argparse, json, os, queue, threading, time, sys
from collections import deque
from dataclasses import dataclass
import numpy as np


# ============================================================ CONFIGURACION
@dataclass
class Config:
    # --- serial ---
    port: str = "COM6"
    baud: int = 921600
    # --- adquisicion ---
    fs_nominal: int = 1000
    n_ventana: int = 2000
    # --- espectro ---
    f_max: float = 60.0 
    n_bins: int = 121 
    detrend: bool = False
    # Calibracion SOLO-DISPLAY del ADXL345 (no afecta al modelo, que usa cuentas crudas).
    # Full-resolution => 3.9 mg/LSB en cualquier rango: 0.0039 g * 9.80665 = 0.03825 m/s2/LSB.
    # Derivado del firmware (DATA_FORMAT full-res). Sirve para mostrar el espectro en mm/s.
    cuentas_a_ms2: float = 0.03825
    # --- mapeo de canales ---
    ch_idx: tuple = (1, 2, 3, 4, 7)
    idx_vfd_hz: int = 6
    idx_torque: int = 7
    # --- salud de señal ---
    lim_acc_raw: float = 16000.0
    # --- motor / VFD ---
    umbral_motor_hz: float = 5.0
    f_min: float = 0.0
    f_max_motor: float = 60.0
    polos_motor: int = 4               # Número de polos del motor (4 para 0.5 HP)
    # RPM derivada del VFD = velocidad SINCRONA (Hz*120/polos). El eje real gira mas
    # lento por deslizamiento. Pon aqui el % medido de tu motor con carga para obtener
    # una RPM de eje ESTIMADA; 0.0 = sin correccion (se reporta la sincrona).
    desliz_nominal_pct: float = 0.0
    # --- salud de señal (corriente / torque) ---
    # Saturacion del canal de corriente (cuentas crudas del ADC). 0 = deshabilitado.
    # Ponlo al fondo de escala real de tu ADS1115/ACS712 para detectar clipping.
    lim_corr_raw: float = 0.0
    # Debajo de esta desviacion estandar (con motor encendido) el canal se considera
    # "congelado" (sensor desconectado o stream detenido).
    std_min_corr: float = 1.0
    std_min_torque: float = 1e-6
    # --- estabilizacion ---
    estabilizacion_s: float = 10.0
    tol_setpoint_hz: float = 2.0
    key_timeout_s: float = 1.0
    # --- seguridad ---
    heartbeat_s: float = 0.5
    comm_timeout_s: float = 1.5
    # --- inferencia ---
    intervalo_infer_s: float = 1.0
    n_promediado: int = 15
    histeresis: float = 0.10
    umbral_decision: float = 0.5
    # --- grabacion ---
    dir_sesiones: str = "sesiones"
    dir_baselines: str = "baselines"
    # --- historial para graficas ---
    decimacion_plot: int = 10
    segundos_plot: float = 4.0
    # --- RPM (keyphasor - opcional) ---
    pulsos_por_vuelta: int = 1          # Solo si usas keyphasor
    rpm_intervalo_s: float = 1.0
    rpm_filtro_ventana: int = 3
    usar_keyphasor: bool = False        # False = usa VFD, True = usa sensor


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ============================================================ ESTADO COMPARTIDO
class EstadoPlanta:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.buffer = np.zeros((cfg.n_ventana, 5))
        self.idx = 0
        self.n = 0
        self.lleno = False
        self.t0 = time.time()
        self.t_ultimo_dato = 0.0
        n_plot = int(cfg.segundos_plot * cfg.fs_nominal / cfg.decimacion_plot)
        self.hist = deque(maxlen=n_plot)
        self._dec = 0
        self.vfd_hz = 0.0
        self.torque = 0.0
        self.rpm = 0.0
        self.rpm_1x_hz = 0.0          # frecuencia de rotación en Hz (RPM/60)
        self.rpm_fuente = "—"         # "VFD" (estimada) | "keyphasor" (medida) | "—"
        # tiempos de las ultimas muestras -> fs instantaneo (no promedio de por vida)
        self._t_muestras = deque(maxlen=cfg.fs_nominal)
        self.salud = {c: "--" for c in ("ax", "ay", "az", "corriente", "torque")}
        self.veredicto = None
        self.vfd_estado = "desconocido"
        self.freq_setpoint = 0.0
        self.comm_ok = False
        self.motor_bloqueado = False
        self.ultimo_comando = "—"
        # ---- Keyphasor (solo si se usa) ----
        self._pulse_count = 0
        self._last_pulse_us = None
        self._t_last_pulse = 0.0
        self._last_rpm_update = time.time()
        self._rpm_raw = 0.0
        self._rpm_hist_vfd = deque(maxlen=cfg.rpm_filtro_ventana)
        self._rpm_hist_key = deque(maxlen=cfg.rpm_filtro_ventana)
        # ---- Fases ----
        self.fase = "apagado"
        self.estab_restante = 0.0

    def push(self, ch5, vfd_hz, torque):
        with self.lock:
            self.buffer[self.idx] = ch5
            self.idx = (self.idx + 1) % self.cfg.n_ventana
            self.n += 1
            if self.n >= self.cfg.n_ventana:
                self.lleno = True
            self.vfd_hz = vfd_hz
            self.torque = torque
            ahora = time.time()
            self.t_ultimo_dato = ahora
            self._t_muestras.append(ahora)
            self.comm_ok = True
            self._dec += 1
            if self._dec >= self.cfg.decimacion_plot:
                self._dec = 0
                self.hist.append(list(ch5))
            # RPM desde el VFD SOLO si no se usa keyphasor (si no, el sensor manda).
            if not self.cfg.usar_keyphasor:
                self._actualizar_rpm_vfd()

    def push_key(self, t_us):
        # Solo se usa si `usar_keyphasor = True`
        if not self.cfg.usar_keyphasor:
            return
        with self.lock:
            ahora = time.time()
            self._pulse_count += 1
            # RPM por PERIODO entre pulsos consecutivos: preciso con 1 pulso/vuelta
            # (el conteo por ventana redondea feo con pocos pulsos, p.ej. 19.5 -> 20 Hz).
            # rpm = 60e6 / (dt_us * pulsos_por_vuelta).  t_us viene del firmware (micros).
            if self._last_pulse_us is not None:
                dt_us = t_us - self._last_pulse_us
                if dt_us > 0:                        # guarda contra wrap del contador de micros
                    if self.vfd_hz <= self.cfg.umbral_motor_hz:
                        rpm_inst = 0.0
                        self._rpm_hist_key.clear()
                    else:
                        rpm_inst = 60e6 / (dt_us * self.cfg.pulsos_por_vuelta)
                        self._rpm_hist_key.append(rpm_inst)
                    rpm_filtrada = np.mean(self._rpm_hist_key) if self._rpm_hist_key else rpm_inst
                    self._rpm_raw = rpm_inst
                    self.rpm = rpm_filtrada
                    self.rpm_1x_hz = self.rpm / 60.0
                    self.rpm_fuente = "keyphasor"    # velocidad MEDIDA del eje
            self._last_pulse_us = t_us
            self._t_last_pulse = ahora
            self._last_rpm_update = ahora

    def _actualizar_rpm_vfd(self):
        """RPM ESTIMADA a partir de la frecuencia del VFD (NO medida en el eje).

        RPM_sincrona = Hz*120/polos. El eje real gira mas lento por deslizamiento;
        se aplica `desliz_nominal_pct` si el usuario lo configuro (0 = sincrona)."""
        if self.vfd_hz > self.cfg.umbral_motor_hz:
            rpm_sinc = (self.vfd_hz * 120.0) / self.cfg.polos_motor
            rpm_calc = rpm_sinc * (1.0 - self.cfg.desliz_nominal_pct / 100.0)
            # Pequeño filtro para suavizar (si el VFD fluctúa)
            self._rpm_hist_vfd.append(rpm_calc)
            rpm_filtrada = np.mean(self._rpm_hist_vfd) if self._rpm_hist_vfd else rpm_calc
            self.rpm = rpm_filtrada
            self.rpm_1x_hz = self.rpm / 60.0
            self.rpm_fuente = "VFD"          # velocidad ESTIMADA (sincrona - deslizamiento)
        else:
            self.rpm = 0.0
            self.rpm_1x_hz = 0.0
            self.rpm_fuente = "—"
            self._rpm_hist_vfd.clear()

    def snapshot(self):
        with self.lock:
            if self.lleno:
                buf = np.concatenate([self.buffer[self.idx:], self.buffer[:self.idx]])
            else:
                buf = self.buffer[:self.n].copy()
            # fs INSTANTANEO: a partir de los tiempos de las ultimas muestras, no del
            # promedio de por vida (que enmascara stalls). Fallback al promedio global.
            if len(self._t_muestras) >= 2:
                span = self._t_muestras[-1] - self._t_muestras[0]
                fs = (len(self._t_muestras) - 1) / span if span > 1e-6 else 0.0
            else:
                fs = self.n / max(1e-6, time.time() - self.t0)
            rpm = self.rpm
            # Si se usa keyphasor y no hay pulso, forzar 0
            if self.cfg.usar_keyphasor:
                sin_key = (self._last_pulse_us is None) or (time.time() - self._t_last_pulse > self.cfg.key_timeout_s)
                if self.vfd_hz <= self.cfg.umbral_motor_hz or sin_key:
                    rpm = 0.0
            # Si no se usa keyphasor, la RPM ya viene del VFD
            return dict(
                buf=buf, lleno=self.lleno, n=self.n, fs=fs,
                hist=np.array(self.hist) if self.hist else np.zeros((0, 5)),
                vfd_hz=self.vfd_hz, torque=self.torque,
                rpm=rpm, rpm_1x_hz=rpm/60.0 if rpm > 0 else 0.0,
                rpm_fuente=self.rpm_fuente,
                salud=dict(self.salud), veredicto=self.veredicto,
                vfd_estado=self.vfd_estado, freq_setpoint=self.freq_setpoint,
                comm_ok=self.comm_ok, motor_bloqueado=self.motor_bloqueado,
                ultimo_comando=self.ultimo_comando,
                fase=self.fase, estab_restante=self.estab_restante)

    def set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)


# ============================================================ FUENTES DE DATOS
class FuenteSerial:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ser = None
        self._buf = ""
        self._abrir()

    def _abrir(self):
        try:
            import serial
            self.ser = serial.Serial(self.cfg.port, self.cfg.baud, timeout=1)
            time.sleep(2)
            self.ser.reset_input_buffer()
            self.ser.write(b"RAW_START\n")
            _log(f"Serial abierto en {self.cfg.port}")
        except Exception as e:
            self.ser = None
            _log(f"No se pudo abrir {self.cfg.port} ({e}); reintentando en segundo plano...")

    def lineas(self):
        while True:
            if self.ser is None:
                time.sleep(2)
                self._abrir()
                continue
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1).decode(errors="ignore")
                self._buf += chunk
                while "\n" in self._buf:
                    ln, self._buf = self._buf.split("\n", 1)
                    yield ln.strip()
            except Exception as e:
                _log(f"Serial caido ({e}); reintentando en 2 s...")
                self.ser = None
                time.sleep(2)
                self._abrir()

    def enviar(self, cmd):
        reps = 3 if any(k in cmd for k in ("VFD_RUN", "VFD_STOP", "VFD_FREQ")) else 1
        for i in range(reps):
            try:
                self.ser.write((cmd + "\n").encode())
                self.ser.flush()
            except Exception as e:
                _log(f"No se pudo enviar '{cmd}': {e}")
                return
            if reps > 1:
                time.sleep(0.05)

    def cerrar(self):
        try:
            self.ser.write(b"RAW_STOP\n")
            self.ser.close()
        except Exception:
            pass


class FuenteSimulada:
    def __init__(self, cfg, con_falla=False):
        self.cfg = cfg
        self.con_falla = con_falla
        self._run = False
        self._hz = 0.0

    def lineas(self):
        t_us = 0
        dt_us = 1000
        rng = np.random.default_rng(0)
        next_key = 0.0
        while True:
            for _ in range(50):
                t = t_us / 1e6
                hz = self._hz if self._run else 0.0
                rot = 2 * np.pi * 30 * t
                amp = (40 if self.con_falla else 8) if self._run else 1
                ax = amp * np.sin(rot) + rng.normal(0, 3)
                ay = 258 + amp * np.cos(rot) + rng.normal(0, 3)
                az = 44 + rng.normal(0, 3)
                acs = 12500 + (1300 if self._run else 0) * np.sin(2 * np.pi * 60 * t) + rng.normal(0, 30)
                sct = rng.normal(0, 50)
                torque = (0.19 if self._run else 0.0) + rng.normal(0, 0.005)
                yield f"{t_us},{ax:.0f},{ay:.0f},{az:.0f},{acs:.0f},{sct:.0f},{hz:.1f},{torque:.3f}"
                if self._run and t >= next_key:
                    yield f"KEY,{t_us}"
                    next_key = t + 1.0 / 30.0
                t_us += dt_us
            time.sleep(0.05)

    def enviar(self, cmd):
        if cmd == "PING":
            return
        if cmd == "VFD_RUN":
            self._run = True
        elif cmd == "VFD_STOP":
            self._run = False
        elif cmd.startswith("VFD_FREQ:"):
            try:
                self._hz = float(cmd.split(":", 1)[1])
            except ValueError:
                pass
        _log(f"[SIM] comando: {cmd}")

    def cerrar(self):
        pass


# ============================================================ GRABACION
class Grabador:
    def __init__(self, ruta_base, meta):
        self.f = open(ruta_base + ".csv", "w", buffering=1 << 16)
        self.fk = open(ruta_base + "_key.csv", "w", buffering=1 << 16)
        self.f.write(f"# session={meta.get('session','')} fluid={meta.get('fluid','')} "
                     f"freq={meta.get('freq','')} fill={meta.get('fill','')} label={meta.get('label','')}\n")
        self.f.write("t_us,ax_raw,ay_raw,az_raw,acs_raw,sct_raw,vfd_hz,vfd_torque_Nm\n")
        self.fk.write("t_us\n")
        self.n = 0
        self._q = queue.Queue()
        self._activo = True
        self._hilo = threading.Thread(target=self._escritor, daemon=True)
        self._hilo.start()

    def escribir(self, ln):
        self._q.put(ln)

    def _escritor(self):
        while self._activo or not self._q.empty():
            try:
                ln = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if ln.startswith("KEY"):
                    partes = ln.split(",", 1)
                    if len(partes) == 2:
                        self.fk.write(partes[1] + "\n")
                elif ln and not ln.startswith("#"):
                    self.f.write(ln + "\n")
                    self.n += 1
            except Exception as e:
                _log(f"Grabador: error escribiendo linea ({e}); se continua.")

    def cerrar(self):
        self._activo = False
        self._hilo.join(timeout=5.0)
        self.f.close()
        self.fk.close()


# ============================================================ ADQUISICION
class Adquisidor(threading.Thread):
    def __init__(self, estado, fuente, cfg):
        super().__init__(daemon=True)
        self.estado = estado
        self.fuente = fuente
        self.cfg = cfg
        self.activo = True
        self.grabador = None

    def run(self):
        for ln in self.fuente.lineas():
            if not self.activo:
                break
            try:
                if self.grabador is not None:
                    self.grabador.escribir(ln)
                if not ln or ln.startswith("#"):
                    continue
                if ln.startswith("KEY"):
                    try:
                        self.estado.push_key(int(ln.split(",")[1]))
                    except (ValueError, IndexError):
                        pass
                    continue
                p = ln.split(",")
                if len(p) >= 8:
                    try:
                        ch5 = [float(p[i]) for i in self.cfg.ch_idx]
                        self.estado.push(ch5, float(p[self.cfg.idx_vfd_hz]), float(p[self.cfg.idx_torque]))
                    except (ValueError, IndexError):
                        pass
            except Exception as e:
                _log(f"Adquisidor: error procesando linea ({e}); se continua leyendo.")


# ============================================================ SEGURIDAD
class Seguridad(threading.Thread):
    def __init__(self, gestor, cfg):
        super().__init__(daemon=True)
        self.g = gestor
        self.cfg = cfg
        self.activo = True

    def run(self):
        while self.activo:
            self.g.fuente.enviar("PING")
            est = self.g.estado
            sin_datos = time.time() - est.t_ultimo_dato
            if est.t_ultimo_dato > 0 and sin_datos > self.cfg.comm_timeout_s:
                if est.comm_ok:
                    _log("PERDIDA DE COMUNICACION -> ordenando PARO por seguridad")
                est.set(comm_ok=False)
                if est.vfd_estado == "corriendo":
                    self.g.apagar_motor(motivo="perdida de comunicacion")
            time.sleep(self.cfg.heartbeat_s)


# ============================================================ INFERENCIA
def ventana_fft(buf, cfg):
    # Defensivo: si el buffer aun no llena una ventana completa, no intentes multiplicar
    # por la ventana de Hann (romperia por broadcasting). Devuelve espectro nulo.
    if buf.shape[0] < cfg.n_ventana:
        return np.zeros((cfg.n_bins, 5), np.float32)
    seg = buf[-cfg.n_ventana:]
    hann = np.hanning(cfg.n_ventana)
    cg = hann.sum()
    fn = np.fft.rfftfreq(cfg.n_ventana, 1.0 / cfg.fs_nominal)
    fijo = np.linspace(0.0, cfg.f_max, cfg.n_bins)
    F = np.empty((cfg.n_bins, 5), np.float32)
    for c in range(5):
        x = seg[:, c] - seg[:, c].mean() if cfg.detrend else seg[:, c]
        mag = np.abs(np.fft.rfft(x * hann)) * (2.0 / cg)
        F[:, c] = np.interp(fijo, fn, mag)
    return F


class Inferencia:
    def __init__(self, cfg, model_path=None, norm_path=None, ood_path=None):
        self.cfg = cfg
        self.modelo = None
        self.mean = None
        self.std = None
        self.ood = None
        self.base_sana = None
        self.eps = 1e-6
        self._calib_buf = []
        self.hist = deque(maxlen=cfg.n_promediado)
        if model_path:
            self._cargar(model_path, norm_path, ood_path)

    def calibrar_inicio(self):
        self._calib_buf = []
        _log("CALIBRACION iniciada: graba el agitador SANO ~30 s en el fluido del dia.")

    def calibrar_push(self, buf):
        self._calib_buf.append(ventana_fft(buf, self.cfg))

    def calibrar_fin(self):
        if not self._calib_buf:
            _log("*** AVISO: no se recolectaron ventanas de calibracion; baseline NO aplicado. ***")
            return False
        self.base_sana = np.mean(np.stack(self._calib_buf, 0), axis=0).astype(np.float32)
        _log(f"CALIBRACION lista: linea base sana de {len(self._calib_buf)} ventanas.")
        self._calib_buf = []
        return True

    def guardar_base(self, ruta):
        if self.base_sana is not None:
            d = os.path.dirname(ruta)
            if d:
                os.makedirs(d, exist_ok=True)
            np.savez(ruta, base_sana=self.base_sana)
            _log(f"Linea base guardada en {ruta}")

    def cargar_base(self, ruta):
        if ruta and os.path.exists(ruta):
            self.base_sana = np.load(ruta)["base_sana"].astype(np.float32)
            _log(f"Linea base sana CARGADA de {os.path.basename(ruta)} (forma {self.base_sana.shape})")

    def _cargar(self, model_path, norm_path, ood_path):
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(model_path)) or ".")
        sys.path.insert(0, ".")
        from model import BioreactorDG
        nd = 2
        if norm_path and os.path.exists(norm_path):
            d = np.load(norm_path, allow_pickle=True)
            _log(f"--norm '{os.path.basename(norm_path)}' contiene: {list(d.files)}")
            for km, ks in (("mean", "std"), ("mu", "sigma"), ("X_mean", "X_std"),
                           ("media", "desv"), ("feat_mean", "feat_std")):
                if km in d.files and ks in d.files:
                    self.mean = d[km]
                    self.std = d[ks]
                    _log(f"Normalizacion CARGADA (claves '{km}'/'{ks}', forma {self.mean.shape})")
                    break
            if self.mean is None:
                _log("*** AVISO: no se hallo media/desv en --norm")
            if "meta" in d.files:
                nd = max(2, int(json.loads(str(d["meta"])).get("n_domains", 2)))
        else:
            _log("AVISO: no se paso --norm (o no existe) -> sin normalizacion.")
        self.modelo = BioreactorDG(in_ch=5, n_domains=nd)
        self.modelo.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.modelo.eval()
        if ood_path and os.path.exists(ood_path):
            o = np.load(ood_path)
            self.ood = (o["mu"], o["prec"], float(o["thr"]))
        _log(f"Modelo cargado (OOD {'activo' if self.ood else 'no'})")

    def predecir(self, buf):
        if self.modelo is None:
            return None
        import torch
        F = ventana_fft(buf, self.cfg)
        base_faltante = False
        if self.base_sana is not None:
            F = np.log((F + self.eps) / (self.base_sana + self.eps)).astype(np.float32)
        elif self.mean is not None:
            # El modelo esta normalizado pero no hay linea base sana cargada: el veredicto
            # puede no ser fiable. Se avisa en consola Y se marca para mostrarlo en la HMI.
            base_faltante = True
            _log("*** AVISO: modelo normalizado pero NO hay linea base sana en vivo.")
        if self.mean is not None:
            F = (F - self.mean) / self.std
        x = torch.tensor(F[None], dtype=torch.float32)
        with torch.no_grad():
            logit, _ = self.modelo(x, lambd=0.0)
            p = float(torch.sigmoid(logit)[0])
            z = self.modelo.extract_features(x).numpy()[0]
        self.hist.append(p)
        p_avg = float(np.mean(self.hist))
        u = self.cfg.umbral_decision
        margen = getattr(self.cfg, "histeresis", 0.10)
        prev = getattr(self, "_ultimo_desb", False)
        if p_avg >= u + margen:
            desb = True
        elif p_avg <= u - margen:
            desb = False
        else:
            desb = prev
        self._ultimo_desb = desb
        ood = False
        dist = None
        if self.ood is not None:
            mu, prec, thr = self.ood
            d = z - mu
            dist = float(np.sqrt(max(0.0, d @ prec @ d)))
            ood = dist > thr
        return dict(clase="FALLA" if desb else "SANO",
                    conf=p_avg if desb else 1 - p_avg,
                    p_desb=p_avg, ood=ood, dist=dist,
                    ood_activo=self.ood is not None,   # si False: sin guardia OOD
                    base_faltante=base_faltante)


# ============================================================ GESTOR
class GestorPlanta:
    def __init__(self, cfg, fuente, infer):
        self.cfg = cfg
        self.fuente = fuente
        self.estado = EstadoPlanta(cfg)
        self.infer = infer
        self.adq = Adquisidor(self.estado, fuente, cfg)
        self.seg = Seguridad(self, cfg)
        self._t_infer = 0.0
        self._t_estable0 = None

    def iniciar(self):
        self.adq.start()
        self.seg.start()
        _log("Gestor iniciado.")

    def calibrar_automatica(self, estab_s=30, grab_s=60, guardar="base_sana.npz", on_progreso=None):
        def _run():
            try:
                if self.infer is None or self.infer.modelo is None:
                    _log("CALIBRACION: no hay modelo cargado.")
                    if on_progreso:
                        on_progreso("Sin modelo", 0.0)
                    return
                for i in range(int(estab_s)):
                    if on_progreso:
                        on_progreso(f"Estabilizando… {int(estab_s)-i}s", i/(estab_s+grab_s))
                    time.sleep(1.0)
                self.infer.calibrar_inicio()
                t0 = time.time()
                n = 0
                while time.time() - t0 < grab_s:
                    est = self.estado.snapshot()
                    buf = est.get("buf")
                    if buf is not None and len(buf) >= self.cfg.n_ventana:
                        self.infer.calibrar_push(buf)
                        n += 1
                    frac = (estab_s + (time.time()-t0)) / (estab_s+grab_s)
                    if on_progreso:
                        on_progreso(f"Grabando sano… {int(grab_s-(time.time()-t0))}s  ({n} ventanas)", frac)
                    time.sleep(0.2)
                ok = self.infer.calibrar_fin()
                if ok:
                    self.infer.guardar_base(guardar)
                    if on_progreso:
                        on_progreso(f"✓ Calibrado ({n} ventanas). Base guardada.", 1.0)
                else:
                    if on_progreso:
                        on_progreso("✗ Calibracion sin datos. Reintenta.", 0.0)
            except Exception as e:
                _log(f"CALIBRACION error: {e}")
                if on_progreso:
                    on_progreso(f"Error: {e}", 0.0)
        threading.Thread(target=_run, daemon=True).start()

    def detener(self):
        self.apagar_motor(motivo="cierre del gestor")
        self.adq.activo = False
        self.seg.activo = False
        self.fuente.cerrar()
        _log("Gestor detenido.")

    def _clamp(self, hz):
        return max(self.cfg.f_min, min(self.cfg.f_max_motor, float(hz)))

    def encender_motor(self, hz, confirmado=False):
        if self.estado.motor_bloqueado:
            _log("RECHAZADO: paro de emergencia activo. Rearma primero (rearmar()).")
            return False
        if not confirmado:
            _log("RECHAZADO: arranque requiere confirmado=True.")
            return False
        if not self.estado.comm_ok:
            _log("RECHAZADO: sin comunicacion con el banco.")
            return False
        hz = self._clamp(hz)
        self.fuente.enviar(f"VFD_FREQ:{hz:.2f}")
        self.fuente.enviar("VFD_RUN")
        self.estado.set(vfd_estado="corriendo", freq_setpoint=hz, ultimo_comando=f"ENCENDER {hz:.1f}Hz")
        _log(f"Motor ENCENDIDO a {hz:.1f} Hz")
        return True

    def apagar_motor(self, motivo="manual"):
        self.fuente.enviar("VFD_STOP")
        self.estado.set(vfd_estado="detenido", ultimo_comando=f"APAGAR ({motivo})")
        _log(f"Motor APAGADO ({motivo})")

    def set_frecuencia(self, hz):
        if self.estado.vfd_estado != "corriendo":
            _log("Aviso: set_frecuencia con motor no corriendo.")
        hz = self._clamp(hz)
        self.fuente.enviar(f"VFD_FREQ:{hz:.2f}")
        self.estado.set(freq_setpoint=hz, ultimo_comando=f"FREQ {hz:.1f}Hz")

    def paro_emergencia(self):
        self.fuente.enviar("VFD_STOP")
        self.estado.set(vfd_estado="detenido", motor_bloqueado=True, ultimo_comando="PARO DE EMERGENCIA")
        _log("*** PARO DE EMERGENCIA *** (enclavado; usa rearmar() para volver)")

    def rearmar(self):
        self.estado.set(motor_bloqueado=False, ultimo_comando="REARME")
        _log("Sistema rearmado.")

    def grabar_sesion(self, nombre, meta):
        os.makedirs(self.cfg.dir_sesiones, exist_ok=True)
        ruta = os.path.join(self.cfg.dir_sesiones, nombre)
        self.adq.grabador = Grabador(ruta, meta)
        _log(f"Grabando sesion: {ruta}.csv")

    def detener_grabacion(self):
        if self.adq.grabador:
            n = self.adq.grabador.n
            self.adq.grabador.cerrar()
            self.adq.grabador = None
            _log(f"Grabacion detenida ({n} muestras).")

    def actualizar(self):
        snap = self.estado.snapshot()
        if snap["n"] > 0:
            self.estado.set(salud=self._salud(snap["buf"], snap["vfd_hz"]))
            motor_on = snap["vfd_hz"] > self.cfg.umbral_motor_hz
            setpoint = snap["freq_setpoint"]
            cerca = (motor_on and setpoint > self.cfg.umbral_motor_hz
                     and abs(snap["vfd_hz"] - setpoint) <= self.cfg.tol_setpoint_hz)

            if not motor_on:
                self.estado.set(veredicto=None, fase="apagado", estab_restante=0.0)
                self.infer.hist.clear()
                self._t_estable0 = None
            elif not cerca:
                self.estado.set(veredicto=None, fase="rampa",
                                estab_restante=self.cfg.estabilizacion_s)
                self.infer.hist.clear()
                self._t_estable0 = None
            else:
                if self._t_estable0 is None:
                    self._t_estable0 = time.time()
                    self.infer.hist.clear()
                restante = self.cfg.estabilizacion_s - (time.time() - self._t_estable0)
                if restante > 0:
                    self.estado.set(veredicto=None, fase="estabilizando",
                                    estab_restante=max(0.0, restante))
                else:
                    self.estado.set(fase="diagnostico", estab_restante=0.0)
                    if (self.infer.modelo and snap["lleno"]
                            and time.time() - self._t_infer > self.cfg.intervalo_infer_s):
                        self.estado.set(veredicto=self.infer.predecir(snap["buf"]))
                        self._t_infer = time.time()
        return self.estado.snapshot()

    def _salud(self, buf, vfd_hz):
        nombres = ("ax", "ay", "az", "corriente", "torque")
        motor_on = vfd_hz > self.cfg.umbral_motor_hz
        salud = {}
        for i, c in enumerate(nombres):
            x = buf[:, i]
            if c in ("ax", "ay", "az"):
                if np.any(np.abs(x) > self.cfg.lim_acc_raw):
                    salud[c] = "EMI"
                elif motor_on and np.std(x) < 1e-3:
                    salud[c] = "muerto"
                else:
                    salud[c] = "ok"
            elif c == "corriente":
                # Saturacion del ADC (solo si se configuro un fondo de escala).
                if self.cfg.lim_corr_raw > 0 and np.any(np.abs(x) >= self.cfg.lim_corr_raw):
                    salud[c] = "sat"
                # Canal congelado con el motor encendido -> sensor caido o stream detenido.
                elif motor_on and np.std(x) < self.cfg.std_min_corr:
                    salud[c] = "muerto"
                else:
                    salud[c] = "ok"
            else:  # torque (estimacion del VFD): detecta valor pegado/no actualizado.
                if motor_on and np.std(x) < self.cfg.std_min_torque:
                    salud[c] = "muerto"
                else:
                    salud[c] = "ok"
        return salud

    def get_estado(self):
        return self.estado.snapshot()


# ============================================================ VISTA POR CONSOLA
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--simular", action="store_true")
    ap.add_argument("--falla", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--norm")
    ap.add_argument("--ood")
    ap.add_argument("--segundos", type=float, default=0)
    ap.add_argument("--demo-arranque", action="store_true")
    # Opción para forzar el uso del keyphasor (por defecto usa VFD)
    ap.add_argument("--usar-keyphasor", action="store_true", help="Usar sensor inductivo en lugar del VFD")
    a = ap.parse_args()

    cfg = Config()
    if a.port:
        cfg.port = a.port
    cfg.baud = a.baud
    cfg.usar_keyphasor = a.usar_keyphasor

    if a.simular:
        fuente = FuenteSimulada(cfg, con_falla=a.falla)
        _log("MODO SIMULACION (sin hardware)")
    elif a.port:
        fuente = FuenteSerial(cfg)
    else:
        sys.exit("Indica --simular o --port COMx")

    infer = Inferencia(cfg, a.model, a.norm, a.ood)
    if not infer.modelo:
        _log("Sin modelo: monitoreo solamente (KPIs + salud de señal)")

    g = GestorPlanta(cfg, fuente, infer)
    g.iniciar()
    t_ini = time.time()
    arrancado = False
    try:
        while True:
            snap = g.actualizar()
            if a.simular and a.demo_arranque and not arrancado and time.time() - t_ini > 2:
                g.encender_motor(45.0, confirmado=True)
                arrancado = True
            vib = float(np.sqrt(np.mean((snap["buf"][:, 1] - snap["buf"][:, 1].mean()) ** 2))) if snap["n"] else 0
            corr = float(np.sqrt(np.mean(snap["buf"][:, 3] ** 2))) if snap["n"] else 0
            sal = " ".join(f"{c}:{snap['salud'][c]}" for c in ("ax", "ay", "az", "corriente", "torque"))
            v = snap["veredicto"]
            vtxt = (f"{v['clase']} {v['conf']*100:.0f}%" + (" [OOD!]" if v.get("ood") else "")) if v \
                   else ("llenando" if not snap["lleno"] else "—")
            comm = "OK" if snap["comm_ok"] else "CAIDA"
            print(f"\rfs={snap['fs']:4.0f} RPM={snap['rpm']:5.0f} VFD={snap['vfd_hz']:4.1f}Hz[{snap['vfd_estado'][:4]}] "
                  f"vib={vib:6.1f} I={corr:6.0f} M={snap['torque']:.3f} | {sal} | comm:{comm} | {vtxt}   ",
                  end="", flush=True)
            if a.segundos and time.time() - t_ini > a.segundos:
                break
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        g.detener()
        print()


if __name__ == "__main__":
    main()