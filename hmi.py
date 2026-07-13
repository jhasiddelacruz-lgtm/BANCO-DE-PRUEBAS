
import argparse, base64, glob, json, os, threading, time, sys
import numpy as np
import plotly.graph_objects as go

from gestor_planta import (Config, GestorPlanta, FuenteSerial,
                           Inferencia, ventana_fft)

COL = dict(ok="#00C896", warn="#FFB84D", alert="#FF5757", off="#5A6270",
           bg="#0E1117", panel="#1A1F2B", txt="#E6E9EF")
CANALES = ["ax", "ay", "az", "corriente", "torque"]
# Liquidos disponibles para baseline: (etiqueta visible, nombre de archivo sin extension).
# Para agregar/quitar un liquido, edita SOLO esta lista.
LIQUIDOS = [("CMC 50", "cmc_50"), ("CMC 75", "cmc_75"), ("CMC 100", "cmc_100"),
            ("CMC 125", "cmc_125"), ("CMC 150", "cmc_150")]


# ============================================================ logica pura
def _notch_display(x, fs, freqs, Q=30.0):
    """Elimina frecuencias puntuales (ruido de red 60/120 Hz) SOLO para el grafico.
    NO se usa en la cadena del modelo (predecir usa la senal completa). Usa scipy si
    esta; si no, biquad numpy puro (RBJ), fase cero (ida y vuelta)."""
    x = np.asarray(x, float)
    if not freqs:
        return x
    try:
        from scipy.signal import iirnotch, filtfilt
        y = x.copy()
        for f0 in freqs:
            if 0 < f0 < fs / 2:
                b, a = iirnotch(f0, Q, fs)
                y = filtfilt(b, a, y)
        return y
    except Exception:
        def biquad(f0):
            w0 = 2 * np.pi * f0 / fs
            al = np.sin(w0) / (2 * Q); cw = np.cos(w0)
            b = np.array([1.0, -2 * cw, 1.0])
            a = np.array([1 + al, -2 * cw, 1 - al])
            return b / a[0], a / a[0]

        def aplica(sig, b, a):
            y = np.empty_like(sig); z1 = z2 = 0.0
            for i in range(len(sig)):
                xn = sig[i]; yn = b[0] * xn + z1
                z1 = b[1] * xn - a[1] * yn + z2
                z2 = b[2] * xn - a[2] * yn
                y[i] = yn
            return y

        y = x.copy()
        for f0 in freqs:
            if 0 < f0 < fs / 2:
                b, a = biquad(f0)
                y = aplica(y, b, a)
                y = aplica(y[::-1], b, a)[::-1]
        return y


# Frecuencias de red a eliminar SOLO en el grafico del espectro (no afecta al modelo).
NOTCH_DISPLAY = [60.0, 120.0]

# Rango del eje Y del espectro: se auto-ajusta a los picos reales en pantalla (con
# margen), con un piso minimo para que nunca se vea vacio. Como el notch quita el
# ruido de 60 Hz, ningun pico falso dispara la escala.
# Auto-escala del eje Y: se ajusta al pico real en pantalla (con margen). En VELOCIDAD
# los valores son fracciones (mucho menores que en aceleracion), asi que el piso es
# diminuto: solo evita escala cero con el motor apagado. La escala real la fija el pico.
YMAX_PISO = 0.05        # piso minimo (motor apagado); la escala util la da el pico
YMAX_MARGEN = 1.25      # 25% de aire por encima del pico mas alto


def _notch_espectro_fijo(fr, mag, freqs=NOTCH_DISPLAY, ancho=1.5):
    """Suprime los bins cercanos a las frecuencias de red en un espectro ya calculado
    (malla fija), interpolando desde los bordes. Para que el FANTASMA sano coincida
    visualmente con el vivo (que va filtrado por notch temporal). Solo visual."""
    mag = np.asarray(mag, float).copy()
    fr = np.asarray(fr, float)
    for f0 in freqs:
        band = (fr >= f0 - ancho) & (fr <= f0 + ancho)
        if not band.any():
            continue
        idx = np.where(band)[0]
        lo, hi = idx[0] - 1, idx[-1] + 1
        if lo >= 0 and hi < len(mag):
            mag[idx] = np.interp(fr[idx], [fr[lo], fr[hi]], [mag[lo], mag[hi]])
        else:
            mag[idx] = 0.0
    return mag


def espectro_display(buf, fs=1000, fmax=60.0, canal=2, fmin=10.0, n_ventana=2000, cal_ms2=None):
    # canal: 0=ax, 1=ay, 2=az, 3=corriente, 4=torque. Por defecto 2=az (radial al desbalance).
    # SOLO VISUALIZACION: convierte aceleracion -> VELOCIDAD (integra dividiendo cada bin
    # por 2*pi*f). Resalta el 1x/2x (bajas frecuencias) y atenua el ruido agudo.
    # Si cal_ms2 se pasa (m/s2 por cuenta del ADXL), la salida esta en mm/s RMS, comparable
    # con ISO 10816 y con apps tipo Resonance. Si es None, queda en velocidad PROPORCIONAL
    # (unidades arbitrarias). El modelo NO usa esto (sigue en aceleracion, cuentas crudas).
    # fmin=10 Hz: no se dibuja por debajo (la integracion divide por 2*pi*f, amplificando
    # ~10-60x mas las frecuencias bajas; el ruido de banda baja del ADXL y las derivas lentas
    # se inflan en un "lomo" que ensucia el borde izquierdo). No hay firma mecanica bajo 10 Hz
    # (el 1x esta en ~20 Hz), asi que recortar ahi limpia la vista sin perder diagnostico.
    if buf.shape[0] < 256:
        return np.array([0.0]), np.array([0.0])
    x = buf[-n_ventana:, canal]; x = x - x.mean()
    x = _notch_display(x, fs, NOTCH_DISPLAY)     # quita ruido de red 60/120 Hz (solo visual)
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) * (2.0 / w.sum())   # amplitud de ACELERACION [cuentas, pico]
    fr = np.fft.rfftfreq(len(x), 1.0 / fs)
    # --- integracion a VELOCIDAD: V(f) = A(f) / (2*pi*f); el bin f=0 se anula ---
    vel = np.zeros_like(mag)
    nz = fr > 0
    vel[nz] = mag[nz] / (2.0 * np.pi * fr[nz])
    if cal_ms2 is not None:
        # cuentas(pico) -> m/s2(pico) -> m/s(pico) [ya integrado] -> mm/s(pico) -> mm/s(RMS)
        vel = vel * cal_ms2 * 1000.0 / np.sqrt(2.0)
    m = (fr >= fmin) & (fr <= fmax)              # dibuja de fmin a fmax (evita el lomo de baja frecuencia)
    return fr[m], vel[m]


def color_salud(e):
    return {"ok": COL["ok"], "EMI": COL["alert"], "muerto": COL["warn"],
            "sat": COL["alert"]}.get(e, COL["off"])


def cargar_csv_sesion(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("t_us"):
                continue
            p = ln.split(",")
            if len(p) >= 8:
                try:
                    rows.append([float(p[i]) for i in (1, 2, 3, 4, 7)])
                except ValueError:
                    pass
    return np.array(rows) if rows else np.zeros((0, 5))


def predecir_oneshot(infer, buf, cfg):
    if infer.modelo is None or buf.shape[0] < cfg.n_ventana:
        return None
    import torch
    F = ventana_fft(buf, cfg)
    # MISMA cadena que la inferencia en vivo (Inferencia.predecir): si hay linea base
    # sana, se aplica la log-razon antes de normalizar. Sin esto, el diagnostico offline
    # alimentaria features en un espacio distinto al de entrenamiento -> veredicto erroneo.
    if infer.base_sana is not None:
        F = np.log((F + infer.eps) / (infer.base_sana + infer.eps)).astype(np.float32)
    if infer.mean is not None:
        F = (F - infer.mean) / infer.std
    with torch.no_grad():
        logit, _ = infer.modelo(torch.tensor(F[None], dtype=torch.float32), lambd=0.0)
        return float(torch.sigmoid(logit)[0])


def img_uri(path):
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


def _fmt(x, nd=3):
    """Formatea una metrica: 3 decimales; None/NaN/ausente -> '—'."""
    import math
    if x is None:
        return "—"
    try:
        xf = float(x)
        return "—" if math.isnan(xf) else f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# ============================================================ gestor
def construir_gestor(a):
    # NiceGUI puede re-ejecutar este script (p. ej. al renderizar la pagina). Sin esta
    # guarda, main() correria de nuevo, intentaria abrir el COM otra vez (-> "Acceso
    # denegado", porque la 1a ejecucion ya lo tiene) y se caeria la HMI. El gestor se
    # guarda en el modulo gestor_planta, que SI persiste entre re-ejecuciones del proceso.
    import gestor_planta as _gp
    cache = getattr(_gp, "_HMI_GESTOR", None)
    if cache is not None:
        return cache
    cfg = Config()
    if a.port:
        cfg.port = a.port
    if getattr(a, "umbral", None) is not None:
        cfg.umbral_decision = a.umbral
    if getattr(a, "promedio", None) is not None:
        cfg.n_promediado = a.promedio
    if getattr(a, "histeresis", None) is not None:
        cfg.histeresis = a.histeresis
    # Keyphasor: fuente de velocidad MEDIDA del eje (para las marcas 1x/2x/3x del espectro).
    # Con esto, la marca 1x cae sobre el pico real (~19.5 Hz) y no sobre el sincrono del VFD.
    cfg.usar_keyphasor = bool(getattr(a, "usar_keyphasor", False))
    if getattr(a, "ppr", None) is not None:
        cfg.pulsos_por_vuelta = a.ppr
    if not a.port:
        sys.exit("Indica el puerto del banco: --port COMx (la HMI es solo para demo en vivo).")
    fuente = FuenteSerial(cfg)
    infer = Inferencia(cfg, a.model, a.norm, a.ood)
    if getattr(a, "base", None):
        infer.cargar_base(a.base)
    g = GestorPlanta(cfg, fuente, infer)
    g.iniciar()
    def _bg():
        while True:
            g.actualizar(); time.sleep(0.25)
    threading.Thread(target=_bg, daemon=True).start()
    _gp._HMI_GESTOR = (cfg, g)
    return cfg, g


# ============================================================ PANTALLA 1: OPERACION
def pantalla_operacion(g, cfg):
    from nicegui import ui
    w = {}
    # Banner grande de estado (lo mas visible en un proyector).
    w["banner"] = ui.label("—").classes(
        "w-full text-center text-3xl font-bold py-4 rounded-lg").style(
        f"background:{COL['off']};color:#000")
    with ui.row().classes("w-full items-center justify-end gap-3"):
        w["comm"] = ui.label("COMM —").classes("px-3 py-1 rounded text-sm font-bold")
        w["vfd"] = ui.label("VFD —").classes("px-3 py-1 rounded text-sm font-bold")

    with ui.row().classes("w-full gap-4 no-wrap"):
        with ui.card().classes("w-1/3").style(f"background:{COL['panel']}"):
            ui.label("Control del motor").classes("text-lg font-bold").style(f"color:{COL['txt']}")
            w["slider"] = ui.slider(min=0, max=int(cfg.f_max_motor), value=30, step=1)
            w["slbl"] = ui.label("30 Hz").style(f"color:{COL['txt']}")
            w["slider"].on("update:model-value", lambda e: w["slbl"].set_text(f"{int(w['slider'].value)} Hz"))

            async def on_on():
                d = ui.dialog()
                with d, ui.card():
                    ui.label(f"¿Encender el motor a {int(w['slider'].value)} Hz?")
                    ui.label("Asegurate de que el banco este despejado.").classes("text-sm").style("color:#aaa")
                    with ui.row():
                        ui.button("Cancelar", on_click=lambda: d.submit(False)).props("flat")
                        ui.button("ENCENDER", on_click=lambda: d.submit(True)).style(f"background:{COL['ok']}")
                res = await d
                if res:
                    ok = g.encender_motor(float(w["slider"].value), confirmado=True)
                    ui.notify("Motor encendido" if ok else "Arranque rechazado", type="positive" if ok else "negative")

            def apagar_directo():
                g.apagar_motor()
                ui.notify("Motor apagado", type="warning")

            with ui.row().classes("w-full gap-2"):
                ui.button("▶ ENCENDER", on_click=on_on).style(f"background:{COL['ok']};color:#000")
                ui.button("⏹ APAGAR", on_click=apagar_directo).style(f"background:{COL['off']}")
            ui.button("⛔ PARO DE EMERGENCIA",
                      on_click=lambda: (g.paro_emergencia(), ui.notify("PARO DE EMERGENCIA", type="negative"))
                      ).classes("w-full text-lg font-bold").style(f"background:{COL['alert']};color:#fff")
            ui.button("Rearmar", on_click=lambda: (g.rearmar(), ui.notify("Rearmado"))).props("flat")
            w["setp"] = ui.label("Setpoint: 0.0 Hz").classes("text-sm").style("color:#aaa")

        with ui.card().classes("w-2/3").style(f"background:{COL['panel']}"):
            ui.label("Diagnostico").classes("text-lg font-bold").style(f"color:{COL['txt']}")
            w["ver"] = ui.label("—").classes("text-4xl font-bold").style(f"color:{COL['txt']}")
            w["ood"] = ui.label("").classes("text-sm font-bold")
            w["base_activa"] = None

    with ui.row().classes("w-full gap-4 no-wrap"):
        for tit, key in [("RPM (est.)", "rpm"), ("VFD (Hz)", "hz"), ("Vibracion RMS (u.a.)", "vib"),
                         ("Corriente RMS (ADC)", "corr"), ("Torque (Nm, est. VFD)", "torque")]:
            with ui.card().classes("flex-1 items-center").style(f"background:{COL['panel']}"):
                ui.label(tit).classes("text-xs").style("color:#888")
                w[f"k_{key}"] = ui.label("0").classes("text-2xl font-bold").style(f"color:{COL['ok']}")

    with ui.row().classes("w-full gap-2 items-center"):
        ui.label("Salud de señal:").classes("text-sm").style("color:#888")
        for c in CANALES:
            w[f"s_{c}"] = ui.label(f"{c}: —").classes("px-2 py-1 rounded text-xs font-bold")
        ui.space()
        # Leyenda de ejes (z/x/y) al estilo Resonance; el espectro muestra los tres.
        ui.label("Ejes:").classes("text-sm").style("color:#888")
        for _lbl, _col in [("z", "#3B9EFF"), ("x", COL["alert"]), ("y", COL["warn"])]:
            ui.label(f"● {_lbl}").classes("text-sm font-bold").style(f"color:{_col}")

    with ui.row().classes("w-full gap-4 no-wrap"):
        fw = go.Figure()
        for nm in ("ax", "ay", "az"):
            fw.add_scatter(y=[], mode="lines", name=nm)
        fw.update_layout(template="plotly_dark", height=280, margin=dict(l=40, r=10, t=30, b=30),
                         title="Forma de onda (vibracion)", paper_bgcolor=COL["bg"], plot_bgcolor=COL["panel"])
        w["fw"] = fw; w["pw"] = ui.plotly(fw).classes("w-1/2")
        fs = go.Figure()
        # 3 trazas superpuestas al estilo Resonance: z (az), x (ax), y (ay).
        # Orden e indices FIJOS: data[0]=z(canal2), data[1]=x(canal0), data[2]=y(canal1).
        EJES_SPEC = [("z", 2, "#3B9EFF"), ("x", 0, COL["alert"]), ("y", 1, COL["warn"])]
        w["ejes_spec"] = EJES_SPEC
        for nm, _canal, col in EJES_SPEC:
            r, g_, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
            fs.add_scatter(x=[], y=[], mode="lines", name=nm,
                           fill="tozeroy", fillcolor=f"rgba({r},{g_},{b},0.15)",
                           line=dict(color=col, width=2))
        fs.update_layout(template="plotly_dark", height=280, margin=dict(l=40, r=10, t=30, b=30),
                         title="Espectro de VELOCIDAD (z/x/y) 0–60 Hz · mm/s RMS · marcas 1x/2x/3x",
                         paper_bgcolor=COL["bg"], plot_bgcolor=COL["panel"],
                         showlegend=True, legend=dict(x=0.85, y=0.98, bgcolor="rgba(0,0,0,0)"),
                         xaxis_title="Frecuencia [Hz]", yaxis_title="Velocidad [mm/s RMS]",
                         yaxis=dict(range=[0, YMAX_PISO], fixedrange=True))
        w["fs"] = fs; w["ps"] = ui.plotly(fs).classes("w-1/2")

    # Frecuencia dominante por eje (estilo Resonance).
    with ui.row().classes("w-full justify-around"):
        ui.label("Frecuencia dominante:").classes("text-sm self-center").style("color:#888")
        for nm, _canal, col in [("z", 2, "#3B9EFF"), ("x", 0, COL["alert"]), ("y", 1, COL["warn"])]:
            with ui.column().classes("items-center"):
                w[f"dom_{nm}"] = ui.label("—").classes("text-2xl font-bold").style(f"color:{col}")
                ui.label(nm).classes("text-xs").style("color:#888")

    with ui.card().classes("w-full").style(f"background:{COL['panel']}"):
        ui.label("Calibracion de linea base (motor SANO, estable)").classes("text-sm font-bold").style(f"color:{COL['txt']}")

        def usar_baseline(lbl, fn):
            ruta = os.path.join(cfg.dir_baselines, fn + ".npz")
            if not os.path.exists(ruta):
                ui.notify(f"No hay baseline guardada para {lbl}. Calíbrala primero.", type="negative")
                return
            g.infer.cargar_base(ruta)
            w["base_activa"] = lbl
            ui.notify(f"Baseline {lbl} cargada", type="positive")

        def calibrar():
            fn = w["cal_liq"].value
            lbl = dict((f, l) for l, f in LIQUIDOS).get(fn, fn)
            ruta = os.path.join(cfg.dir_baselines, fn + ".npz")
            ui.notify(f"Calibrando {lbl}: 30 s estabilidad + 60 s grabacion. Manten el motor SANO.", type="positive")
            def prog(txt, frac):
                w["cal_lbl"].set_text(txt)
                w["cal_bar"].set_value(frac)
            g.calibrar_automatica(estab_s=30, grab_s=60, guardar=ruta, on_progreso=prog)
            w["base_activa"] = lbl

        with ui.row().classes("items-center gap-3"):
            w["cal_liq"] = ui.select({fn: lbl for lbl, fn in LIQUIDOS}, value=LIQUIDOS[2][1],
                                     label="Líquido a calibrar").props("dense").classes("w-40")
            ui.button("⚙ CALIBRAR (30s + 60s)", on_click=calibrar).style(f"background:{COL['ok']};color:#000")
            # Boton nuevo: abre el menu de los 5 liquidos para USAR una baseline ya guardada.
            with ui.button("📂 USAR BASELINE").style(f"background:{COL['warn']};color:#000"):
                with ui.menu():
                    for lbl, fn in LIQUIDOS:
                        ui.menu_item(lbl, on_click=lambda l=lbl, f=fn: usar_baseline(l, f))
            w["cal_bar"] = ui.linear_progress(value=0.0, show_value=False).classes("flex-1")
            w["cal_lbl"] = ui.label("Sin calibrar").classes("text-sm").style("color:#aaa")
        ui.label("Calibrar: elige el líquido y pulsa CALIBRAR con el agitador SANO y estable (~90 s); "
                 "se guarda en baselines/. Usar baseline: carga una ya guardada para correr en vivo "
                 "sin recalibrar.").classes("text-xs").style("color:#777")

    with ui.card().classes("w-full").style(f"background:{COL['panel']}"):
        ui.label("Grabar sesion (para el pipeline)").classes("text-sm font-bold").style(f"color:{COL['txt']}")
        with ui.row().classes("items-end gap-3"):
            w["gs"] = ui.input("Sesion", value="S01").props("dense")
            w["gf"] = ui.input("Fluido", value="agua").props("dense")
            w["gq"] = ui.number("Freq", value=45).props("dense").classes("w-20")
            w["gl"] = ui.select({1: "desbalance", 0: "sano"}, value=0, label="Estado").props("dense").classes("w-32")
            def grab():
                nom = f"{w['gf'].value}_{int(w['gq'].value)}hz_{w['gs'].value}"
                g.grabar_sesion(nom, dict(session=w["gs"].value, fluid=w["gf"].value,
                                          freq=w["gq"].value, fill=80, label=w["gl"].value))
                ui.notify(f"Grabando {nom}")
            ui.button("● Grabar", on_click=grab).style(f"background:{COL['alert']};color:#fff")
            ui.button("■ Detener", on_click=lambda: (g.detener_grabacion(), ui.notify("Grabacion detenida"))).props("flat")

    def refrescar():
        s = g.get_estado()
        ok = s["comm_ok"]
        w["comm"].set_text("COMM OK" if ok else "COMM CAIDA")
        w["comm"].style(f"background:{COL['ok'] if ok else COL['alert']};color:#000")
        ve = s["vfd_estado"]
        w["vfd"].set_text(f"VFD {ve}"); w["vfd"].style(f"background:{COL['ok'] if ve=='corriendo' else COL['off']};color:#000")
        w["setp"].set_text(f"Setpoint: {s['freq_setpoint']:.1f} Hz" + ("  · BLOQUEADO" if s["motor_bloqueado"] else ""))
        buf = s["buf"]; n = s["n"]
        _ev = 2  # az: eje radial al desbalance (KPI de vibracion RMS)
        vib = float(np.sqrt(np.mean((buf[:, _ev] - buf[:, _ev].mean()) ** 2))) if n else 0.0
        corr = float(np.sqrt(np.mean(buf[:, 3] ** 2))) if n else 0.0
        _fuente_rpm = {"keyphasor": "med", "VFD": "est"}.get(s.get("rpm_fuente", "—"), "")
        w["k_rpm"].set_text(f"{s['rpm']:.0f}" + (f" ({_fuente_rpm})" if _fuente_rpm else ""))
        w["k_hz"].set_text(f"{s['vfd_hz']:.1f}")
        w["k_vib"].set_text(f"{vib:.1f}"); w["k_corr"].set_text(f"{corr:.0f}"); w["k_torque"].set_text(f"{s['torque']:.3f}")
        for c in CANALES:
            e = s["salud"][c]; w[f"s_{c}"].set_text(f"{c}: {e}"); w[f"s_{c}"].style(f"background:{color_salud(e)};color:#000")
        v = s["veredicto"]
        fase = s.get("fase", "apagado")
        if v:
            dz = v["clase"] == "FALLA"
            estado_txt = "DESBALANCE DETECTADO" if dz else "OPERACIÓN NOMINAL"
            estado_col = COL["alert"] if dz else COL["ok"]
            # Avisos que restan confianza al veredicto (antes solo iban a consola):
            avisos = []
            if v.get("base_faltante"):
                avisos.append("⚠ SIN LÍNEA BASE — veredicto no fiable")
            if v.get("ood") and v.get("ood_activo"):
                avisos.append("⚠ FUERA DE DOMINIO — recalibrar")
            elif not v.get("ood_activo", True):
                avisos.append("· sin guardia OOD")
            w["ood"].set_text("   ".join(avisos))
            w["ood"].style(f"color:{COL['warn']}")
        else:
            w["ood"].set_text("")
            if fase == "estabilizando":
                estado_txt = f"Estabilizando… {s.get('estab_restante', 0):.0f}s"
                estado_col = COL["warn"]
            elif fase == "rampa":
                estado_txt = "Esperando setpoint…"
                estado_col = COL["warn"]
            elif s["vfd_hz"] <= cfg.umbral_motor_hz:
                estado_txt = "MOTOR APAGADO"
                estado_col = COL["off"]
            elif not s["lleno"]:
                estado_txt = "Llenando buffer…"
                estado_col = COL["off"]
            else:
                estado_txt = "Sin modelo"
                estado_col = COL["off"]
        w["ver"].set_text(estado_txt); w["ver"].style(f"color:{estado_col}")
        w["banner"].set_text(estado_txt); w["banner"].style(f"background:{estado_col};color:#000")
        hist = s["hist"]
        if hist.shape[0] > 1:
            for i in range(3):
                w["fw"].data[i].y = hist[:, i]
            w["pw"].update()
        fs_real = s.get("fs", 1000)
        pico_global = 0.0
        for idx, (nm, canal, _col) in enumerate(w["ejes_spec"]):
            fr, mag = espectro_display(buf, fs=fs_real, canal=canal,
                                       n_ventana=cfg.n_ventana, cal_ms2=cfg.cuentas_a_ms2)
            w["fs"].data[idx].x = fr; w["fs"].data[idx].y = mag
            # Frecuencia dominante del eje = pico del espectro (ya recortado a >=10 Hz por
            # espectro_display, asi que ignora el lomo de baja frecuencia y apunta al 1x real).
            if len(fr) and len(mag):
                f_dom = float(fr[int(np.argmax(mag))])
                pico_eje = float(mag.max())
                w[f"dom_{nm}"].set_text(f"{f_dom:.1f}")
                pico_global = max(pico_global, pico_eje)
            else:
                w[f"dom_{nm}"].set_text("—")
        # --- techo del eje Y auto-ajustado al pico mas alto de los 3 ejes ---
        ymax = max(YMAX_PISO, pico_global * YMAX_MARGEN)
        # El eje Y se autoajusta; mostramos el pico ABSOLUTO (mm/s RMS) en el titulo para
        # que la gravedad real sea visible aunque la escala cambie.
        w["fs"].layout.title = (f"Espectro de VELOCIDAD (z/x/y) 0–60 Hz · "
                                f"pico={pico_global:.2f} mm/s RMS · marcas 1x/2x/3x")
        w["fs"].layout.yaxis.range = [0, ymax]   # escala ajustada a los picos (add_vline no la reajusta)
        w["fs"].layout.shapes = []
        w["fs"].layout.annotations = []
        f1 = s.get("rpm_1x_hz") or (s["rpm"] / 60.0)   # 1x real del keyphasor (snapshot); fallback rpm/60
        if f1 > 1.0:
            for h, lab, c in [(1, "1x", COL["alert"]), (2, "2x", COL["warn"]), (3, "3x", COL["ok"])]:
                fh = h * f1
                if fh <= 60:
                    w["fs"].add_vline(x=fh, line=dict(color=c, dash="dash"),
                                      annotation_text=lab, annotation_position="top")
        w["ps"].update()

    ui.timer(0.2, refrescar)


# ============================================================ PANTALLA 2: ANALISIS DE SESION
def pantalla_analisis(g, cfg):
    from nicegui import ui
    cont = ui.column().classes("w-full")

    def listar():
        d = cfg.dir_sesiones
        return sorted(glob.glob(os.path.join(d, "*.csv"))) if os.path.isdir(d) else []

    sel = ui.select(listar() or ["(sin sesiones grabadas)"], label="Sesion grabada").classes("w-96")
    ui.button("↻ Actualizar lista", on_click=lambda: sel.set_options(listar() or ["(sin sesiones)"])).props("flat")
    salida = ui.column().classes("w-full")

    def diagnosticar():
        salida.clear()
        path = sel.value
        if not path or not os.path.exists(path):
            with salida:
                ui.label("Selecciona una sesion valida.").style("color:#aaa")
            return
        buf = cargar_csv_sesion(path)
        with salida:
            ui.label(f"Sesion: {os.path.basename(path)} — {buf.shape[0]} muestras").style(f"color:{COL['txt']}")
            p = predecir_oneshot(g.infer, buf, cfg)
            if p is None:
                ui.label("Sin modelo cargado (usa --model) o sesion muy corta.").style("color:#aaa")
            else:
                dz = p >= cfg.umbral_decision
                ui.label("DESBALANCE DETECTADO" if dz else "OPERACIÓN NOMINAL").classes("text-3xl font-bold").style(
                    f"color:{COL['alert'] if dz else COL['ok']}")
                ui.label(f"Índice de anomalía = {p*100:.1f}%").style("color:#aaa")
            fr, mag = espectro_display(buf, n_ventana=cfg.n_ventana, cal_ms2=cfg.cuentas_a_ms2)
            fig = go.Figure(); fig.add_scatter(x=fr, y=mag, fill="tozeroy", line=dict(color=COL["ok"]))
            fig.update_layout(template="plotly_dark", height=320, title="Espectro de la sesion (velocidad, mm/s RMS)",
                              paper_bgcolor=COL["bg"], plot_bgcolor=COL["panel"], xaxis_title="Hz",
                              yaxis_title="mm/s RMS")
            ui.plotly(fig).classes("w-full")

    ui.button("Diagnosticar esta sesion", on_click=diagnosticar).style(f"background:{COL['ok']};color:#000")
    return cont


# ============================================================ PANTALLA 3: RESULTADOS
def pantalla_resultados(runs_dir, figs_dir):
    from nicegui import ui
    box = ui.column().classes("w-full")

    def cargar():
        box.clear()
        with box:
            algo = False
            for proto in ("lodo", "loso"):
                rp = os.path.join(runs_dir, f"reporte_{proto}.json")
                if os.path.exists(rp):
                    algo = True
                    r = json.load(open(rp, encoding="utf-8"))
                    ui.label(f"Protocolo {proto.upper()}").classes("text-lg font-bold").style(f"color:{COL['ok']}")
                    ci = f" {r['auc_ci95']}" if r.get("auc_ci95") else ""
                    md = (f"| Métrica | Valor |\n|---|---|\n"
                          f"| AUC macro | {_fmt(r.get('auc_macro'))} ± {_fmt(r.get('auc_macro_sd'))} |\n"
                          f"| AUC agrupado | {_fmt(r.get('auc_pooled'))}{ci} |\n"
                          f"| Sensibilidad | {_fmt(r.get('sens'))} |\n| Especificidad | {_fmt(r.get('spec'))} |\n"
                          f"| ECE | {_fmt(r.get('ece'))} |\n")
                    ui.markdown(md).style(f"color:{COL['txt']}")
            figs = [os.path.join(figs_dir, f) for f in
                    ("roc_lodo.png", "tsne_dominios.png", "saliencia_espectral.png")]
            figs = [f for f in figs if os.path.exists(f)]
            if figs:
                algo = True
                with ui.row().classes("w-full flex-wrap gap-4"):
                    for f in figs:
                        ui.image(img_uri(f)).classes("w-[45%]")
            if not algo:
                ui.label("Aun no hay resultados. Corre el pipeline (train/evaluate/make_figures) "
                         "y pulsa Actualizar.").style("color:#aaa")

    ui.button("↻ Actualizar resultados", on_click=cargar).props("flat")
    cargar()


# ============================================================ PANTALLA 4: SISTEMA Y COSTO
def pantalla_sistema():
    from nicegui import ui
    # Desglose estimado de la Fase 3 (ADXL345, sin PT100). REEMPLAZAR con cotizaciones reales.
    costos = [("ESP32 DevKit V1", 8), ("ADS1115 (ADC 16-bit)", 5), ("ADXL345 x2", 16),
              ("ACS712-5A (corriente)", 4), ("SCT-013-030 (backup)", 12),
              ("Keyphasor LJ12A3-4-Z/BX", 7), ("MAX485 (Modbus)", 3),
              ("Motor Y712-4 0.5 HP", 95), ("VFD Delta MS300", 140),
              ("Biorreactor PMMA (13.3 L)", 110), ("Tablero/cableado/conectores", 65),
              ("Estructura metalica", 75), ("Acople/paletas/sellos", 50),
              ("Blindaje EMI + consumibles", 27), ("Fuente de alimentacion", 53)]
    total = sum(v for _, v in costos)

    ui.label("Sistema y costo").classes("text-2xl font-bold").style(f"color:{COL['ok']}")
    with ui.row().classes("w-full gap-4"):
        for tit, val, sub in [("Costo total (estimado)", f"USD {total}", "Fase 3"),
                              ("Equivalente comercial", "USD 8 000+", "Bently Nevada"),
                              ("Ahorro", "10× a 40×", "segun referencia")]:
            with ui.card().classes("flex-1 items-center").style(f"background:{COL['panel']}"):
                ui.label(tit).classes("text-xs").style("color:#888")
                ui.label(val).classes("text-2xl font-bold").style(f"color:{COL['ok']}")
                ui.label(sub).classes("text-xs").style("color:#888")

    fig = go.Figure(go.Bar(x=[v for _, v in costos], y=[n for n, _ in costos], orientation="h",
                           marker_color=COL["ok"], text=[f"${v}" for _, v in costos], textposition="outside"))
    fig.update_layout(template="plotly_dark", height=460, title=f"Desglose de costo (estimado) — total USD {total}",
                      paper_bgcolor=COL["bg"], plot_bgcolor=COL["panel"], yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=40, t=40, b=10))
    ui.plotly(fig).classes("w-full")
    ui.label("Precios estimados — reemplazar con cotizaciones reales.").classes("text-xs").style("color:#888")

    comer = [("Este proyecto", total), ("SKF Marlin", 4000), ("Bently Nevada", 8000),
             ("Applikon ez2", 10000), ("B&K Vibro", 15000), ("Eppendorf DASbox", 20000), ("Sartorius Biostat A", 25000)]
    figc = go.Figure(go.Bar(x=[n for n, _ in comer], y=[v for _, v in comer],
                            marker_color=[COL["ok"]] + [COL["warn"]] * 6,
                            text=[f"${v:,}" for _, v in comer], textposition="outside"))
    figc.update_layout(template="plotly_dark", height=420, title="Comparativa comercial (escala log)",
                       paper_bgcolor=COL["bg"], plot_bgcolor=COL["panel"], yaxis=dict(type="log"),
                       xaxis=dict(tickangle=-25), margin=dict(l=10, r=10, t=40, b=80))
    ui.plotly(figc).classes("w-full")

    ui.markdown(
        "**Madurez tecnologica — TRL 4.** Banco funcionalmente completo y validado contra fallas "
        "controladas (desbalance presente/ausente) en laboratorio, con generalizacion de dominio sobre "
        "el fluido (protocolos LODO/LOSO). **No** esta certificado para uso industrial (FDA-PAT, ISO 13374, GMP) "
        "ni reemplaza una solucion comercial certificada. Competitivo para investigacion, prototipado y docencia."
    ).style(f"color:{COL['txt']}")

    for arq in ("arquitectura_ciberfisica.png", "figs/arquitectura_ciberfisica.png"):
        if os.path.exists(arq):
            ui.image(img_uri(arq)).classes("w-full")
            break


# ============================================================ APP
def construir_ui(g, cfg, runs_dir, figs_dir):
    from nicegui import ui
    ui.dark_mode().enable()
    ui.label("🧬 Biorreactor — HMI").classes("text-2xl font-bold").style(f"color:{COL['ok']}")
    with ui.tabs().classes("w-full") as tabs:
        ui.tab("Operacion", icon="speed")
        ui.tab("Analisis", icon="science")
        ui.tab("Resultados", icon="insights")
        ui.tab("Sistema y costo", icon="savings")
    with ui.tab_panels(tabs, value="Operacion").classes("w-full"):
        with ui.tab_panel("Operacion"):
            pantalla_operacion(g, cfg)
        with ui.tab_panel("Analisis"):
            pantalla_analisis(g, cfg)
        with ui.tab_panel("Resultados"):
            pantalla_resultados(runs_dir, figs_dir)
        with ui.tab_panel("Sistema y costo"):
            pantalla_sistema()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="puerto serie del banco, p. ej. COM6 (obligatorio)")
    ap.add_argument("--model"); ap.add_argument("--norm"); ap.add_argument("--ood")
    ap.add_argument("--base", help="ruta a base_sana.npz (linea base SANA del medio). "
                    "Necesario si el modelo se entreno con --baseline. Se genera al calibrar "
                    "o con calibrar_base.py")
    ap.add_argument("--umbral", type=float, default=None,
                    help="umbral de decision FALLA (0-1). Usa el OPTIMO de evaluate.py "
                         "(ej. 0.22 LODO, 0.29 LOSO) para captar mas fallas en vivo. Def: 0.5")
    ap.add_argument("--promedio", type=int, default=None,
                    help="ventanas promediadas para el veredicto (suavizado). Mas alto = mas "
                         "estable, menos destellos. Def: 15. Prueba 20-30 si parpadea.")
    ap.add_argument("--histeresis", type=float, default=None,
                    help="zona muerta alrededor del umbral (anti-parpadeo). Def: 0.10. "
                         "Sube a 0.15-0.20 si oscila entre SANO y FALLA.")
    ap.add_argument("--runs", default="runs"); ap.add_argument("--figs", default="figs")
    ap.add_argument("--puerto-web", type=int, default=8080)
    ap.add_argument("--usar-keyphasor", action="store_true",
                    help="usar el keyphasor (sensor de vuelta) como velocidad del eje para las "
                         "marcas 1x/2x/3x, en vez del sincrono del VFD. Recomendado para la demo.")
    ap.add_argument("--ppr", type=int, default=None,
                    help="pulsos por vuelta del keyphasor (def: 1)")
    a = ap.parse_args()
    cfg, g = construir_gestor(a)
    construir_ui(g, cfg, a.runs, a.figs)
    from nicegui import ui
    ui.run(title="HMI Biorreactor", dark=True, reload=False, port=a.puerto_web, show=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
