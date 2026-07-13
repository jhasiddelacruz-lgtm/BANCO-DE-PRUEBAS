#!/usr/bin/env python3
"""
augmentation.py  —  quinto eslabon: aumento de datos para generalizacion de dominio.

Compensa que hay POCOS dominios fuente (pocos fluidos), que es el mayor riesgo del DG.

  1) Mixup multi-fuente de Dirichlet (alpha=0.4): mezcla n muestras con pesos
     Dirichlet. Si se le pasan los dominios, prioriza compañeros de OTROS fluidos,
     creando "dominios intermedios" sinteticos -> amplia la cobertura.   (Shi 2023)
  2) Enmascaramiento espectral: pone a cero bandas de frecuencia al azar, para que
     el modelo no dependa de un solo bin.                                 (Faysal 2022)

Opera sobre tensores torch de un lote:
  X: (B, 121, 5) float   |   y: (B,) float (0/1)   |   domain: (B,) long o None

El mixup devuelve etiqueta SUAVE (float en [0,1]) -> usar BCEWithLogitsLoss.
Nota: el ancla (la propia muestra i) va incluida, asi que el dominio del cabezal
DANN se mantiene como domain[i] en train.py (no se mezcla el dominio).

Ablacion: augment_batch(..., use_mixup=False) apaga SOLO el mixup (deja jitter de
ganancia, reponderacion de bandas y enmascaramiento). Lo usa train.py --sin-mixup.

Prueba rapida:  python augmentation.py
Requiere: torch
"""
import torch


def _partner_diff_domain(domain):
    """Por cada i elige un j con dominio distinto (si existe), si no, cualquiera."""
    B = domain.size(0)
    out = torch.empty(B, dtype=torch.long, device=domain.device)
    for i in range(B):
        diff = (domain != domain[i]).nonzero(as_tuple=True)[0]
        pool = diff if len(diff) > 0 else torch.arange(B, device=domain.device)
        out[i] = pool[torch.randint(len(pool), (1,), device=domain.device)]
    return out


def dirichlet_mixup(X, y, domain=None, alpha=0.4, n_mix=2, cross_domain=True):
    """Mezcla n_mix muestras por salida con pesos Dirichlet(alpha). Devuelve (X_mix, y_mix suave)."""
    B = X.size(0)
    dev = X.device
    w = torch.distributions.Dirichlet(torch.full((n_mix,), float(alpha))).sample((B,)).to(dev)  # (B, n_mix)

    idx = torch.empty(B, n_mix, dtype=torch.long, device=dev)
    idx[:, 0] = torch.arange(B, device=dev)                  # ancla = la propia muestra
    for k in range(1, n_mix):
        idx[:, k] = (_partner_diff_domain(domain) if (cross_domain and domain is not None)
                     else torch.randperm(B, device=dev))

    Xm = torch.zeros_like(X)
    ym = torch.zeros(B, device=dev)
    yf = y.float()
    for k in range(n_mix):
        Xm += w[:, k].view(B, 1, 1) * X[idx[:, k]]
        ym += w[:, k] * yf[idx[:, k]]
    return Xm, ym


def spectral_mask(X, max_width=20, n_masks=2, p=0.5):
    """Pone a cero bandas de frecuencia aleatorias (mismas bandas en los 5 canales)."""
    if p <= 0:
        return X
    B, F, C = X.shape
    X = X.clone()
    for i in range(B):
        if torch.rand(1).item() > p:
            continue
        for _ in range(n_masks):
            w = int(torch.randint(1, max_width + 1, (1,)).item())
            start = int(torch.randint(0, max(1, F - w), (1,)).item())
            X[i, start:start + w, :] = 0.0
    return X


def channel_gain_jitter(X, lo=0.85, hi=1.18, p=0.5):
    """Escala cada canal por una ganancia aleatoria (por muestra). Rompe la
    dependencia del NIVEL absoluto de amplitud, que es lo que delata al fluido."""
    if p <= 0:
        return X
    B, F, C = X.shape
    apply = (torch.rand(B, device=X.device) < p).view(B, 1, 1)
    g = torch.empty(B, 1, C, device=X.device).uniform_(lo, hi)
    g = torch.where(apply, g, torch.ones_like(g))
    return X * g


def band_reweight(X, n_bands=6, lo=0.7, hi=1.4, p=0.5):
    """Multiplica bandas contiguas de frecuencia por ganancias aleatorias (igual en
    los 5 canales). Simula variacion de viscosidad SIN inventar un fluido nuevo."""
    if p <= 0:
        return X
    B, F, C = X.shape
    X = X.clone()
    edges = torch.linspace(0, F, n_bands + 1).long()
    for i in range(B):
        if torch.rand(1).item() > p:
            continue
        for k in range(n_bands):
            g = float(torch.empty(1).uniform_(lo, hi).item())
            X[i, edges[k]:edges[k + 1], :] *= g
    return X


def augment_batch(X, y, domain=None, alpha=0.4, n_mix=2, cross_domain=True,
                  mask_width=20, mask_n=2, mask_p=0.5,
                  gain_lo=0.85, gain_hi=1.18, gain_p=0.5,
                  band_n=6, band_lo=0.7, band_hi=1.4, band_p=0.5,
                  use_mixup=True):
    """Aplica mixup Dirichlet + jitter de ganancia + reponderacion de bandas +
    enmascaramiento espectral. Devuelve (X_aug, y_suave). Todo es solo-train.

    use_mixup=False (ablacion --sin-mixup): NO aplica el mixup; deja la etiqueta dura
    (float) y aplica el resto de aumentos."""
    if use_mixup:
        X, y = dirichlet_mixup(X, y, domain, alpha, n_mix, cross_domain)
    else:
        y = y.float()
    X = channel_gain_jitter(X, gain_lo, gain_hi, gain_p)
    X = band_reweight(X, band_n, band_lo, band_hi, band_p)
    X = spectral_mask(X, mask_width, mask_n, mask_p)
    return X, y


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 16
    X = torch.rand(B, 121, 5)
    y = torch.randint(0, 2, (B,)).float()
    domain = torch.randint(0, 3, (B,))            # 3 fluidos
    Xa, ya = augment_batch(X, y, domain, alpha=0.4, n_mix=2)
    print("===== prueba de augmentation.py =====")
    print(f"X entrada   : {tuple(X.shape)}   y(0/1): {y.tolist()}")
    print(f"X aumentada : {tuple(Xa.shape)}")
    print(f"y suave     : {[round(v,2) for v in ya.tolist()]}")
    print(f"y suave en [0,1]: {bool((ya>=0).all() and (ya<=1).all())}")
    cambio = (Xa - X).abs().mean().item()
    ceros = (Xa == 0).float().mean().item()
    print(f"cambio medio por mixup : {cambio:.4f}  (>0 => mezcla aplicada)")
    print(f"fraccion de bins en 0  : {ceros:.3f}  (enmascaramiento espectral)")
    # prueba de la ablacion sin mixup
    Xb, yb = augment_batch(X, y, domain, use_mixup=False)
    print(f"\nsin mixup -> y dura preservada: {bool((yb==y).all())}  (debe ser True)")
