# ============================================================
# ZT TRADERS - Nucleo IA v3 (sin Streamlit: se puede probar solo)
#
# Contiene TODO el aprendizaje nuevo:
#  - 15 variables de mercado por vela (antes: solo RSI y Bollinger)
#  - Construccion del dataset de backtesting (miles de ejemplos)
#  - Entrenamiento con division por tiempo + calibracion de probabilidad
#  - Evaluacion honesta: precision y cobertura por umbral de confianza
#
# Convencion de datos: DataFrames con columnas en minuscula
#   open, high, low, close, volume (volume opcional)
# y el tiempo en el indice (DatetimeIndex) o en columna 'timestamp' (ms).
#
# Correccion v3.4: las velas forex de 1m de Yahoo vienen planas
# (max == min); cuerpo/mechas/rango_rel ahora valen 0 en velas planas
# en vez de NaN (antes TODAS las filas forex eran invalidas y jamas
# habia señal en EUR/USD y similares). Las variables son las mismas:
# el modelo entrenado sigue siendo compatible, no hay que reentrenar.
# ============================================================
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ta

# ---------------- Constantes ----------------

UMBRAL_NEUTRO_PCT = 0.03     # movimientos menores a esto = ruido/empate, no entrenan
UMBRAL_CONFIANZA_ML = 0.62   # probabilidad minima para emitir señal con modelo
UMBRAL_SCORE_MIN = 3         # puntaje tecnico minimo si no hay modelo entrenado

FEATURES_BASE = [
    "rsi",              # RSI 14
    "precio_relativo",  # posicion dentro de Bollinger (0=banda baja, 1=banda alta)
    "ema_gap",          # (EMA9-EMA21)/precio*100  -> tendencia corta
    "dist_ema50",       # (precio-EMA50)/precio*100 -> tendencia media
    "macd_hist",        # histograma MACD / precio *100
    "adx",              # fuerza de tendencia 0-100
    "atr_rel",          # ATR/precio*100 -> volatilidad
    "bb_ancho",         # ancho Bollinger / precio *100
    "roc_3",            # variacion % ultimas 3 velas
    "roc_5",            # variacion % ultimas 5 velas
    "stoch_k",          # estocastico %K 0-100
    "vol_rel",          # volumen / su media 20 (1.0 si no hay volumen)
    "cuerpo",           # (cierre-apertura)/(max-min) de la vela, -1..1
    "rango_rel",        # (max-min)/precio*100
    "hora",             # hora UTC 0-23
    # --- v3.2: conocimiento de trader profesional (accion del precio) ---
    "mecha_sup",        # mecha superior / rango (rechazo desde arriba)
    "mecha_inf",        # mecha inferior / rango (rechazo desde abajo)
    "envolvente",       # +1 envolvente alcista, -1 bajista, 0 ninguna
    "pin_bar",          # +1 pin bar alcista, -1 bajista, 0 ninguna
    "inside_bar",       # 1 si la vela quedo dentro de la anterior
    "dist_resistencia", # distancia al maximo de 20 velas, en ATR (tope 10)
    "dist_soporte",     # distancia al minimo de 20 velas, en ATR (tope 10)
    "ruptura",          # +1 rompe maximo previo, -1 rompe minimo previo, 0 no
    "posicion_rango",   # posicion del precio en su rango de 20 velas (0..1)
    "regimen_cod",      # 0 rango, 1 tendencia alcista, 2 tendencia bajista, 3 volatil
]

REGIMENES = {0: "Rango", 1: "Tendencia alcista", 2: "Tendencia bajista", 3: "Volátil"}

def nombre_regimen(cod):
    try:
        return REGIMENES[int(round(float(cod)))]
    except Exception:
        return "Sin régimen"
FEATURES_COMPLETAS = FEATURES_BASE + ["es_call"]

# ---------------- Indicadores / variables ----------------

def _horas(df):
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        return pd.Series(idx.hour, index=df.index, dtype=float)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return pd.Series(ts.dt.hour.values, index=df.index, dtype=float)
    return pd.Series(12.0, index=df.index)

def agregar_indicadores(df):
    """Recibe OHLC(V) en minusculas y devuelve copia con todas las variables v3."""
    df = df.copy()
    close, high, low = df["close"], df["high"], df["low"]

    df["_rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["_bb_high"] = bb.bollinger_hband()
    df["_bb_low"] = bb.bollinger_lband()
    df["_sma20"] = close.rolling(20).mean()
    ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["_macd_hist"] = ta.trend.MACD(close).macd_diff()
    df["_adx"] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
    df["_atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["_stoch"] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()

    rango_bb = (df["_bb_high"] - df["_bb_low"]).replace(0, np.nan)
    rango_vela = (high - low).replace(0, np.nan)

    df["rsi"] = df["_rsi"]
    df["precio_relativo"] = ((close - df["_bb_low"]) / rango_bb).clip(0, 1)
    df["ema_gap"] = (ema9 - ema21) / close * 100
    df["dist_ema50"] = (close - ema50) / close * 100
    df["macd_hist"] = df["_macd_hist"] / close * 100
    df["adx"] = df["_adx"]
    df["atr_rel"] = df["_atr"] / close * 100
    df["bb_ancho"] = rango_bb / close * 100
    df["roc_3"] = close.pct_change(3) * 100
    df["roc_5"] = close.pct_change(5) * 100
    df["stoch_k"] = df["_stoch"]
    if "volume" in df.columns and df["volume"].fillna(0).sum() > 0:
        vol_sma = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["vol_rel"] = (df["volume"] / vol_sma).fillna(1.0).clip(0, 10)
    else:
        df["vol_rel"] = 1.0
    df["cuerpo"] = ((close - df["open"]) / rango_vela).clip(-1, 1)
    df["rango_rel"] = rango_vela / close * 100
    # Forex en Yahoo: las velas de 1m vienen "planas" (max == min en casi
    # todas), rango_vela = 0 y estas divisiones daban NaN en TODAS las
    # filas -> filas_validas() = 0 y el forex NUNCA emitia señal (bug
    # corregido en v3.4). Una vela plana no tiene cuerpo ni mechas: 0.
    df["cuerpo"] = df["cuerpo"].fillna(0.0)
    df["rango_rel"] = df["rango_rel"].fillna(0.0)
    df["hora"] = _horas(df)

    # --- v3.2: accion del precio, niveles y regimen ---
    apertura = df["open"]
    max_cuerpo = np.maximum(apertura, close)
    min_cuerpo = np.minimum(apertura, close)
    cuerpo_ratio = (close - apertura).abs() / rango_vela
    df["mecha_sup"] = ((high - max_cuerpo) / rango_vela).clip(0, 1)
    df["mecha_inf"] = ((min_cuerpo - low) / rango_vela).clip(0, 1)
    df["mecha_sup"] = df["mecha_sup"].fillna(0.0)
    df["mecha_inf"] = df["mecha_inf"].fillna(0.0)
    cuerpo_ratio = cuerpo_ratio.fillna(0.0)

    prev_o = apertura.shift(1)
    prev_c = close.shift(1)
    env_alcista = (close > apertura) & (prev_c < prev_o) & (close >= prev_o) & (apertura <= prev_c)
    env_bajista = (close < apertura) & (prev_c > prev_o) & (close <= prev_o) & (apertura >= prev_c)
    df["envolvente"] = np.where(env_alcista, 1.0, np.where(env_bajista, -1.0, 0.0))
    pin_alcista = (df["mecha_inf"] >= 0.6) & (cuerpo_ratio <= 0.35)
    pin_bajista = (df["mecha_sup"] >= 0.6) & (cuerpo_ratio <= 0.35)
    df["pin_bar"] = np.where(pin_alcista, 1.0, np.where(pin_bajista, -1.0, 0.0))
    df["inside_bar"] = ((high <= high.shift(1)) & (low >= low.shift(1))).astype(float)

    max_prev = high.shift(1).rolling(20).max()
    min_prev = low.shift(1).rolling(20).min()
    atr_seguro = df["_atr"].replace(0, np.nan)
    df["dist_resistencia"] = ((max_prev - close) / atr_seguro).clip(0, 10)
    df["dist_soporte"] = ((close - min_prev) / atr_seguro).clip(0, 10)
    df["ruptura"] = np.where(close > max_prev, 1.0, np.where(close < min_prev, -1.0, 0.0))
    rango_20 = (high.rolling(20).max() - low.rolling(20).min()).replace(0, np.nan)
    df["posicion_rango"] = ((close - low.rolling(20).min()) / rango_20).clip(0, 1)

    atr_q80 = df["atr_rel"].rolling(100, min_periods=50).quantile(0.8)
    es_volatil = df["atr_rel"] >= atr_q80
    df["regimen_cod"] = np.select(
        [es_volatil,
         (df["_adx"] >= 25) & (df["ema_gap"] > 0),
         (df["_adx"] >= 25) & (df["ema_gap"] < 0)],
        [3.0, 1.0, 2.0],
        default=0.0,
    )
    return df

def fila_features(df_ind, pos):
    """Vector base (sin direccion) de la vela en la posicion pos de un df con indicadores."""
    fila = df_ind.iloc[pos]
    return [float(fila[c]) for c in FEATURES_BASE]

def filas_validas(df_ind):
    """Posiciones (indices enteros) cuyas variables no tienen NaN."""
    mascara = df_ind[FEATURES_BASE].notna().all(axis=1)
    return np.where(mascara.values)[0]

# ---------------- Puntaje tecnico (direccion base) ----------------

def direccion_por_puntaje(rsi, precio, bb_low, bb_high, sma, sma_prev):
    """Regla explicable de la casa. Devuelve (direccion, score, razones)."""
    score = 0
    razones = []
    if rsi <= 30:
        score += 3; razones.append("RSI Sobreventa")
    elif rsi <= 45:
        score += 1; razones.append("RSI Bajo")
    elif rsi >= 70:
        score -= 3; razones.append("RSI Sobrecompra")
    elif rsi >= 55:
        score -= 1; razones.append("RSI Alto")
    if precio <= bb_low:
        score += 2; razones.append("Banda Inf")
    elif precio >= bb_high:
        score -= 2; razones.append("Banda Sup")
    if precio > sma and sma >= sma_prev:
        score += 1; razones.append("Tend. Alcista")
    elif precio < sma and sma <= sma_prev:
        score -= 1; razones.append("Tend. Bajista")
    if score > 0:
        direccion = "CALL"
    elif score < 0:
        direccion = "PUT"
    else:
        direccion = "CALL" if rsi < 50 else "PUT"
        razones.append("Sin ventaja clara")
    return direccion, score, razones

def evaluar_confluencia(f, direccion):
    """Regla de confluencia del trader profesional.

    f = dict de variables de la vela (por nombre). Devuelve
    (confirmaciones, bloqueos, detalles): la señal solo deberia emitirse si
    hay >= 2 confirmaciones y ningun bloqueo duro.
    """
    signo = 1 if direccion == "CALL" else -1
    confirmaciones = 0
    bloqueos = []
    detalles = []

    # 1) Tendencia (EMAs alineadas)
    if f.get("ema_gap", 0) * signo > 0 and f.get("dist_ema50", 0) * signo > 0:
        confirmaciones += 1
        detalles.append("Tendencia a favor")
    elif f.get("adx", 0) >= 30 and f.get("dist_ema50", 0) * signo < 0:
        bloqueos.append(f"contra tendencia fuerte (ADX {int(f.get('adx', 0))})")

    # 2) Nivel de soporte / resistencia cercano
    d_res = f.get("dist_resistencia", 99)
    d_sop = f.get("dist_soporte", 99)
    if direccion == "CALL":
        if d_sop <= 1.2 and d_res >= 1.5:
            confirmaciones += 1
            detalles.append("Cerca de soporte con espacio arriba")
        if d_res < 0.4:
            bloqueos.append("precio pegado a resistencia")
    else:
        if d_res <= 1.2 and d_sop >= 1.5:
            confirmaciones += 1
            detalles.append("Cerca de resistencia con espacio abajo")
        if d_sop < 0.4:
            bloqueos.append("precio pegado a soporte")

    # 3) Patron de vela (envolvente o pin bar)
    patron = f.get("envolvente", 0) or f.get("pin_bar", 0)
    if patron * signo > 0:
        confirmaciones += 1
        detalles.append("Patrón de vela a favor")
    elif patron * signo < 0:
        detalles.append("Patrón de vela en contra")

    # 4) Impulso o ruptura en la direccion
    if f.get("roc_3", 0) * signo > 0 or f.get("ruptura", 0) * signo > 0:
        confirmaciones += 1
        detalles.append("Impulso a favor")

    return confirmaciones, bloqueos, detalles

# ---------------- Dataset de backtesting ----------------

def construir_dataset(lista_df):
    """A partir de DataFrames OHLC historicos genera el dataset etiquetado.

    Por cada vela t: variables conocidas al cierre de t, entrada al cierre de t
    y salida al cierre de t+1 (operacion de 1 minuto, igual que la app en vivo).
    Se crean DOS ejemplos por decision (CALL y PUT) con su resultado real.
    Los movimientos menores al umbral neutro se descartan (ruido).
    Devuelve (X, y, grupos, tiempos): grupos identifica la decision (las dos
    direcciones comparten grupo) y tiempos ordena para la division temporal.
    """
    X, y, grupos, tiempos = [], [], [], []
    grupo_actual = 0
    for df in lista_df:
        if df is None or len(df) < 60:
            continue
        d = agregar_indicadores(df)
        validas = set(filas_validas(d).tolist())
        closes = d["close"].values
        if isinstance(d.index, pd.DatetimeIndex):
            marcas = d.index.values
        else:
            marcas = np.arange(len(d))
        for t in range(len(d) - 1):
            if t not in validas:
                continue
            entrada, salida = closes[t], closes[t + 1]
            if not entrada or np.isnan(entrada) or np.isnan(salida):
                continue
            mov_pct = (salida - entrada) / entrada * 100
            if abs(mov_pct) < UMBRAL_NEUTRO_PCT:
                continue
            base = fila_features(d, t)
            subio = 1 if salida > entrada else 0
            X.append(base + [1]); y.append(subio);        grupos.append(grupo_actual); tiempos.append(marcas[t])
            X.append(base + [0]); y.append(1 - subio);    grupos.append(grupo_actual); tiempos.append(marcas[t])
            grupo_actual += 1
    return (np.array(X, dtype=float), np.array(y, dtype=int),
            np.array(grupos, dtype=int), np.array(tiempos))

# ---------------- Entrenamiento + evaluacion honesta ----------------

def entrenar_modelo(X, y, grupos, tiempos):
    """Division por TIEMPO (60% entrena, 20% calibra, 20% prueba), sin mezclar.

    El modelo base es HistGradientBoosting y la probabilidad se CALIBRA
    (sigmoide) para que un 70% anunciado se acerque a ganar ~70% de las veces.
    La evaluacion de prueba reproduce el uso real: por decision se calcula la
    probabilidad de CALL y de PUT, se elige la mayor y se emite solo si
    supera el umbral; se reporta precision y cobertura por umbral.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    if len(X) == 0 or len(set(y.tolist())) < 2:
        raise ValueError("Dataset insuficiente o de una sola clase.")

    orden = np.argsort(np.array([str(t) for t in tiempos]), kind="stable")
    X, y, grupos = X[orden], y[orden], grupos[orden]
    grupos_unicos = np.unique(grupos)  # ya quedan en orden temporal
    if len(grupos_unicos) < 100:
        raise ValueError(f"Solo {len(grupos_unicos)} decisiones utiles; se necesitan al menos 100.")
    g_train = set(grupos_unicos[: int(len(grupos_unicos) * 0.6)].tolist())
    g_cal = set(grupos_unicos[int(len(grupos_unicos) * 0.6): int(len(grupos_unicos) * 0.8)].tolist())

    def mascara(conjunto):
        return np.array([g in conjunto for g in grupos])

    m_tr, m_cal = mascara(g_train), mascara(g_cal)
    m_te = ~(m_tr | m_cal)

    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          random_state=42)
    base.fit(X[m_tr], y[m_tr])
    try:
        # sklearn >= 1.6: el modelo ya entrenado se envuelve con FrozenEstimator
        from sklearn.frozen import FrozenEstimator
        modelo = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    except Exception:
        # sklearn < 1.6
        modelo = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
    modelo.fit(X[m_cal], y[m_cal])

    prob_test = modelo.predict_proba(X[m_te])[:, list(modelo.classes_).index(1)]
    brier = float(brier_score_loss(y[m_te], prob_test))

    # Evaluacion por decision en el tramo de prueba
    idx_test = np.where(m_te)[0]
    posicion = {int(fila): k for k, fila in enumerate(idx_test)}
    idx_regimen = FEATURES_BASE.index("regimen_cod")
    decisiones = {}
    for i in idx_test:
        entrada_dec = decisiones.setdefault(grupos[i], {"regimen": X[i][idx_regimen]})
        entrada_dec["call" if X[i][-1] == 1 else "put"] = (prob_test[posicion[int(i)]], y[i])
    por_umbral = []
    total_dec = len(decisiones)
    for umbral in (0.55, 0.60, 0.62, 0.65, 0.70):
        emitidas = ganadas = 0
        for d in decisiones.values():
            if "call" not in d or "put" not in d:
                continue
            p_call, y_call = d["call"]
            p_put, _ = d["put"]
            if max(p_call, p_put) < umbral:
                continue
            emitidas += 1
            ganada_real = y_call if p_call >= p_put else 1 - y_call
            ganadas += ganada_real
        por_umbral.append({
            "umbral": umbral,
            "emitidas": int(emitidas),
            "precision_%": round(ganadas / emitidas * 100, 1) if emitidas else 0.0,
            "cobertura_%": round(emitidas / total_dec * 100, 1) if total_dec else 0.0,
        })

    # Precision por regimen al umbral de trabajo (0.62): el "diario" del cerebro
    por_regimen = []
    for cod, nombre in REGIMENES.items():
        subset = {g: d for g, d in decisiones.items() if int(round(float(d.get("regimen", -1)))) == cod}
        emitidas = ganadas = 0
        for d in subset.values():
            if "call" not in d or "put" not in d:
                continue
            p_call, y_call = d["call"]
            p_put, _ = d["put"]
            if max(p_call, p_put) < UMBRAL_CONFIANZA_ML:
                continue
            emitidas += 1
            ganadas += y_call if p_call >= p_put else 1 - y_call
        por_regimen.append({
            "regimen": nombre,
            "decisiones": int(len(subset)),
            "emitidas_0.62": int(emitidas),
            "precision_%": round(ganadas / emitidas * 100, 1) if emitidas else 0.0,
        })

    metricas = {
        "fecha_entrenamiento": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ejemplos_entrenamiento": int(m_tr.sum()),
        "decisiones_prueba": int(total_dec),
        "brier": round(brier, 4),
        "por_umbral": por_umbral,
        "por_regimen": por_regimen,
    }
    return {"modelo": modelo, "features": FEATURES_COMPLETAS, "metricas": metricas}

MIN_OPS_MODELO_VIVO = 60     # operaciones reales minimas para activar el cerebro en vivo
REENTRENAR_CADA_OPS = 20     # reentrenar cuando se juntan 20 operaciones nuevas

# El patron central del trading es una INTERACCION (direccion x contexto):
# comprar en tendencia alcista no es lo mismo que comprar en rango, y con
# pocos datos un arbol tarda en descubrir esas combinaciones solo. Por eso
# el modelo en vivo agrega 3 variables derivadas que las hacen explicitas.
FEATURES_VIVO = FEATURES_COMPLETAS + ["alineado_tendencia", "alineado_ema", "espacio_favorable"]

def _extender_vivo(vector_base, es_call):
    fv = dict(zip(FEATURES_BASE, vector_base))
    regimen = int(round(float(fv.get("regimen_cod", 0))))
    dir_sign = 1.0 if es_call else -1.0
    if (regimen == 1 and es_call) or (regimen == 2 and not es_call):
        alineado = 1.0
    elif regimen in (1, 2):
        alineado = -1.0
    else:
        alineado = 0.0
    alineado_ema = float(np.sign(float(fv.get("ema_gap", 0.0)))) * dir_sign
    espacio = (float(fv.get("dist_resistencia", 0.0)) - float(fv.get("dist_soporte", 0.0))) * dir_sign
    return list(vector_base) + [1.0 if es_call else 0.0, alineado, alineado_ema, espacio]

def ops_validas_para_vivo(historial):
    """Operaciones del historial que sirven para aprender (traen variables
    completas y resultado)."""
    return [h for h in (historial or [])
            if h.get("features") and len(h["features"]) == len(FEATURES_COMPLETAS)
            and h.get("resultado") in ("GANADA", "PERDIDA")]

def autopsia_perdidas(historial):
    """Autopsia de señales (v3.9, idea destilada de 'signal-postmortem' del
    proyecto claude-trading-skills, MIT): atribuye cada operacion PERDIDA a
    una causa probable segun el estado del mercado guardado con la señal.
    Devuelve {causa: cantidad} ordenado de mayor a menor."""
    causas = {}
    for h in ops_validas_para_vivo(historial):
        if h.get("resultado") != "PERDIDA":
            continue
        f = h["features"]
        es_call = f[-1] == 1
        regimen = int(f[FEATURES_BASE.index("regimen_cod")])
        dist_res = float(f[FEATURES_BASE.index("dist_resistencia")])
        dist_sop = float(f[FEATURES_BASE.index("dist_soporte")])
        if (regimen == 1 and not es_call) or (regimen == 2 and es_call):
            causa = "Contra la tendencia del régimen"
        elif (es_call and dist_res < 0.5) or (not es_call and dist_sop < 0.5):
            causa = "Pegada a soporte/resistencia"
        elif regimen == 0:
            causa = "En rango, sin confirmación"
        elif regimen == 3:
            causa = "En volatilidad extrema"
        else:
            causa = "Sin causa clara"
        causas[causa] = causas.get(causa, 0) + 1
    return dict(sorted(causas.items(), key=lambda kv: -kv[1]))

def postura_sesion(historial, ventana=20):
    """Postura de exposición de la sesión (v3.9, idea destilada de
    'exposure-coach', MIT): mira el acierto de las ultimas operaciones
    reales y recomienda cuanto exponerse. Devuelve (postura, acierto_% o
    None, n). Es una recomendacion visible; nunca bloquea señales."""
    ops = ops_validas_para_vivo(historial)[-ventana:]
    if len(ops) < 10:
        return ("SIN DATOS", None, len(ops))
    acierto = sum(1 for h in ops if h["resultado"] == "GANADA") / len(ops) * 100
    if acierto >= 55:
        return ("PLENA", round(acierto, 1), len(ops))
    if acierto >= 45:
        return ("NEUTRA", round(acierto, 1), len(ops))
    return ("REDUCIR", round(acierto, 1), len(ops))

def _gatillo_rechazo(fv, es_call):
    """Gatillo de rechazo en velas japonesas: pin bar, envolvente o mecha
    larga contra el movimiento (el precio intento seguir y fue rechazado
    en el nivel). Es la confirmacion clasica de que el extremo se agoto;
    sin el, entrar en rango es entrar a mitad del ruido."""
    if es_call:
        return (fv.get("pin_bar", 0) == 1 or fv.get("envolvente", 0) == 1
                or float(fv.get("mecha_inf", 0)) >= 0.5)
    return (fv.get("pin_bar", 0) == -1 or fv.get("envolvente", 0) == -1
            or float(fv.get("mecha_sup", 0)) >= 0.5)

def lecciones_aprendidas(historial):
    """LECCIONES DE LA EXPERIENCIA (v3.12; reglas con evidencia en v3.18):
    convierte los RESULTADOS (victorias Y derrotas) en reglas correctivas
    ACTIVAS. La v3.12 las sacaba de la autopsia de perdidas: 'esta causa
    acumula muchas perdidas'. Eso confunde VOLUMEN con RIESGO: el rango
    acumulaba el 41% de las perdidas porque acumulaba el 46% de las
    operaciones, y sin embargo era el regimen que MAS ganaba (45%). Y la
    leccion contra-tendencia castigaba el comportamiento principal de la
    app (87% de las señales en tendencia van contra ella) sin evidencia
    de que a favor rindiera mejor. Desde v3.18 cada leccion nace de
    comparar ACIERTOS (contexto contra global o contra su opuesto) con
    muestra minima; si la evidencia no alcanza, la leccion NO existe y
    el panel lo dice. Devuelve [{id, titulo, evidencia, efecto,
    castigadas?}]."""
    ops = _decisivas_para_aprender(historial)
    lecciones = []
    if len(ops) >= 30:
        wr_global = (sum(1 for h in ops if h["resultado"] == "GANADA")
                     / len(ops) * 100)

        def _wr(filtro):
            sub = [h for h in ops if filtro(h)]
            if not sub:
                return 0.0, 0
            return (sum(1 for h in sub if h["resultado"] == "GANADA")
                    / len(sub) * 100), len(sub)

        wr_vol, n_vol = _wr(lambda h: _regimen_de(h) == 3)
        if n_vol >= 30 and wr_vol < wr_global - 5:
            lecciones.append({
                "id": "VOLATILIDAD",
                "titulo": "En volatilidad extrema, vara más alta",
                "evidencia": f"En régimen volátil el acierto fue {wr_vol:.1f}% en {n_vol} ops, contra {wr_global:.1f}% global",
                "efecto": "En régimen volátil el nivel de la señal baja un escalón."})
        wr_rango, n_rango = _wr(lambda h: _regimen_de(h) == 0)
        if n_rango >= 30 and wr_rango < wr_global - 5:
            lecciones.append({
                "id": "RANGO_FLOJO",
                "titulo": "En rango el acierto cae: vara más alta",
                "evidencia": f"En régimen de rango el acierto fue {wr_rango:.1f}% en {n_rango} ops, contra {wr_global:.1f}% global",
                "efecto": "En régimen de rango el nivel de la señal baja un escalón."})

        def _es_contra(h):
            f = h["features"]
            return ((int(f[24]) == 1 and f[-1] != 1)
                    or (int(f[24]) == 2 and f[-1] == 1))

        def _es_favor(h):
            f = h["features"]
            return ((int(f[24]) == 1 and f[-1] == 1)
                    or (int(f[24]) == 2 and f[-1] != 1))

        wr_contra, n_contra = _wr(_es_contra)
        wr_favor, n_favor = _wr(_es_favor)
        if n_favor >= 30 and n_contra >= 20 and wr_contra < wr_favor - 5:
            lecciones.append({
                "id": "CONTRA_TENDENCIA",
                "titulo": "Contra la tendencia rinde menos que a favor",
                "evidencia": f"Contra la tendencia: {wr_contra:.1f}% en {n_contra} ops; a favor: {wr_favor:.1f}% en {n_favor} ops",
                "efecto": "Las señales contra la tendencia del régimen bajan un escalón de nivel adicional."})
    por_pd = {}
    for h in ops:
        clave = (h.get("par"), h.get("señal"))
        n, g = por_pd.get(clave, (0, 0))
        por_pd[clave] = (n + 1, g + (1 if h["resultado"] == "GANADA" else 0))
    castigadas = []
    for (par, direc), (n, g) in por_pd.items():
        if par and direc in ("CALL", "PUT") and n >= 15 and g / n * 100 <= 35:
            castigadas.append(f"{par} {direc} ({g}/{n} = {g / n * 100:.0f}%)")
    if castigadas:
        lecciones.append({
            "id": "DIRECCION_CASTIGADA",
            "titulo": "Direcciones castigadas por la experiencia",
            "evidencia": "Acierto de 35% o menos con 15+ operaciones en: " + "; ".join(castigadas),
            "efecto": "En esos pares y direcciones el nivel baja un escalón hasta que el acierto se recupere.",
            "castigadas": castigadas})
    return lecciones

def aplicar_lecciones(nivel, fv, direccion, par, lecciones):
    """Aplica las lecciones activas al NIVEL de una señal nueva. fv es el
    dict de variables por nombre. Devuelve (nivel_ajustado, [motivos]).
    La señal NUNCA se bloquea ni cambia de direccion (regla de Jose):
    solo se gradua con la honestidad de la experiencia, como un trader
    que ya perdio en ese mismo contexto y dejo de confiar a ciegas."""
    if not lecciones or nivel not in ("Baja", "Media", "Alta"):
        return nivel, []
    orden = ["Baja", "Media", "Alta"]
    idx = orden.index(nivel)
    motivos = []
    es_call = direccion == "CALL"
    regimen = int(fv.get("regimen_cod", 0))
    ids = {l["id"] for l in lecciones}

    def _baja(motivo):
        nonlocal idx
        if idx > 0:
            idx -= 1
            motivos.append(motivo)

    if "RANGO_FLOJO" in ids and regimen == 0:
        _baja("régimen de rango con acierto medido bajo el global")
    if "VOLATILIDAD" in ids and regimen == 3:
        _baja("régimen volátil con acierto medido bajo el global")
    if "CONTRA_TENDENCIA" in ids and ((regimen == 1 and not es_call) or (regimen == 2 and es_call)):
        _baja("contra la tendencia del régimen (medido peor que a favor)")
    for l in lecciones:
        if l["id"] == "DIRECCION_CASTIGADA" and any(
                c.startswith(f"{par} {direccion} ") or c.startswith(f"{par} {direccion}(")
                for c in l.get("castigadas", [])):
            _baja(f"{par} {direccion} castigado por sus resultados")
    return orden[idx], motivos

# ============================================================
# CATALOGADOR ZT (v3.14)
# ============================================================
# La idea del Catalogador de DAM Trader, en version honesta: catalogar
# que par x direccion esta funcionando con resultados MEDIDOS (no
# prometidos), separando etiquetas limpias de sucias y con el filtro
# "solo con tendencia" de ellos. Estados con muestra minima para no
# concluir de ruido: OPERABLE exige >=20 ops y acierto >= 53,8% (el
# punto de equilibrio con payout 86% en IQ Option).

NOMBRES_REGIMEN = {0: "Rango", 1: "Tendencia alcista", 2: "Tendencia bajista", 3: "Volátil"}


def _regimen_de(op):
    f = op.get("features")
    if f and len(f) > 24:
        try:
            return int(round(float(f[24])))
        except (TypeError, ValueError):
            return None
    return None


def catalogo(historial, solo_limpias=False, solo_tendencia=False):
    """Efectividad medida por par x direccion (+ resumen por regimen).

    Devuelve {"total", "filas", "regimenes", "mejor", "peor"}; cada fila:
    par, direccion, ops, ganadas, perdidas, acierto, limpias, estado."""
    limpias_set = ("Twelve Data", "Binance")
    validas = [h for h in (historial or []) if h.get("resultado") in ("GANADA", "PERDIDA")]
    if solo_limpias:
        validas = [h for h in validas if h.get("fuente_datos") in limpias_set]
    if solo_tendencia:
        validas = [h for h in validas if _regimen_de(h) in (1, 2)]

    grupos = {}
    for h in validas:
        k = (h.get("par") or "?", h.get("señal") or "?")
        g = grupos.setdefault(k, {"ops": 0, "g": 0, "limpias": 0})
        g["ops"] += 1
        if h["resultado"] == "GANADA":
            g["g"] += 1
        if h.get("fuente_datos") in limpias_set:
            g["limpias"] += 1

    filas = []
    for (par, sen), g in grupos.items():
        ac = (g["g"] / g["ops"] * 100) if g["ops"] else 0.0
        if g["ops"] < 10:
            estado = "⚪ SIN MUESTRA"
        elif ac >= 53.8 and g["ops"] >= 20:
            estado = "🟢 OPERABLE"
        elif ac < 45 and g["ops"] >= 15:
            estado = "🔴 DESCANSAR"
        else:
            estado = "🟡 EN OBSERVACIÓN"
        filas.append({"par": par, "direccion": sen, "ops": g["ops"], "ganadas": g["g"],
                      "perdidas": g["ops"] - g["g"], "acierto": round(ac, 1),
                      "limpias": g["limpias"], "estado": estado})
    filas.sort(key=lambda f: (-f["ops"], -f["acierto"]))

    por_reg = {}
    for h in validas:
        r = _regimen_de(h)
        nombre = NOMBRES_REGIMEN.get(r, "Sin dato")
        a, b = por_reg.get(nombre, (0, 0))
        por_reg[nombre] = (a + 1, b + (1 if h["resultado"] == "GANADA" else 0))
    regimenes = [{"regimen": n, "ops": a, "acierto": round(b / a * 100, 1) if a else 0.0}
                 for n, (a, b) in sorted(por_reg.items(), key=lambda kv: -kv[1][0])]

    con_muestra = [f for f in filas if f["ops"] >= 10]
    mejor = max(con_muestra, key=lambda f: f["acierto"]) if con_muestra else None
    peor = min(con_muestra, key=lambda f: f["acierto"]) if con_muestra else None
    return {"total": len(validas), "filas": filas, "regimenes": regimenes,
            "mejor": mejor, "peor": peor}


def entrenar_modelo_vivo(historial):
    """Cerebro EN VIVO: aprende de las operaciones reales de la app.

    Cada operacion registrada trae el estado del mercado (26 variables,
    incluida la direccion tomada) y su resultado real (GANADA/PERDIDA,
    juzgado por la app con el precio de entrada y el de cierre). Con eso
    aprende en que estados del mercado las señales de ESTE sistema ganan
    y en cuales pierden, tambien en forex, donde no hay backtest de
    Binance. Division por orden de registro: 70% entrena, 30% se usa
    para calibrar y medir precision sin tocar. Devuelve un info
    compatible con prob_ganada()/guardar_modelo(), o None si aun no
    hay operaciones suficientes.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss

    ops = ops_validas_para_vivo(historial)
    # ETIQUETAS LIMPIAS (v3.8): las operaciones juzgadas con precios
    # sincronizados (Twelve Data en forex, Binance en crypto) valen mas
    # que las juzgadas con el precio retrasado de Yahoo. Cuando hay
    # suficientes operaciones limpias, el cerebro entrena SOLO con esas.
    limpias = [h for h in ops if h.get("fuente_datos") in ("Twelve Data", "Binance")]
    usando_limpias = len(limpias) >= MIN_OPS_MODELO_VIVO
    if usando_limpias:
        ops = limpias
    if len(ops) < MIN_OPS_MODELO_VIVO:
        return None
    X = np.nan_to_num(np.array([_extender_vivo(h["features"][:len(FEATURES_BASE)],
                                               h["features"][-1] == 1) for h in ops], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array([1 if h["resultado"] == "GANADA" else 0 for h in ops], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    corte = int(len(ops) * 0.7)
    X_tr, y_tr, X_te, y_te = X[:corte], y[:corte], X[corte:], y[corte:]
    if len(set(y_tr.tolist())) < 2 or len(y_te) < 10:
        return None

    def _base():
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                              max_leaf_nodes=15, min_samples_leaf=15,
                                              random_state=42)
    modelo = CalibratedClassifierCV(_base(), method="sigmoid", cv=3)
    modelo.fit(X_tr, y_tr)
    prob_te = modelo.predict_proba(X_te)[:, list(modelo.classes_).index(1)]
    brier = float(brier_score_loss(y_te, prob_te))
    con_filtro = prob_te >= 0.55
    metricas = {
        "fecha_entrenamiento": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ejemplos_entrenamiento": int(len(ops)),
        "etiquetas_limpias": bool(usando_limpias),
        "ops_etiqueta_limpia": int(len(limpias)),
        "winrate_prueba_%": round(float(y_te.mean()) * 100, 1),
        "precision_filtro_0.55_%": round(float(y_te[con_filtro].mean()) * 100, 1) if con_filtro.sum() else 0.0,
        "cobertura_filtro_0.55_%": round(float(con_filtro.mean()) * 100, 1),
        "brier": round(brier, 4),
    }
    # TERMOMETRO DE TERRENO (v3.15, idea del indice de disimilitud de
    # FreqAI de Freqtrade): se guarda como es el terreno con el que el
    # cerebro aprendio (media y desviacion por variable + la distancia
    # tipica de sus propios ejemplos), para poder avisar cuando una
    # señal nueva cae en un contexto que el modelo nunca vio.
    Xb = np.nan_to_num(np.array([h["features"][:len(FEATURES_BASE)] for h in ops], dtype=float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    _media = Xb.mean(axis=0)
    _desv = Xb.std(axis=0) + 1e-9
    _dists = np.sqrt((((Xb - _media) / _desv) ** 2).mean(axis=1))
    terreno = {"media": _media.tolist(), "desv": _desv.tolist(),
               "umbral": float(np.percentile(_dists, 90))}
    return {"modelo": modelo, "features": FEATURES_VIVO,
            "metricas": metricas, "tipo": "vivo", "terreno": terreno}


def disimilitud(info_modelo, vector_base):
    """Distancia del contexto actual al terreno aprendido por el cerebro
    en vivo: raiz del promedio de z-scores al cuadrado sobre las 26
    variables base. Devuelve {"distancia", "umbral", "desconocido"} o
    None si el modelo no trae estadisticas de terreno (entrenado antes
    de la v3.15: el termometro se activa en su proximo reentrenamiento,
    que ocurre solo cada 20 operaciones nuevas)."""
    if not info_modelo:
        return None
    t = info_modelo.get("terreno")
    if not t:
        return None
    try:
        x = np.nan_to_num(np.array(list(vector_base)[:len(FEATURES_BASE)], dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        media = np.array(t["media"], dtype=float)
        desv = np.array(t["desv"], dtype=float)
        z = (x - media) / desv
        dist = float(np.sqrt((z ** 2).mean()))
        z_max = float(np.abs(z).max())
        umbral = float(t["umbral"])
        # Desconocido si el conjunto se aleja (distancia RMS sobre el
        # percentil 90 del entrenamiento) O si UNA sola variable esta en
        # un valor que el entrenamiento practicamente nunca vio
        # (|z| > 4,5: en ~200 ejemplos x 26 variables ocurre ~0 veces);
        # sin esta segunda regla, un extremo en una variable se diluye
        # al promediar las 26.
        return {"distancia": round(dist, 2), "umbral": round(umbral, 2),
                "z_max": round(z_max, 1),
                "desconocido": bool(dist > umbral or z_max > 4.5)}
    except Exception:
        return None

def cargar_modelo_vivo(ruta_pkl="modelo_vivo.pkl"):
    """Como cargar_modelo pero para el modelo en vivo (otras variables)."""
    try:
        with open(ruta_pkl, "rb") as f:
            info = pickle.load(f)
        if info.get("features") != FEATURES_VIVO:
            return None
        return info
    except Exception:
        return None

def prob_ganada_vivo(info_modelo, vector_base, es_call):
    """Probabilidad de GANADA segun el modelo en vivo (variables extendidas)."""
    if info_modelo is None:
        return None
    try:
        X = np.array([_extender_vivo(vector_base, es_call)], dtype=float)
        proba = info_modelo["modelo"].predict_proba(X)[0]
        clases = list(info_modelo["modelo"].classes_)
        return float(proba[clases.index(1)])
    except Exception:
        return None

def guardar_modelo(info, ruta_pkl="modelo_v3.pkl", ruta_metricas="modelo_v3_metricas.json"):
    with open(ruta_pkl, "wb") as f:
        pickle.dump(info, f)
    with open(ruta_metricas, "w") as f:
        json.dump(info["metricas"], f, indent=2, ensure_ascii=False)

def cargar_modelo(ruta_pkl="modelo_v3.pkl"):
    try:
        with open(ruta_pkl, "rb") as f:
            info = pickle.load(f)
        if info.get("features") != FEATURES_COMPLETAS:
            return None
        return info
    except Exception:
        return None

def prob_ganada(info_modelo, vector_base, es_call):
    """Probabilidad calibrada de GANADA para la direccion indicada."""
    if info_modelo is None:
        return None
    try:
        X = np.array([list(vector_base) + [1 if es_call else 0]], dtype=float)
        proba = info_modelo["modelo"].predict_proba(X)[0]
        clases = list(info_modelo["modelo"].classes_)
        return float(proba[clases.index(1)])
    except Exception:
        return None

# ---------------- Descarga historica Binance (backtesting) ----------------

def descargar_historico_binance(symbol, dias, sesion=None):
    """Descarga velas 1m de Binance paginando de 1000 en 1000. Devuelve df OHLCV."""
    import requests
    import time as _time
    sesion = sesion or requests.Session()
    fin_ms = int(_time.time() * 1000)
    inicio_ms = fin_ms - int(dias) * 24 * 60 * 60 * 1000
    filas = []
    cursor = inicio_ms
    while cursor < fin_ms:
        r = sesion.get("https://data-api.binance.vision/api/v3/klines",
                       params={"symbol": symbol, "interval": "1m", "limit": 1000,
                               "startTime": cursor, "endTime": fin_ms},
                       timeout=15)
        datos = r.json()
        if not isinstance(datos, list):
            raise RuntimeError(f"Binance no devolvio velas para {symbol}: {str(datos)[:120]}")
        if not datos:
            break
        filas.extend(datos)
        cursor = int(datos[-1][0]) + 60_000
        if len(datos) < 1000:
            break
    if not filas:
        return None
    df = pd.DataFrame(filas, columns=["timestamp", "open", "high", "low", "close", "volume",
                                      "close_time", "quote_asset_volume", "number_of_trades",
                                      "taker_buy_base", "taker_buy_quote", "ignore"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]].dropna()

# ============================================================
# MEMORIA DE PATRONES + CALIBRACION EMPIRICA (v3.18)
# ============================================================
# Lo que Jose pidio el 2026-11-11: que la app aprenda de TODAS las
# perdidas Y ganancias mirando la accion del precio que enseñan los
# cursos de trading: rompimientos, velas martillo (pin bar), respeto
# del nivel anterior, envolventes, mechas de rechazo. Esas variables YA
# se guardan con cada operacion (v3.2); lo que faltaba era MEDIR que
# contextos precedieron victorias y cuales precedieron derrotas, y
# dejar que esa medida mueva el nivel de las señales nuevas.
# La medida del historial real (995 ops, 2026-11-11) sorprendio: los
# patrones de rechazo A FAVOR perdian mucho (pin bar a favor 20,0% en
# 30 ops; gatillo de rechazo 22,6% en 53 ops; mecha larga a favor
# 20,5% en 44 ops), mientras el pin bar EN CONTRA ganaba 79,2% (24
# ops). En velas de 1 minuto el rechazo clasico se comporta al reves:
# el impulso manda. Por eso la memoria NO trae reglas fijas de libro:
# trae lo que MIDIO en las operaciones de ESTA app, con muestra minima
# y suavizado hacia el acierto global, y se corrige sola si el
# comportamiento cambia al acumular mas operaciones.

MIN_MUESTRA_CONTEXTO = 20   # un contexto "habla" solo con 20+ ops decisivas
K_SUAVIZADO = 20            # el acierto global pesa como 20 ops previas
UMBRAL_SUBE_MEMORIA = 53.8  # punto de equilibrio IQ Option (payout 86%)
UMBRAL_BAJA_MEMORIA = 40.0  # por debajo de esto, el contexto castiga


def _decisivas_para_aprender(historial):
    """Decisivas con variables completas; prefiere etiquetas limpias
    cuando hay >=60 (la misma regla del modelo en vivo, v3.8)."""
    ops = ops_validas_para_vivo(historial)
    limpias = [h for h in ops if h.get("fuente_datos") in ("Twelve Data", "Binance")]
    return limpias if len(limpias) >= MIN_OPS_MODELO_VIVO else ops


def _fv_de_op(h):
    """(dict de variables por nombre, es_call) de una operacion guardada."""
    f = h["features"]
    fv = dict(zip(FEATURES_BASE, f[:len(FEATURES_BASE)]))
    return fv, f[-1] == 1


def _contextos_de(fv, es_call):
    """Contextos de accion del precio de la vela actual. Las claves son
    estables: la misma funcion clasifica el historial guardado y la
    señal nueva, asi la memoria y la aplicacion nunca discrepan."""
    s = 1.0 if es_call else -1.0
    pin = float(fv.get("pin_bar", 0) or 0)
    env = float(fv.get("envolvente", 0) or 0)
    rup = float(fv.get("ruptura", 0) or 0)
    mecha_sup = float(fv.get("mecha_sup", 0) or 0)
    mecha_inf = float(fv.get("mecha_inf", 0) or 0)
    # OJO: distancia 0.0 = precio PEGADO al nivel; no usar "or 99" porque
    # 0.0 es falsy y lo mandaria a "lejos". Solo None (sin dato) es 99.
    d_res = fv.get("dist_resistencia", 99)
    d_res = 99.0 if d_res is None else float(d_res)
    d_sop = fv.get("dist_soporte", 99)
    d_sop = 99.0 if d_sop is None else float(d_sop)
    inside = float(fv.get("inside_bar", 0) or 0)

    def _lado(v):
        return "a favor" if v * s > 0 else ("en contra" if v * s < 0 else "ausente")

    dmin = min(d_res, d_sop)
    mecha_fav = mecha_inf if es_call else mecha_sup
    if dmin <= 0.5:
        nivel = "cerca + rechazo" if mecha_fav >= 0.5 else "cerca sin rechazo"
    elif dmin >= 1.5:
        nivel = "lejos de niveles"
    else:
        nivel = "distancia media"
    return {
        "pin bar": _lado(pin),
        "envolvente": _lado(env),
        "ruptura": _lado(rup),
        "gatillo": "con gatillo" if _gatillo_rechazo(fv, es_call) else "sin gatillo",
        "nivel": nivel,
        "mecha": "mecha larga a favor" if mecha_fav >= 0.5 else "sin mecha larga",
        "inside": "inside bar" if inside == 1 else "no inside",
    }


def memoria_patrones(historial):
    """Tabla contexto -> {n, ganadas, acierto, ajustado, habla}.

    El acierto AJUSTADO suaviza hacia el acierto global (prior de
    K_SUAVIZADO operaciones): un contexto con pocas operaciones no
    puede disparar un ajuste por si solo, y uno con muchas manda.
    Devuelve {"global", "n_ops", "contextos"}; las claves de contextos
    son tuplas (nombre, clave). Solo se calcula en memoria, nunca se
    guarda: siempre refleja el historial actual."""
    ops = _decisivas_para_aprender(historial)
    if not ops:
        return {"global": None, "n_ops": 0, "contextos": {}}
    ganadas = sum(1 for h in ops if h["resultado"] == "GANADA")
    wr_global = ganadas / len(ops)
    celdas = {}
    for h in ops:
        fv, es_call = _fv_de_op(h)
        for nombre, clave in _contextos_de(fv, es_call).items():
            c = celdas.setdefault((nombre, clave), [0, 0])
            c[0] += 1
            c[1] += 1 if h["resultado"] == "GANADA" else 0
    contextos = {}
    for k, (n, wins) in celdas.items():
        ajustado = (wins + K_SUAVIZADO * wr_global) / (n + K_SUAVIZADO)
        contextos[k] = {"n": n, "ganadas": wins,
                        "acierto": round(wins / n * 100, 1),
                        "ajustado": round(ajustado * 100, 1),
                        "habla": n >= MIN_MUESTRA_CONTEXTO}
    return {"global": round(wr_global * 100, 1), "n_ops": len(ops),
            "contextos": contextos}


def aplicar_memoria(nivel, fv, direccion, memoria):
    """Ajusta el nivel de una señal nueva segun la memoria de patrones:
    como maximo UN escalon, en un solo sentido (el contexto mas extremo
    que hable). Nunca bloquea ni cambia la direccion. Devuelve
    (nivel_ajustado, [motivos])."""
    if not memoria or not memoria.get("contextos") or nivel not in ("Baja", "Media", "Alta"):
        return nivel, []
    orden = ["Baja", "Media", "Alta"]
    idx = orden.index(nivel)
    ctx = _contextos_de(fv, direccion == "CALL")
    peor = None
    mejor = None
    for nombre, clave in ctx.items():
        celda = memoria["contextos"].get((nombre, clave))
        if not celda or not celda["habla"]:
            continue
        aj = celda["ajustado"]
        if peor is None or aj < peor[0]:
            peor = (aj, nombre, clave, celda)
        if mejor is None or aj > mejor[0]:
            mejor = (aj, nombre, clave, celda)
    motivos = []
    if peor is not None and peor[0] <= UMBRAL_BAJA_MEMORIA:
        if idx > 0:
            idx -= 1
            motivos.append(f"{peor[1]} '{peor[2]}' ganó {peor[3]['acierto']}% en {peor[3]['n']} ops reales")
    elif mejor is not None and mejor[0] >= UMBRAL_SUBE_MEMORIA:
        if idx < 2:
            idx += 1
            motivos.append(f"{mejor[1]} '{mejor[2]}' ganó {mejor[3]['acierto']}% en {mejor[3]['n']} ops reales")
    return orden[idx], motivos


# Tramos de la confianza MOSTRADA (la que ve Jose en la tarjeta). La
# calibracion responde: "las señales que mostraron X% ¿cuanto ganaron
# de verdad?" Mezcla las dos procedencias (probabilidad del modelo y
# puntaje) a proposito: calibra el numero que ve el usuario.
TRAMOS_CONFIANZA = [(0, 50, "<50"), (50, 55, "50-54"), (55, 60, "55-59"),
                    (60, 65, "60-64"), (65, 70, "65-69"), (70, 101, "70+")]


def _tramo_de(confianza):
    try:
        c = int(float(confianza))
    except (TypeError, ValueError):
        c = 50
    for lo, hi, etiqueta in TRAMOS_CONFIANZA:
        if lo <= c < hi:
            return etiqueta
    return "70+"


def calibracion_confianza(historial):
    """Tramo de confianza mostrada -> acierto medido (con suavizado).
    Devuelve {"global", "n_ops", "tramos"}; cada tramo trae n, ganadas,
    acierto, ajustado y habla (20+ ops)."""
    ops = _decisivas_para_aprender(historial)
    if not ops:
        return {"global": None, "n_ops": 0, "tramos": {}}
    ganadas = sum(1 for h in ops if h["resultado"] == "GANADA")
    wr_global = ganadas / len(ops)
    celdas = {}
    for h in ops:
        etiqueta = _tramo_de(h.get("confianza"))
        c = celdas.setdefault(etiqueta, [0, 0])
        c[0] += 1
        c[1] += 1 if h["resultado"] == "GANADA" else 0
    tramos = {}
    for etiqueta, (n, wins) in celdas.items():
        ajustado = (wins + K_SUAVIZADO * wr_global) / (n + K_SUAVIZADO)
        tramos[etiqueta] = {"n": n, "ganadas": wins,
                            "acierto": round(wins / n * 100, 1),
                            "ajustado": round(ajustado * 100, 1),
                            "habla": n >= MIN_MUESTRA_CONTEXTO}
    return {"global": round(wr_global * 100, 1), "n_ops": len(ops),
            "tramos": tramos}


def nivel_calibrado(confianza, historial):
    """Nivel desde el acierto MEDIDO del tramo de confianza, o (None, "")
    si el tramo no tiene muestra suficiente (entonces la app usa sus
    reglas de siempre). Alta = el tramo gano >=53,8% (equilibrio);
    Media = >=45%; si no, Baja. El numero de confianza que se muestra
    y se guarda NO se toca: calibrar el numero guardado realimentaria
    la tabla y la corromperia."""
    cal = calibracion_confianza(historial)
    etiqueta = _tramo_de(confianza)
    celda = cal["tramos"].get(etiqueta)
    if not celda or not celda["habla"]:
        return None, ""
    aj = celda["ajustado"]
    nivel = "Alta" if aj >= 53.8 else ("Media" if aj >= 45 else "Baja")
    nota = (f"Calibrado: las señales con confianza {etiqueta}% ganaron "
            f"{celda['acierto']}% en {celda['n']} ops reales")
    return nivel, nota
