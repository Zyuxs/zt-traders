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
    df["hora"] = _horas(df)

    # --- v3.2: accion del precio, niveles y regimen ---
    apertura = df["open"]
    max_cuerpo = np.maximum(apertura, close)
    min_cuerpo = np.minimum(apertura, close)
    cuerpo_ratio = (close - apertura).abs() / rango_vela
    df["mecha_sup"] = ((high - max_cuerpo) / rango_vela).clip(0, 1)
    df["mecha_inf"] = ((min_cuerpo - low) / rango_vela).clip(0, 1)

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
