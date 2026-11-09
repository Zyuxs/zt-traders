# ============================================================
# ZT TRADERS INSTITUTIONAL PRO - v3 (backtesting + señales filtradas)
# Requiere el archivo nucleo_ia.py en la misma carpeta/repo.
# Misma app, mismas fases y misma interfaz. Cambios internos:
#  1) La zona dinamica es un fragmento que se actualiza cada 1s:
#     la pagina completa (grafico TradingView, encabezado, selector)
#     YA NO se recarga cada 0.3-1s.
#  2) Datos de mercado con cache (Binance/Yahoo) y una sola sesion HTTP.
#  3) El modelo ML se entrena solo cuando hay datos nuevos suficientes
#     (antes se reentrenaba en CADA analisis, leyendo el historial de disco).
#  4) Bug de confianza ML corregido: antes, si el modelo predecía PERDIDA
#     con mucha seguridad, eso SUBIA la confianza de la señal.
#     Ahora se usa directamente la probabilidad de GANADA.
#  5) Señal por puntaje (RSI + Bollinger + tendencia), sin el sesgo a CALL
#     de la regla anterior ("RSI <= 50 o precio en banda baja = CALL").
#  6) Sin datos suficientes ya no se inventa un CALL al 65%: devuelve ESPERA
#     y esa operacion no se registra ni contamina el aprendizaje.
#  7) Historial y patrones se cargan una sola vez en memoria y se guardan
#     con escritura atomica (antes se leian de disco en cada recarga).
#  8) La clave maestra y el token admin ya NO estan escritos en el codigo:
#     se leen de Secrets (zt_master_key / zt_admin_token) o de las variables
#     de entorno ZT_MASTER_KEY / ZT_ADMIN_TOKEN.
#  9) Reloj y ciclo en hora de Caracas (el servidor de Streamlit Cloud
#     trabaja en UTC; antes el reloj mostrado no era tu hora local).
# 10) Si la ventana de cierre (seg 57-59) se pierde por una llamada lenta,
#     la operacion se cierra al iniciar el siguiente ciclo, no se pierde.
# 11) v2.1: el HTML de la zona dinamica se limpia de indentacion antes de
#     mostrarse (limpiar_html). Sin eso, Markdown lo trataba como bloque de
#     codigo y la caja de la señal se veia como texto crudo <div>...</div>.
# --- Novedades v3 ---
# 12) Aprendizaje real en nucleo_ia.py: 15 variables por vela (tendencia,
#     volatilidad, impulso, volumen, hora), no solo RSI + Bollinger.
# 13) El modelo se entrena con BACKTESTING desde el panel Admin (descarga
#     meses de velas Binance y genera miles de ejemplos etiquetados), con
#     division por tiempo y probabilidad CALIBRADA.
# 14) Señales FILTRADAS: con modelo, solo se emite si la probabilidad
#     calibrada >= 62%; sin modelo, solo si el puntaje tecnico >= 3.
#     El resto de los ciclos la app dice ESPERA y no opera.
# 15) El modelo entrenado se guarda en modelo_v3.pkl y se aplica solo a
#     crypto; forex opera con el filtro de puntaje (sin modelo, avisado).
# --- Ajustes v3.1 (a partir de la aclaracion de Jose: operar en IQ Option) ---
# 16) La direccion la ELIGE el modelo comparando la probabilidad de CALL y
#     de PUT y quedandose con la mayor (antes solo validaba la direccion del
#     puntaje). Si el modelo dice PUT, la señal es PUT (BAJA en IQ Option).
# 17) Candado de datos frescos: si la ultima vela es vieja (mercado cerrado,
#     fin de semana forex / OTC), la app da ESPERA en vez de operar con
#     precios que ya no representan al mercado.
# --- Novedades v3.2: conocimiento de trader profesional ---
# 18) Nuevas variables en nucleo_ia: accion del precio (envolvente, pin bar,
#     inside bar, mechas de rechazo), distancia a soportes/resistencias en
#     ATR, rupturas, posicion en el rango y REGIMEN de mercado (25 variables).
#     OJO: al cambiar las variables, el modelo_v3.pkl anterior queda
#     incompatible y la app lo ignora: hay que REENTRENAR desde el panel.
# 19) Regla de confluencia: la señal solo sale con >= 2 confirmaciones
#     independientes (tendencia, nivel S/R, patron de vela, impulso) y sin
#     bloqueos duros (pegado al nivel contrario o contra tendencia fuerte).
# 20) El entrenamiento ahora tambien mide precision por REGIMEN, para saber
#     en que tipo de mercado gana o pierde el cerebro (su "diario").
# --- Correccion v3.3 (feedback de Jose: una noche entera sin señales) ---
# 21) SEÑAL SIEMPRE: en cada ciclo de 2 minutos el cerebro emite CALL o PUT
#     (la direccion mas probable). Los filtros de la v3.2 APILADOS (62% +
#     confluencia >= 2 + sin bloqueos + puntaje >= 3) dejaban pasar casi
#     nada: las probabilidades calibradas de 1 minuto rara vez llegan a 62%.
# 22) La calidad ahora se GRADUA, no se bloquea: cada señal trae NIVEL
#     (Alta / Media / Baja) y advertencias. La confluencia y los bloqueos
#     suben o bajan el nivel y se muestran como avisos, no como veto.
# 23) ESPERA queda SOLO para cuando no hay datos utilizables (insuficientes
#     o desactualizados): sin datos no hay analisis posible.
#     No hay que reentrenar: las variables y el modelo no cambian.
# --- v3.4: forex sin señales + capa de razonamiento IA (gratis en prueba) ---
# 24) BUG FOREX: EUR/USD y similares no daban señales. La descarga de Yahoo
#     via yfinance devolvia vacio o pocas velas desde la nube, y el indice
#     podia venir sin zona horaria (la edad del dato se calculaba mal y el
#     candado de frescura bloqueaba todo). Ahora: descarga normalizada a
#     UTC y RESPALDO por la API grafica de Yahoo (query2.finance.yahoo.com),
#     y el ESPERA muestra cuantas velas llegaron (diagnostico visible).
# 25) Capa de razonamiento (razonamiento_ia.py, opcional y apagada sin
#     clave): un LLM revisa cada señal como trader senior (CONFIRMA /
#     SUBE / REBAJA el NIVEL, jamas veta ni cambia la direccion), resume el
#     clima de la sesion y analiza el historial en el panel Admin (diario
#     inteligente). Proveedores con nivel GRATUITO para la fase de prueba:
#     Gemini (Google AI Studio), Groq y OpenRouter; se configuran solo
#     desde Secrets (ia_proveedor, ia_api_key, ia_segunda_opinion...).
#     Sin clave o con la API caida, la app funciona igual que la v3.3.
# --- v3.5: APRENDIZAJE EN VIVO (lo que pide Jose: que aprenda operando) ---
# 26) Cada operacion se registra con par, tipo, nivel y sus 26 variables, y
#     la app ya juzga su resultado con el precio real de entrada y el de
#     cierre (sabe si gano o perdio). Con >= 60 operaciones reales se
#     entrena SOLO un modelo en vivo (nucleo_ia.entrenar_modelo_vivo) y se
#     reentrena cada 20 operaciones nuevas. Es el cerebro de FOREX (donde
#     no hay backtest de Binance) y respaldo del de crypto.
# 27) Panel "Aprendizaje en vivo" visible para todos: progreso de
#     aprendizaje, acierto real global y por par / direccion / nivel /
#     tipo, y el estado del cerebro en vivo con su precision medida en
#     operaciones que no vio al entrenar. Asi se VE si esta aprendiendo.
# --- v3.6: TWELVE DATA para forex (observacion de Jose) ---
# 29) Jose noto que Yahoo Finance va retrasado frente a IQ Option
#     (~3 pips en EUR/USD en su observacion) y que Twelve Data va parejo
#     con IQ Option. Forex ahora se analiza y se juzga (entrada/cierre)
#     con Twelve Data como fuente principal y Yahoo de respaldo; la
#     razon de cada señal dice la fuente ("Datos: Twelve Data"). Clave
#     en Secrets: twelvedata_api_key. Incluye todo lo de la v3.5.1.
# --- v3.5.1: RESPALDO DEL APRENDIZAJE (sin cambiar nada de la v3.5) ---
# 28) Panel Admin "Respaldo del aprendizaje": botones para descargar
#     historial.json, patrones.json y modelo_vivo.pkl. El aprendizaje vive
#     solo en el disco del servidor de Streamlit y se puede perder al
#     reiniciar o volver a subir la app; con la copia guardada, subir ese
#     archivo junto a la app lo restaura (el cerebro en vivo se reentrena
#     solo desde el historial restaurado al llegar a 60 operaciones).
# ============================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import yfinance as yf
import ta
import json
import os
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import nucleo_ia
import razonamiento_ia

st.set_page_config(page_title="ZT TRADERS INSTITUTIONAL PRO", layout="centered", page_icon="⚡")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"
ARCHIVO_LICENCIAS = "licencias.json"
ZONA_HORARIA = ZoneInfo("America/Caracas")

def ahora_local():
    return datetime.now(ZONA_HORARIA)

# ---------------- Utilidades ----------------

@st.cache_resource
def sesion_http():
    s = requests.Session()
    s.headers.update({"User-Agent": "ZT-Traders/2.0"})
    return s

def _secret(nombre, env_nombre):
    try:
        valor = st.secrets.get(nombre, None)
        if valor:
            return valor
    except Exception:
        pass
    return os.getenv(env_nombre)

def master_keys_configuradas():
    valor = _secret("zt_master_key", "ZT_MASTER_KEY")
    if not valor:
        return []
    if isinstance(valor, (list, tuple)):
        return [str(v).strip() for v in valor if str(v).strip()]
    return [v.strip() for v in str(valor).split(",") if v.strip()]

def admin_token_configurado():
    valor = _secret("zt_admin_token", "ZT_ADMIN_TOKEN")
    return str(valor) if valor else None

def guardar_json_atomico(archivo, datos):
    tmp = archivo + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(datos, f, indent=2)
        os.replace(tmp, archivo)
        return True
    except Exception:
        return False

def leer_json(archivo, defecto):
    if os.path.exists(archivo):
        try:
            with open(archivo, "r") as f:
                return json.load(f)
        except Exception:
            return defecto
    return defecto

# ---------------- Licencias ----------------

def cargar_licencias():
    return leer_json(ARCHIVO_LICENCIAS, {})

def guardar_licencias(licencias):
    guardar_json_atomico(ARCHIVO_LICENCIAS, licencias)

def licencia_vigente(datos):
    """True/False si se puede comprobar la fecha; None si la fecha es invalida."""
    try:
        return ahora_local().date() <= datetime.strptime(datos["vence"], "%Y-%m-%d").date()
    except Exception:
        return None

if "licencia_activa" not in st.session_state:
    st.session_state.licencia_activa = False
if "usuario_actual" not in st.session_state:
    st.session_state.usuario_actual = ""

params = st.query_params
device_token = params.get("zt_token", None)
admin_token = admin_token_configurado()

if device_token and not st.session_state.licencia_activa:
    if admin_token and device_token == admin_token:
        st.session_state.licencia_activa = True
        st.session_state.usuario_actual = "👑 Admin Master"
    else:
        licencias = cargar_licencias()
        for k, v in licencias.items():
            if v.get("device_id") == device_token and licencia_vigente(v):
                st.session_state.licencia_activa = True
                st.session_state.usuario_actual = f"Licencia ({k[:8]}...)"
                break

if not st.session_state.licencia_activa:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    body { background: #04060d; color: #eef2ff; font-family: 'Inter', Arial, sans-serif; }
    .stApp { background: radial-gradient(900px 500px at 50% 0%, #12224a 0%, #04060d 70%) !important; }
    .login-container { max-width: 330px; margin: 70px auto; padding: 30px 25px; background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02)); border-radius: 20px; border: 1px solid rgba(140,170,255,0.30); text-align: center; box-shadow: 0 18px 44px rgba(0,0,0,0.5); }
    .login-title { font-size: 25px; font-weight: 900; letter-spacing: 3px; background: linear-gradient(90deg, #7dd3fc, #a5b4fc, #f0abfc); -webkit-background-clip: text; background-clip: text; color: transparent; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-container"><div class="login-title">⚡ ZT TRADERS</div><div style="color:#aaa; font-size:11px;">Sistema de Acceso Privado</div>', unsafe_allow_html=True)
    with st.form("login_form"):
        key_input = st.text_input("License Key", type="password")
        submit_key = st.form_submit_button("🚀 CONECTAR", use_container_width=True)
        if submit_key:
            claves_master = master_keys_configuradas()
            if claves_master and key_input in claves_master:
                st.session_state.licencia_activa = True
                st.session_state.usuario_actual = "👑 Admin Master"
                if admin_token:
                    st.query_params["zt_token"] = admin_token
                st.rerun()
            else:
                licencias = cargar_licencias()
                if key_input in licencias:
                    datos = licencias[key_input]
                    vigente = licencia_vigente(datos)
                    if vigente is False:
                        st.error("❌ Key expirada.")
                        st.stop()
                    if vigente is None:
                        st.error("❌ Key con fecha inválida. Contacta al administrador.")
                        st.stop()
                    if datos.get("device_id") is None:
                        datos["device_id"] = str(uuid.uuid4())
                        licencias[key_input] = datos
                        guardar_licencias(licencias)
                    st.query_params["zt_token"] = datos["device_id"]
                    st.session_state.licencia_activa = True
                    st.session_state.usuario_actual = f"Licencia ({key_input[:8]}...)"
                    st.rerun()
                else:
                    st.error("❌ Key no encontrada.")
    if not master_keys_configuradas():
        st.caption("Admin: configura zt_master_key y zt_admin_token en Secrets para habilitar el acceso maestro.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------- Estado persistente (historial / patrones) ----------------
# Se cargan UNA vez por sesion y se mantienen en memoria.
# Antes: cargar_historial()/cargar_patrones() leian disco en cada recarga (varias por segundo).

PATRONES_BASE = {"total_ops": 0, "total_ganadas": 0, "winrate": 0.0}

if "hist_cache" not in st.session_state:
    st.session_state.hist_cache = leer_json(ARCHIVO_HISTORIAL, [])
if "pesos" not in st.session_state:
    st.session_state.pesos = {**PATRONES_BASE, **leer_json(ARCHIVO_PATRONES, {})}

def resetear_memoria():
    for archivo in (ARCHIVO_HISTORIAL, ARCHIVO_PATRONES):
        if os.path.exists(archivo):
            os.remove(archivo)
    st.session_state.hist_cache = []
    st.session_state.pesos = dict(PATRONES_BASE)

def aprender(señal, resultado, indicadores, precio_real_entrada, features=None,
             par=None, tipo=None, nivel=None, confianza=None):
    """Registra el resultado una sola vez por operacion y actualiza memoria + disco."""
    hist = st.session_state.hist_cache
    pesos = st.session_state.pesos
    hist.append({
        "fecha": ahora_local().strftime("%H:%M:%S"),
        "señal": señal,
        "resultado": resultado,
        "precio_entrada_real": precio_real_entrada,
        "indicadores": indicadores,
        "features": features,
        "par": par,
        "tipo": tipo,
        "nivel": nivel,
        "confianza": confianza,
    })
    pesos["total_ops"] += 1
    if resultado == "GANADA":
        pesos["total_ganadas"] += 1
    pesos["winrate"] = (pesos["total_ganadas"] / pesos["total_ops"] * 100) if pesos["total_ops"] > 0 else 0.0
    guardar_json_atomico(ARCHIVO_HISTORIAL, hist)
    guardar_json_atomico(ARCHIVO_PATRONES, pesos)
    quizas_entrenar_vivo()

# ---------------- Modelo IA v3 (entrenado con backtesting) ----------------

ARCHIVO_MODELO_V3 = "modelo_v3.pkl"

@st.cache_resource
def modelo_v3_info():
    """Carga modelo_v3.pkl si existe (se crea desde el panel Admin)."""
    return nucleo_ia.cargar_modelo(ARCHIVO_MODELO_V3)

# ---------------- Modelo EN VIVO (aprende de las operaciones reales) ----------------

ARCHIVO_MODELO_VIVO = "modelo_vivo.pkl"

@st.cache_resource
def modelo_vivo_info():
    """Carga modelo_vivo.pkl: el cerebro que aprende de las operaciones
    reales registradas por la app (entrada, salida, gano/perdio)."""
    return nucleo_ia.cargar_modelo_vivo(ARCHIVO_MODELO_VIVO)

def quizas_entrenar_vivo():
    """Reentrena el modelo en vivo cuando hay operaciones nuevas
    suficientes. Se llama tras registrar cada operacion; entrenar con
    cientos de filas toma ~1 segundo y solo ocurre cada 20 operaciones."""
    try:
        hist = st.session_state.hist_cache
        validas = len(nucleo_ia.ops_validas_para_vivo(hist))
        if validas < nucleo_ia.MIN_OPS_MODELO_VIVO:
            return
        info = modelo_vivo_info()
        entrenadas = info["metricas"]["ejemplos_entrenamiento"] if info else 0
        if info is not None and validas - entrenadas < nucleo_ia.REENTRENAR_CADA_OPS:
            return
        nuevo = nucleo_ia.entrenar_modelo_vivo(hist)
        if nuevo is not None:
            nucleo_ia.guardar_modelo(nuevo, ARCHIVO_MODELO_VIVO, "modelo_vivo_metricas.json")
            modelo_vivo_info.clear()
    except Exception:
        pass

# ---------------- Datos de mercado (con cache) ----------------

@st.cache_data(ttl=20, show_spinner=False)
def obtener_datos_binance(symbol, interval="1m", limit=50):
    try:
        r = sesion_http().get(
            "https://data-api.binance.vision/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        datos = r.json()
        if not isinstance(datos, list) or not datos:
            return None
        df = pd.DataFrame(datos, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.dropna(subset=["close"])
    except Exception:
        return None

def _normalizar_yahoo(datos):
    """Deja el DataFrame de Yahoo con columnas open/high/low/close/volume y
    el indice en UTC con zona horaria (sin esto, la edad del dato se
    calculaba mal y el candado de frescura bloqueaba el forex)."""
    if datos is None or len(datos) == 0:
        return None
    if isinstance(datos.columns, pd.MultiIndex):
        datos.columns = datos.columns.get_level_values(0)
    datos = datos.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Volume": "volume"})
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in datos.columns]
    datos = datos[cols].dropna(subset=["close"]).copy()
    idx = pd.to_datetime(datos.index)
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("America/New_York")
    datos.index = idx.tz_convert("UTC")
    return datos.sort_index()

@st.cache_data(ttl=20, show_spinner=False)
def _descargar_yahoo(symbol, period, interval="1m"):
    try:
        datos = yf.download(symbol, period=period, interval=interval, progress=False)
        return _normalizar_yahoo(datos)
    except Exception:
        return None

@st.cache_data(ttl=20, show_spinner=False)
def _yahoo_chart_api(symbol, rango="2d"):
    """Respaldo: API grafica de Yahoo (la que usa su web). yfinance a veces
    devuelve vacio desde IPs de nube; esta ruta suele responder igual."""
    try:
        import urllib.parse as _up
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{_up.quote(symbol, safe='')}?interval=1m&range={rango}"
        r = sesion_http().get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if r.status_code != 200:
            return None
        res = (r.json().get("chart", {}).get("result") or [None])[0]
        if not res or not res.get("timestamp"):
            return None
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open": q.get("open"), "high": q.get("high"), "low": q.get("low"),
            "close": q.get("close"), "volume": q.get("volume") or [0] * len(res["timestamp"]),
        }, index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
        return df.dropna(subset=["close"]).sort_index()
    except Exception:
        return None

def obtener_datos_yahoo(symbol):
    # 1) yfinance 1d; 2) yfinance 2d; 3) API grafica de Yahoo.
    df = _descargar_yahoo(symbol, "1d")
    if df is None or len(df) < 60:
        df2 = _descargar_yahoo(symbol, "2d")
        if df2 is not None and (df is None or len(df2) > len(df)):
            df = df2
    if df is None or len(df) < 60:
        df3 = _yahoo_chart_api(symbol, "2d")
        if df3 is not None and (df is None or len(df3) > len(df)):
            df = df3
    return df

@st.cache_data(ttl=3, show_spinner=False)
def _precio_binance(symbol):
    try:
        return float(sesion_http().get(f"https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}", timeout=5).json()["price"])
    except Exception:
        return None

@st.cache_data(ttl=3, show_spinner=False)
def _precio_yahoo(symbol):
    # Descarga propia con ttl corto: NO reutiliza la descarga de analisis (ttl=20),
    # porque el precio de entrada/cierre necesita ser lo mas fresco posible.
    try:
        datos = yf.download(symbol, period="1d", interval="1m", progress=False)
        datos = _normalizar_yahoo(datos)
        if datos is not None and len(datos) > 0:
            return float(datos["close"].iloc[-1])
    except Exception:
        pass
    df = _yahoo_chart_api(symbol, "1d")
    if df is not None and len(df) > 0:
        return float(df["close"].iloc[-1])
    return None

# ---------------- Twelve Data: fuente principal de FOREX (v3.6) ----------------
# Observacion de Jose (2026-11-09): Yahoo Finance va retrasado frente a
# IQ Option (el vio ~3 pips de diferencia en EUR/USD) y Twelve Data va
# parejo con IQ Option. En operaciones de 1 minuto, 3 pips voltean el
# resultado: por eso forex ahora se analiza y se juzga con Twelve Data.
# Plan Basic de Jose: 800 creditos/dia, 8/min. Cada llamada = 1 credito y
# el cache de Streamlit se comparte entre sesiones: la serie se pide con
# ttl 45s y el precio con ttl 10s. Si no hay clave, se acaban los
# creditos o la API falla, la app cae a Yahoo automaticamente.

def _twelvedata_key():
    return _secret("twelvedata_api_key", "TWELVEDATA_API_KEY")

def _simbolo_twelvedata(codigo_yahoo):
    base = str(codigo_yahoo).replace("=X", "")
    return f"{base[:3]}/{base[3:6]}" if len(base) >= 6 else base

def _td_request(codigo_yahoo, outputsize):
    key = _twelvedata_key()
    if not key:
        return None
    try:
        r = sesion_http().get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": _simbolo_twelvedata(codigo_yahoo), "interval": "1min",
                    "outputsize": outputsize, "apikey": key, "format": "JSON",
                    "timezone": "UTC"},
            timeout=12)
        j = r.json()
        valores = j.get("values")
        if not valores:
            return None
        df = pd.DataFrame(valores)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        else:
            df["volume"] = 0.0
        return df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    except Exception:
        return None

@st.cache_data(ttl=45, show_spinner=False)
def _serie_twelvedata(codigo_yahoo, outputsize=120):
    return _td_request(codigo_yahoo, outputsize)

@st.cache_data(ttl=10, show_spinner=False)
def _precio_twelvedata(codigo_yahoo):
    df = _td_request(codigo_yahoo, 1)
    if df is not None and len(df):
        return float(df["close"].iloc[-1])
    return None

def obtener_datos_forex(codigo_par):
    """(df, fuente): Twelve Data primero; Yahoo de respaldo."""
    df = _serie_twelvedata(codigo_par, 120)
    if df is not None and len(df) >= 60:
        return df, "Twelve Data"
    dfy = obtener_datos_yahoo(codigo_par)
    return dfy, "Yahoo Finance"

def obtener_precio_actual(codigo_par, tipo="crypto"):
    if tipo == "crypto":
        return _precio_binance(codigo_par)
    precio = _precio_twelvedata(codigo_par)
    return precio if precio is not None else _precio_yahoo(codigo_par)

# ---------------- Analisis de señal (por puntaje) ----------------

def _edad_ultimo_dato_seg(df):
    """Segundos desde la ultima vela recibida; None si no se puede saber."""
    try:
        if "timestamp" in df.columns:
            ultimo = float(df["timestamp"].iloc[-1]) / 1000.0
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = pd.Timestamp(df.index[-1])
            if ts.tz is None:
                ts = ts.tz_localize("America/New_York")
            ultimo = ts.timestamp()
        else:
            return None
        return max(0.0, datetime.now(timezone.utc).timestamp() - ultimo)
    except Exception:
        return None

def analizar_par(codigo_par, tipo="crypto"):
    espera = {"señal": "ESPERA", "confianza": 0, "razon": "Datos insuficientes",
              "indicadores": {"rsi": 50, "precio_relativo": 0.5}, "features": None,
              "nivel": "", "advertencias": []}
    if tipo == "crypto":
        df = obtener_datos_binance(codigo_par, interval="1m", limit=120)
        fuente_datos = "Binance"
    else:
        df, fuente_datos = obtener_datos_forex(codigo_par)
    if df is None or len(df) < 60:
        n = 0 if df is None else len(df)
        return {"señal": "ESPERA", "confianza": 0,
                "razon": f"Datos insuficientes: el proveedor entregó {n} velas (se necesitan 60)",
                "indicadores": {"rsi": 50, "precio_relativo": 0.5}, "features": None,
                "nivel": "", "advertencias": []}

    # Unico ESPERA que queda: datos viejos = no hay mercado que analizar
    edad = _edad_ultimo_dato_seg(df)
    limite_edad = 180 if tipo == "crypto" else 420
    if edad is not None and edad > limite_edad:
        return {"señal": "ESPERA", "confianza": 0,
                "razon": f"Datos desactualizados ({int(edad // 60)} min) — mercado cerrado o proveedor atrasado",
                "indicadores": {"rsi": 50, "precio_relativo": 0.5}, "features": None,
                "nivel": "", "advertencias": []}

    try:
        d = nucleo_ia.agregar_indicadores(df)
        validas = nucleo_ia.filas_validas(d)
        if len(validas) == 0:
            return espera
        pos = int(validas[-1])
        fila = d.iloc[pos]
        precio = float(fila["close"])
        rsi = float(fila["_rsi"])
        bb_low = float(fila["_bb_low"]); bb_high = float(fila["_bb_high"])
        sma = float(fila["_sma20"])
        sma_prev = float(d["_sma20"].iloc[pos - 5]) if pos >= 5 else sma
        if precio <= 0:
            return espera

        direccion_puntaje, score, razones = nucleo_ia.direccion_por_puntaje(rsi, precio, bb_low, bb_high, sma, sma_prev)
        vector_base = nucleo_ia.fila_features(d, pos)
        fv = dict(zip(nucleo_ia.FEATURES_BASE, vector_base))
        regimen_nombre = nucleo_ia.nombre_regimen(fv.get("regimen_cod", 0))
        indicadores = {"rsi": rsi, "precio_relativo": float(fila["precio_relativo"])}

        # CON MODELO: elige direccion por probabilidad, SIEMPRE emite.
        # Crypto: modelo de backtest Binance si existe. Forex (y crypto sin
        # backtest): modelo EN VIVO, aprendido de las operaciones reales
        # registradas por la propia app (v3.5).
        info_modelo = modelo_v3_info() if tipo == "crypto" else None
        fuente_modelo = "backtest Binance" if info_modelo else ""
        if info_modelo is None:
            info_modelo = modelo_vivo_info()
            if info_modelo is not None:
                fuente_modelo = f"en vivo: {info_modelo['metricas']['ejemplos_entrenamiento']} operaciones reales"
        _prob_fn = nucleo_ia.prob_ganada_vivo if fuente_modelo.startswith("en vivo") else nucleo_ia.prob_ganada
        p_call = _prob_fn(info_modelo, vector_base, True) if info_modelo else None
        p_put = _prob_fn(info_modelo, vector_base, False) if info_modelo else None
        if p_call is not None and p_put is not None:
            es_call = p_call >= p_put
            direccion = "CALL" if es_call else "PUT"
            p = max(p_call, p_put)
            confianza = int(round(p * 100))
            conf, bloqueos, detalles = nucleo_ia.evaluar_confluencia(fv, direccion)
            advertencias = []
            if bloqueos:
                advertencias += [f"Bloqueo: {b}" for b in bloqueos]
            if conf < 2:
                advertencias.append(f"Confluencia baja ({conf}/2)")
            if p >= 0.58 and conf >= 2 and not bloqueos:
                nivel = "Alta"
            elif p >= 0.54:
                nivel = "Media"
            else:
                nivel = "Baja"
            if p < 0.538:
                advertencias.append("Prob. bajo el equilibrio de IQ Option (53,8% con payout 86%)")
            if direccion != direccion_puntaje:
                razones.append(f"La IA elige {direccion} (el puntaje decía {direccion_puntaje})")
            razones.append(f"IA: SUBE {int(round(p_call * 100))}% vs BAJA {int(round(p_put * 100))}%")
            razones.append(f"Fuente IA: {fuente_modelo}")
            razones.append(f"Régimen: {regimen_nombre} · Confluencia {conf}✓ · Nivel {nivel}")
            razones.append(f"Datos: {fuente_datos}")
            return {"señal": direccion, "confianza": confianza, "razon": " | ".join(razones),
                    "indicadores": indicadores, "features": vector_base + [1 if es_call else 0],
                    "nivel": nivel, "advertencias": advertencias, "fuente": fuente_datos}

        # SIN MODELO: direccion por puntaje (desempate por tendencia), SIEMPRE emite
        direccion = direccion_puntaje
        conf, bloqueos, detalles = nucleo_ia.evaluar_confluencia(fv, direccion)
        confianza = 50 + min(30, abs(score) * 6) + min(6, conf * 2) - (6 if bloqueos else 0)
        confianza = max(50, min(int(confianza), 90))
        advertencias = []
        if bloqueos:
            advertencias += [f"Bloqueo: {b}" for b in bloqueos]
        if conf < 2:
            advertencias.append(f"Confluencia baja ({conf}/2)")
        if confianza >= 70 and conf >= 2 and not bloqueos:
            nivel = "Alta"
        elif confianza >= 58:
            nivel = "Media"
        else:
            nivel = "Baja"
        razones.append(f"Régimen: {regimen_nombre} · Confluencia {conf}✓ · Nivel {nivel}")
        if tipo != "crypto":
            _validas = len(nucleo_ia.ops_validas_para_vivo(st.session_state.hist_cache))
            razones.append(f"Modelo en vivo aprendiendo: {_validas}/{nucleo_ia.MIN_OPS_MODELO_VIVO} operaciones reales")
        else:
            razones.append("Sin modelo IA entrenado")
        razones.append(f"Datos: {fuente_datos}")
        return {"señal": direccion, "confianza": confianza,
                "razon": " | ".join(razones), "indicadores": indicadores,
                "features": vector_base + [1 if direccion == "CALL" else 0],
                "nivel": nivel, "advertencias": advertencias, "fuente": fuente_datos}
    except Exception:
        return espera

# ---------------- Capa de razonamiento IA (v3.4, opcional) ----------------

def _ia_bool(nombre, defecto=False):
    v = _secret(nombre, nombre.upper())
    if v is None:
        return defecto
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "si", "sí", "yes", "on")

def ia_configurar_desde_secrets():
    # Blindado (v3.7.1): si el modulo razonamiento_ia.py no es el correcto o
    # falla por lo que sea, la capa IA se apaga en silencio y la app sigue
    # dando señales con normalidad. Nunca debe tumbar la zona dinamica.
    try:
        razonamiento_ia.configurar(
            proveedor=_secret("ia_proveedor", "IA_PROVEEDOR") or "gemini",
            api_key=_secret("ia_api_key", "IA_API_KEY") or "",
            modelo=_secret("ia_modelo", "IA_MODELO"),
            base_url=_secret("ia_base_url", "IA_BASE_URL"),
        )
        return razonamiento_ia.disponible()
    except Exception:
        return False

def ia_opinion_activa():
    return _ia_bool("ia_segunda_opinion") and ia_configurar_desde_secrets()

def ia_clima_activo():
    return _ia_bool("ia_clima") and ia_configurar_desde_secrets()

_NIVELES = ["Baja", "Media", "Alta"]

def _ajustar_nivel(nivel, veredicto):
    if nivel not in _NIVELES:
        return nivel
    i = _NIVELES.index(nivel)
    if veredicto == "SUBE":
        i = min(i + 1, 2)
    elif veredicto == "REBAJA":
        i = max(i - 1, 0)
    return _NIVELES[i]

def reproducir_voz(mensaje):
    # json.dumps escapa el texto de forma segura para meterlo en el JS.
    js_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            setTimeout(function() {{
                const msg = new SpeechSynthesisUtterance({json.dumps(mensaje)});
                msg.lang = 'es-ES';
                msg.rate = 1.0;
                msg.volume = 1.0;
                window.speechSynthesis.speak(msg);
            }}, 100);
        }}
    }})();
    </script>
    """
    components.html(js_code, height=0, width=0)

PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'}, 'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'}, 'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto'},
    'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto'}, 'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto'}, 'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto'}, 'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto'},
    'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex'}, 'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex'},
    'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex'}, 'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex'},
}

# ---------------- Estado dinamico ----------------

def _init_estado():
    valores = {
        'estado': 'analizando', 'par_sel': 'Bitcoin', 'senal': None,
        'resultado_guardado': False, 'precio_real_entrada': None,
        'voz_reproducida': False, 'ciclo_senal': None, 'ciclo_entrada': None,
        'opinion_ia': None, 'ciclo_opinion': None, 'clima_ia': '', 'clima_ts': 0.0,
    }
    for k, v in valores.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_estado()

def reiniciar_ciclo():
    st.session_state.senal = None
    st.session_state.estado = 'analizando'
    st.session_state.precio_real_entrada = None
    st.session_state.resultado_guardado = False
    st.session_state.voz_reproducida = False
    st.session_state.ciclo_senal = None
    st.session_state.ciclo_entrada = None

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 10px !important; padding-bottom: 0px !important; padding-left: 10px !important; padding-right: 10px !important; max-width: 100% !important; }
.stApp { background: radial-gradient(1200px 600px at 50% -10%, #12224a 0%, #070b18 55%, #04060d 100%) !important; }
body { background: #04060d; color: #eef2ff; font-family: 'Inter', Arial, sans-serif; }
.main-container { width: 100%; max-width: 460px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; font-family: 'Inter', Arial, sans-serif; }
.header { position: relative; background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02)); padding: 16px 12px 13px; border-radius: 18px; border: 1px solid rgba(140,170,255,0.25); text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08); backdrop-filter: blur(6px); }
.header h1 { margin: 0; font-size: 21px; font-weight: 900; letter-spacing: 3px; background: linear-gradient(90deg, #7dd3fc, #a5b4fc, #f0abfc); -webkit-background-clip: text; background-clip: text; color: transparent; }
.header p { color: #8b96b8; margin: 5px 0 0 0; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; }
.badge-version { position: absolute; top: 10px; right: 12px; background: rgba(0,230,140,0.14); color: #00e68c; border: 1px solid rgba(0,230,140,0.45); font-size: 9px; font-weight: 700; letter-spacing: 1px; padding: 2px 8px; border-radius: 999px; }
.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.stat-box { background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015)); border: 1px solid rgba(255,255,255,0.09); border-radius: 14px; padding: 9px 4px; text-align: center; box-shadow: inset 0 1px 0 rgba(255,255,255,0.06); }
.stat-box strong { color: #8b96b8; font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; display: block; margin-bottom: 3px; }
.stat-box span { color: #ffffff; font-size: 16px; font-weight: 800; font-variant-numeric: tabular-nums; }
.hora-box { background: linear-gradient(160deg, rgba(0,230,140,0.10), rgba(0,230,140,0.02)); border: 1px solid rgba(0,230,140,0.35); border-radius: 14px; padding: 8px; text-align: center; }
.hora-box span { color: #00e68c; font-size: 23px; font-weight: 800; letter-spacing: 2px; font-variant-numeric: tabular-nums; text-shadow: 0 0 14px rgba(0,230,140,0.55); }
.signal-box { padding: 14px 10px; border-radius: 16px; text-align: center; font-size: 26px; font-weight: 900; letter-spacing: 3px; margin: 5px 0; }
@keyframes pulso-suave { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.012); } }
.signal-call { background: linear-gradient(135deg, #00b359, #00e676); border: 1px solid rgba(0,255,150,0.7); color: #ffffff; box-shadow: 0 8px 28px rgba(0,230,118,0.35), inset 0 1px 0 rgba(255,255,255,0.35); animation: pulso-suave 1.6s ease-in-out infinite; text-shadow: 0 1px 3px rgba(0,60,25,0.5); }
.signal-put { background: linear-gradient(135deg, #d81b3a, #ff4d5e); border: 1px solid rgba(255,90,110,0.7); color: #ffffff; box-shadow: 0 8px 28px rgba(255,60,90,0.32), inset 0 1px 0 rgba(255,255,255,0.3); animation: pulso-suave 1.6s ease-in-out infinite; text-shadow: 0 1px 3px rgba(70,0,10,0.5); }
.signal-wait { background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01)); border: 1px solid rgba(250,200,60,0.4); color: #facc15; font-size: 15px; letter-spacing: 1.5px; padding: 11px; }
.confianza-bar-container { background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015)); border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; padding: 14px; margin: 9px 0; text-align: center; box-shadow: inset 0 1px 0 rgba(255,255,255,0.06); }
.confianza-bar-label { color: #aab4d4; font-size: 11px; font-weight: 700; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 2px; }
.confianza-bar-track { background: rgba(255,255,255,0.07); border-radius: 999px; height: 26px; position: relative; overflow: hidden; border: 1px solid rgba(255,255,255,0.08); }
.confianza-bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 13px; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.55); }
.confianza-alta { background: linear-gradient(90deg, #00b359, #00e676); box-shadow: 0 0 16px rgba(0,230,118,0.55); }
.confianza-media { background: linear-gradient(90deg, #e8930c, #facc15); box-shadow: 0 0 16px rgba(250,190,20,0.5); }
.confianza-baja { background: linear-gradient(90deg, #e63c1e, #ff7a3c); box-shadow: 0 0 16px rgba(255,100,50,0.5); }
.radar-container { position: relative; width: 154px; height: 154px; margin: 10px auto; }
.radar { width: 154px; height: 154px; border-radius: 50%; position: relative; overflow: hidden; border: 2px solid; }
.radar-green { border-color: rgba(0,230,118,0.8); background: radial-gradient(circle, #0a2e1e 0%, #04140c 75%); box-shadow: 0 0 26px rgba(0,230,118,0.35), inset 0 0 34px rgba(0,230,118,0.12); }
.radar-red { border-color: rgba(255,77,94,0.8); background: radial-gradient(circle, #2e0d13 0%, #140405 75%); box-shadow: 0 0 26px rgba(255,77,94,0.32), inset 0 0 34px rgba(255,77,94,0.12); }
.radar-yellow { border-color: rgba(250,204,21,0.8); background: radial-gradient(circle, #2e2a0d 0%, #141104 75%); box-shadow: 0 0 26px rgba(250,204,21,0.3), inset 0 0 34px rgba(250,204,21,0.12); }
.radar-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 10; }
.radar-number { font-size: 50px; font-weight: 900; line-height: 1; font-variant-numeric: tabular-nums; text-shadow: 0 0 14px currentColor; }
.radar-number-green { color: #00e676; }
.radar-number-red { color: #ff4d5e; }
.radar-number-yellow { color: #facc15; font-size: 17px; font-weight: 800; }
.radar-label { color: #8b96b8; font-size: 9px; font-weight: 700; margin-top: 4px; letter-spacing: 2px; }
.info-panel { background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012)); border: 1px solid rgba(125,211,252,0.22); border-radius: 11px; padding: 7px 11px; font-size: 11.5px; text-align: center; color: #d7def5; line-height: 1.5; }
.info-panel strong { color: #7dd3fc; }
.voz-indicator { background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(217,70,239,0.16)); border: 1px solid rgba(196,130,255,0.45); border-radius: 11px; padding: 5px 9px; font-size: 10px; color: #efe6ff; text-align: center; margin: 4px 0; }
div.stButton > button { background: linear-gradient(160deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03)); color: #eef2ff; border: 1px solid rgba(255,255,255,0.16); border-radius: 12px; font-weight: 700; transition: all 0.15s ease; }
div.stButton > button:hover { border-color: rgba(125,211,252,0.6); color: #ffffff; transform: translateY(-1px); }
div.stFormSubmitButton > button { background: linear-gradient(135deg, #2563eb, #7c3aed); color: #ffffff; border: none; border-radius: 12px; font-weight: 800; letter-spacing: 1px; }
[data-testid="stExpander"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 14px !important; }
[data-baseweb="select"] > div { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.14); border-radius: 12px; }
::-webkit-scrollbar { width: 8px; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Parte estatica (ya NO se recarga cada segundo) ----------------

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div class="header"><span class="badge-version">v3.7.1</span><h1>⚡ ZT TRADERS PRO</h1><p>User: {st.session_state.usuario_actual} · IA AUTO-APRENDIZAJE</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    par_nombre = st.selectbox("", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel), label_visibility="collapsed")
with col2:
    if st.button("🗑️ Reset", key="reset_btn", use_container_width=True):
        resetear_memoria()
        reiniciar_ciclo()
        st.rerun()

if par_nombre != st.session_state.par_sel:
    st.session_state.par_sel = par_nombre
    reiniciar_ciclo()

codigo_info = PARES[st.session_state.par_sel]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

tv_symbol = f"BINANCE:{codigo}" if tipo == 'crypto' else f"FX:{codigo.replace('=X', '')}"
widget_html = f"""
<div class="tradingview-widget-container" style="height:180px;width:100%;margin-bottom:6px;">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{"width": "100%", "height": "180", "symbol": "{tv_symbol}", "interval": "1", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "es", "enable_publishing": false, "hide_top_toolbar": true, "hide_legend": true, "save_image": false, "calendar": false, "hide_volume": true, "support_host": "https://www.tradingview.com"}}
  </script>
</div>
"""
components.html(widget_html, height=185)

# ---------------- Panel IA v3 (solo Admin): backtesting + entrenamiento ----------------
_info_modelo = modelo_v3_info()
if st.session_state.usuario_actual.startswith("👑"):
    with st.expander("🧠 IA v3 — Entrenar con backtesting"):
        if _info_modelo:
            _m = _info_modelo["metricas"]
            st.success(f"Modelo activo — entrenado el {_m['fecha_entrenamiento']} con {_m['ejemplos_entrenamiento']} ejemplos. Brier: {_m['brier']}")
            st.dataframe(pd.DataFrame(_m["por_umbral"]), use_container_width=True, hide_index=True)
            if _m.get("por_regimen"):
                st.caption("Precisión por régimen de mercado (umbral 0.62):")
                st.dataframe(pd.DataFrame(_m["por_regimen"]), use_container_width=True, hide_index=True)
            st.caption("Precisión = % de acierto de las señales emitidas en datos que el modelo nunca vio. Cobertura = % de ciclos en los que sí emite señal.")
        else:
            st.info("Todavía no hay modelo entrenado. Sin modelo, la app usa solo el filtro de puntaje técnico.")
        _codigos_crypto = [v["codigo"] for v in PARES.values() if v["tipo"] == "crypto"]
        _sel = st.multiselect("Pares crypto para el backtest", _codigos_crypto,
                              default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        _dias = st.slider("Días de historial (velas 1m)", 7, 30, 14)
        if st.button("🚀 Entrenar modelo con backtest", use_container_width=True):
            try:
                with st.status("Entrenando modelo v3...", expanded=True) as _estado:
                    _dfs = []
                    for _sym in _sel:
                        _estado.write(f"Descargando {_sym}: {_dias} días de velas 1m de Binance...")
                        _dfh = nucleo_ia.descargar_historico_binance(_sym, _dias, sesion_http())
                        if _dfh is not None:
                            _dfs.append(_dfh)
                            _estado.write(f"  {_sym}: {len(_dfh)} velas")
                    _estado.write("Construyendo dataset etiquetado (esto puede tardar unos minutos)...")
                    _X, _y, _g, _t = nucleo_ia.construir_dataset(_dfs)
                    _estado.write(f"Dataset: {len(_X)} ejemplos, {len(set(_g.tolist()))} decisiones útiles")
                    _estado.write("Entrenando y calibrando (división por tiempo 60/20/20)...")
                    _info = nucleo_ia.entrenar_modelo(_X, _y, _g, _t)
                    nucleo_ia.guardar_modelo(_info, ARCHIVO_MODELO_V3, "modelo_v3_metricas.json")
                    modelo_v3_info.clear()
                    _estado.update(label="✅ Modelo v3 entrenado y guardado", state="complete")
                st.dataframe(pd.DataFrame(_info["metricas"]["por_umbral"]), use_container_width=True, hide_index=True)
                if _info["metricas"].get("por_regimen"):
                    st.dataframe(pd.DataFrame(_info["metricas"]["por_regimen"]), use_container_width=True, hide_index=True)
                st.success("Modelo activo. A partir de ahora las señales crypto se filtran con su probabilidad calibrada (mínimo 62%).")
            except Exception as _e:
                st.error(f"No se pudo entrenar: {_e}")
    with st.expander("💬 Diario inteligente — IA que razona (v3.4)"):
        if not ia_configurar_desde_secrets():
            st.info("Sin clave de IA configurada. En los Secrets de Streamlit agrega: ia_api_key = \"tu clave gratis\" e ia_proveedor = \"gemini\" (o \"groq\" / \"openrouter\"). Claves gratis: Google AI Studio (aistudio.google.com) o Groq (console.groq.com).")
        else:
            st.caption(f"IA conectada (proveedor: {_secret('ia_proveedor', 'IA_PROVEEDOR') or 'gemini'}). Segunda opinión por señal: {'ACTIVADA' if _ia_bool('ia_segunda_opinion') else 'apagada — actívala con ia_segunda_opinion = true en Secrets'}. Clima de sesión: {'ACTIVADO' if _ia_bool('ia_clima') else 'apagado — ia_clima = true'}.")
            if st.button("🧠 Analizar mi historial con IA", use_container_width=True):
                _hist = st.session_state.hist_cache
                _tot = len(_hist)
                _gan = sum(1 for h in _hist if h.get("resultado") == "GANADA")
                _lineas = [f"Operaciones registradas: {_tot}"]
                if _tot:
                    _lineas.append(f"Ganadas: {_gan} ({_gan / _tot * 100:.1f}%)")
                    for _s in ("CALL", "PUT"):
                        _hs = [h for h in _hist if h.get("señal") == _s]
                        if _hs:
                            _g = sum(1 for h in _hs if h.get("resultado") == "GANADA")
                            _lineas.append(f"{_s}: {_g}/{len(_hs)} ganadas ({_g / len(_hs) * 100:.0f}%)")
                    _idx_reg = nucleo_ia.FEATURES_BASE.index("regimen_cod")
                    _por_reg = {}
                    for _h in _hist:
                        _f = _h.get("features")
                        if _f and len(_f) > _idx_reg:
                            _rn = nucleo_ia.nombre_regimen(_f[_idx_reg])
                            _a, _b = _por_reg.get(_rn, (0, 0))
                            _por_reg[_rn] = (_a + 1, _b + (1 if _h.get("resultado") == "GANADA" else 0))
                    for _rn, (_n, _g) in _por_reg.items():
                        _lineas.append(f"Régimen {_rn}: {_g}/{_n} ganadas ({_g / _n * 100:.0f}%)")
                with st.spinner("La IA está leyendo tu historial..."):
                    _informe = razonamiento_ia.analizar_historial("\n".join(_lineas))
                if _informe:
                    st.markdown(_informe)
                else:
                    st.error("La IA no respondió. Revisa tu clave de API o inténtalo de nuevo.")
    with st.expander("💾 Respaldo del aprendizaje — guarda tus operaciones"):
        st.caption("El historial y el cerebro en vivo viven solo en el servidor de Streamlit y se pueden perder al reiniciar o volver a subir la app. Descarga una copia antes de subir una versión nueva y guárdala junto a los archivos de la app: si se pierde, subir ese archivo con la app lo restaura.")
        _hist_respaldo = st.session_state.hist_cache
        st.download_button("⬇️ Descargar historial.json (operaciones reales)",
                           data=json.dumps(_hist_respaldo, indent=2, ensure_ascii=False),
                           file_name=f"historial_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", use_container_width=True)
        st.download_button("⬇️ Descargar patrones.json (contadores)",
                           data=json.dumps(st.session_state.pesos, indent=2, ensure_ascii=False),
                           file_name=f"patrones_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", use_container_width=True)
        if os.path.exists(ARCHIVO_MODELO_VIVO):
            with open(ARCHIVO_MODELO_VIVO, "rb") as _f:
                st.download_button("⬇️ Descargar modelo_vivo.pkl (cerebro en vivo)",
                                   data=_f.read(),
                                   file_name=f"modelo_vivo_{ahora_local().strftime('%Y%m%d_%H%M')}.pkl",
                                   mime="application/octet-stream", use_container_width=True)
        else:
            st.caption("Todavía no hay modelo en vivo (se crea solo al llegar a 60 operaciones reales).")
        st.caption(f"Operaciones en el historial ahora: {len(_hist_respaldo)}.")
elif _info_modelo:
    st.caption(f"🧠 IA v3 activa — modelo entrenado el {_info_modelo['metricas']['fecha_entrenamiento']}")


# ---------------- Panel de aprendizaje en vivo (visible para todos) ----------------
with st.expander("📈 Aprendizaje en vivo — así va aprendiendo la app"):
    _hist = st.session_state.hist_cache
    _validas = nucleo_ia.ops_validas_para_vivo(_hist)
    _n_val = len(_validas)
    _vivo = modelo_vivo_info()
    if _vivo:
        _mv = _vivo["metricas"]
        st.success(f"🧠 Cerebro en vivo ACTIVO — entrenado con {_mv['ejemplos_entrenamiento']} operaciones reales "
                   f"({_mv['fecha_entrenamiento']}). En operaciones que no vio al entrenar: acierto "
                   f"{_mv['winrate_prueba_%']}% en total y {_mv['precision_filtro_0.55_%']}% en las que su "
                   f"probabilidad fue ≥55%. Se reentrena solo cada {nucleo_ia.REENTRENAR_CADA_OPS} operaciones nuevas.")
    else:
        st.info(f"El cerebro en vivo se activa con {nucleo_ia.MIN_OPS_MODELO_VIVO} operaciones reales registradas "
                f"(cada señal se juzga con el precio de entrada y el de cierre: la app sabe si ganó o perdió, y de eso aprende). "
                f"Llevas {_n_val}.")
        st.progress(min(_n_val / nucleo_ia.MIN_OPS_MODELO_VIVO, 1.0))
    if _n_val:
        def _tabla_vivo(clave, etiqueta):
            grupos = {}
            for _h in _validas:
                k = _h.get(clave) or "—"
                a, b = grupos.get(k, (0, 0))
                grupos[k] = (a + 1, b + (1 if _h["resultado"] == "GANADA" else 0))
            filas = [{etiqueta: k, "ops": n, "ganadas": g,
                      "acierto_%": round(g / n * 100, 1) if n else 0.0}
                     for k, (n, g) in sorted(grupos.items())]
            return pd.DataFrame(filas)
        _ganadas = sum(1 for _h in _validas if _h["resultado"] == "GANADA")
        st.caption(f"Operaciones reales registradas: {_n_val} · Acierto global: {round(_ganadas / _n_val * 100, 1)}% "
                   f"(con payout 86% en IQ Option necesitas más del 53,8% para ganar dinero).")
        _c1, _c2 = st.columns(2)
        with _c1:
            st.caption("Acierto por par")
            st.dataframe(_tabla_vivo("par", "par"), use_container_width=True, hide_index=True)
            st.caption("Acierto por dirección")
            st.dataframe(_tabla_vivo("señal", "dirección"), use_container_width=True, hide_index=True)
        with _c2:
            st.caption("Acierto por nivel de señal")
            st.dataframe(_tabla_vivo("nivel", "nivel"), use_container_width=True, hide_index=True)
            st.caption("Acierto por tipo")
            st.dataframe(_tabla_vivo("tipo", "tipo"), use_container_width=True, hide_index=True)

# ---------------- Helpers de la zona dinamica ----------------

def limpiar_html(html):
    """Quita la indentacion de cada linea de un bloque HTML.

    Sin esto, Markdown interpreta las lineas HTML con varios espacios al
    inicio como un BLOQUE DE CODIGO y las muestra como texto crudo
    (ese es el error de la captura: el <div> de la señal se veia sin pintar).
    """
    return "\n".join(linea.strip() for linea in str(html).strip().splitlines())

class ContenedorLimpio:
    """Envuelve st.empty() para limpiar siempre el HTML antes de mostrarlo."""
    def __init__(self, contenedor):
        self._contenedor = contenedor

    def markdown(self, html, **kwargs):
        self._contenedor.markdown(limpiar_html(html), **kwargs)

def barra_confianza_html(confianza_actual, titulo):
    if confianza_actual >= 75:
        clase_barra, icono = "confianza-alta", "🔊"
    elif confianza_actual >= 60:
        clase_barra, icono = "confianza-media", "⚠️"
    else:
        clase_barra, icono = "confianza-baja", "🔇"
    return f"""
    <div class="confianza-bar-container">
        <div class="confianza-bar-label">{icono} {titulo}</div>
        <div class="confianza-bar-track">
            <div class="confianza-bar-fill {clase_barra}" style="width: {confianza_actual}%;">{confianza_actual}%</div>
        </div>
    </div>
    """

def nivel_html_de(señal):
    nivel = señal.get('nivel') or ''
    if not nivel:
        return ""
    emoji = {"Alta": "🟢", "Media": "🟡", "Baja": "🔴"}.get(nivel, "⚪")
    html = f'<div class="info-panel">{emoji} Nivel de la señal: <strong>{nivel}</strong></div>'
    for adv in (señal.get('advertencias') or []):
        html += f'<div class="info-panel">⚠️ {adv}</div>'
    op = señal.get('opinion_ia')
    if op:
        html += f'<div class="info-panel">🧠 IA ({op["veredicto"]}): {op["motivo"]}</div>'
    return html

def cerrar_operacion(codigo_op, tipo_op):
    """Evalua la operacion abierta con el precio actual y la registra una sola vez."""
    señal = st.session_state.senal
    entrada = st.session_state.precio_real_entrada
    if señal is None or señal.get('señal') == 'ESPERA' or not entrada or st.session_state.resultado_guardado:
        st.session_state.resultado_guardado = True
        return
    precio_actual = obtener_precio_actual(codigo_op, tipo_op)
    resultado = 'PERDIDA'
    if precio_actual is not None:
        es_call = señal['señal'] == 'CALL'
        if (es_call and precio_actual > entrada) or (not es_call and precio_actual < entrada):
            resultado = 'GANADA'
    aprender(señal['señal'], resultado, señal.get('indicadores', {}), entrada, señal.get('features'),
             par=codigo_op, tipo=tipo_op, nivel=señal.get('nivel'), confianza=señal.get('confianza'))
    st.session_state.resultado_guardado = True

# ---------------- Zona dinamica: fragmento cada 1s ----------------
# Solo esto se re-ejecuta cada segundo. Nada de sleep() + st.rerun() global.

@st.fragment(run_every="1s")
def zona_dinamica(codigo_f, tipo_f):
    ahora = ahora_local()
    hora_actual = ahora.strftime('%H:%M:%S')
    minuto_actual = ahora.minute
    segundo_actual = ahora.second
    minuto_es_par = (minuto_actual % 2 == 0)
    clave_minuto = ahora.strftime('%Y%m%d%H%M')

    # Cierre de rescate: si el ciclo anterior quedo 'operando' sin registrar
    # (ventana de cierre perdida), se cierra ahora con el precio actual.
    if minuto_es_par and st.session_state.estado == 'operando' and not st.session_state.resultado_guardado:
        cerrar_operacion(codigo_f, tipo_f)

    # Estadisticas + reloj
    pesos = st.session_state.pesos
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><strong>Winrate</strong><span>{round(pesos['winrate'], 1)}%</span></div>
        <div class="stat-box"><strong>Ops</strong><span>{pesos['total_ops']}</span></div>
        <div class="stat-box"><strong>Win</strong><span>{pesos['total_ganadas']}</span></div>
        <div class="stat-box"><strong>ML</strong><span style="font-size:11px;">{len(st.session_state.hist_cache)} ops</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="hora-box"><span>🕐 {hora_actual}</span></div>', unsafe_allow_html=True)

    contenedor = ContenedorLimpio(st.empty())

    # FASE 1: ESCANEO (minutos pares, segundos 0-54)
    if minuto_es_par and segundo_actual < 55:
        if st.session_state.estado != 'analizando':
            st.session_state.estado = 'analizando'
            st.session_state.senal = None
            st.session_state.voz_reproducida = False

        # Segunda opinion IA (v3.4): pre-analisis al segundo 40, para que el
        # veredicto ya este listo cuando la señal salga (segundo 55).
        if segundo_actual >= 40 and st.session_state.ciclo_opinion != clave_minuto:
            st.session_state.ciclo_opinion = clave_minuto
            if ia_opinion_activa():
                try:
                    pre = analizar_par(codigo_f, tipo_f)
                    if pre.get('señal') in ('CALL', 'PUT'):
                        op = razonamiento_ia.segunda_opinion({
                            'par': codigo_f, 'tipo': tipo_f, 'direccion': pre['señal'],
                            'confianza': pre['confianza'], 'nivel': pre.get('nivel') or '-',
                            'razon': pre['razon'], 'regimen': '',
                            'advertencias': pre.get('advertencias') or []})
                        if op:
                            op['direccion'] = pre['señal']
                            st.session_state.opinion_ia = op
                except Exception:
                    pass

        # Clima de sesion IA (v3.4): se renueva cada 15 minutos.
        if ia_clima_activo() and (time.time() - st.session_state.clima_ts > 900):
            st.session_state.clima_ts = time.time()
            try:
                if tipo_f == "crypto":
                    dfc = obtener_datos_binance(codigo_f, interval="1m", limit=120)
                else:
                    dfc, _fuente_clima = obtener_datos_forex(codigo_f)
                if dfc is not None and len(dfc) > 61:
                    var = (float(dfc['close'].iloc[-1]) / float(dfc['close'].iloc[-61]) - 1) * 100
                    txt = razonamiento_ia.clima_sesion(codigo_f, tipo_f, round(var, 2), "segun el analisis del ciclo")
                    if txt:
                        st.session_state.clima_ia = txt
            except Exception:
                pass

        clima_html = f'<div class="info-panel">🧠 Clima IA: {st.session_state.clima_ia}</div>' if st.session_state.clima_ia else ''
        contenedor.markdown(f"""
        <div class="signal-box signal-wait">📊 ESCANEANDO MERCADO...</div>
        {clima_html}
        <div class="info-panel">⏱️ Próxima señal en: <strong>{55 - segundo_actual}s</strong></div>
        <div class="radar-container"><div class="radar radar-yellow"><div class="radar-center"><div class="radar-number radar-number-yellow">IA</div><div class="radar-label">ACTIVA</div></div></div></div>
        """, unsafe_allow_html=True)
        return

    # FASE 2: SEÑAL (minutos pares, segundos 55-59) - se analiza UNA vez por ciclo
    if minuto_es_par and segundo_actual >= 55:
        if st.session_state.ciclo_senal != clave_minuto:
            st.session_state.senal = analizar_par(codigo_f, tipo_f)
            st.session_state.estado = 'senal'
            st.session_state.ciclo_senal = clave_minuto
            st.session_state.voz_reproducida = False

        señal = st.session_state.senal or {'señal': 'ESPERA', 'confianza': 0, 'razon': 'Sin datos'}
        if señal['señal'] == 'ESPERA':
            contenedor.markdown(f"""
            <div class="signal-box signal-wait">⏸️ SIN SEÑAL ESTE CICLO</div>
            <div class="info-panel">Motivo: <strong>{señal["razon"]}</strong></div>
            """, unsafe_allow_html=True)
            return

        # Segunda opinion IA (v3.4): ajusta el NIVEL si la direccion
        # coincide con la del pre-analisis. Nunca veta ni cambia la señal.
        _op = st.session_state.opinion_ia
        if _op and _op.get('direccion') == señal['señal'] and not señal.get('opinion_aplicada'):
            señal['nivel'] = _ajustar_nivel(señal.get('nivel') or 'Media', _op['veredicto'])
            señal['opinion_ia'] = _op
            señal['opinion_aplicada'] = True
            st.session_state.senal = señal

        es_call = señal['señal'] == 'CALL'
        confianza_actual = int(señal['confianza'])
        segundos_restantes = 59 - segundo_actual

        voz_html = ""
        if not st.session_state.voz_reproducida:
            if señal.get('nivel') == 'Alta':
                tipo_texto = "COMPRA" if es_call else "VENTA"
                reproducir_voz(f"Atención. Señal de nivel alto. {tipo_texto}. {confianza_actual} por ciento. Entrar ahora.")
                voz_html = f'<div class="voz-indicator">🔊 VOZ ACTIVADA - Nivel Alta {confianza_actual}%</div>'
            else:
                voz_html = f'<div class="voz-indicator">🔇 Nivel {señal.get("nivel", "-")} {confianza_actual}% - Voz solo en nivel Alta</div>'
            st.session_state.voz_reproducida = True

        clase_box = "signal-call" if es_call else "signal-put"
        clase_radar = "radar-green" if es_call else "radar-red"
        clase_numero = "radar-number-green" if es_call else "radar-number-red"
        tipo_signal = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        contenedor.markdown(
            voz_html
            + barra_confianza_html(confianza_actual, "NIVEL DE CONFIANZA DE LA SEÑAL")
            + nivel_html_de(señal)
            + f"""
            <div class="signal-box {clase_box}">{tipo_signal}</div>
            <div class="info-panel">Razón: <strong>{señal["razon"]}</strong></div>
            <div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes}</div><div class="radar-label">SEGUNDOS</div></div></div></div>
            """,
            unsafe_allow_html=True,
        )
        return

    # FASE 3: OPERACION (minutos impares)
    señal = st.session_state.senal
    if señal is None or señal.get('señal') == 'ESPERA':
        contenedor.markdown("""
        <div class="signal-box signal-wait">⏸️ SIN OPERACIÓN EN ESTE CICLO</div>
        <div class="info-panel">No hubo señal válida en la fase anterior.</div>
        """, unsafe_allow_html=True)
        return

    if st.session_state.ciclo_entrada != clave_minuto:
        st.session_state.estado = 'operando'
        st.session_state.resultado_guardado = False
        st.session_state.precio_real_entrada = obtener_precio_actual(codigo_f, tipo_f)
        st.session_state.ciclo_entrada = clave_minuto

    es_call = señal['señal'] == 'CALL'
    confianza_actual = int(señal['confianza'])
    segundos_restantes_op = 59 - segundo_actual

    entrada_html = ""
    if st.session_state.precio_real_entrada:
        entrada_html = f'<div class="info-panel">Entrada: <strong>{round(st.session_state.precio_real_entrada, 5)}</strong></div>'

    # Precio EN VIVO durante la operacion (v3.6.1): la Entrada es la foto
    # del segundo :00 y no se mueve; este precio si se mueve, para que el
    # usuario compare contra IQ Option en el mismo instante y vea si va
    # ganando o perdiendo y por cuanto.
    vivo_html = ""
    _precio_ahora = obtener_precio_actual(codigo_f, tipo_f)
    if _precio_ahora and st.session_state.precio_real_entrada:
        _ent = float(st.session_state.precio_real_entrada)
        _dif = float(_precio_ahora) - _ent
        _va_ganando = (_dif > 0) if es_call else (_dif < 0)
        if tipo_f == "forex":
            _pip = 0.01 if "JPY" in str(codigo_f) else 0.0001
            _dif_txt = f"{_dif / _pip:+.1f} pips"
        else:
            _dif_txt = f"{_dif / _ent * 100:+.3f}%"
        vivo_html = (f'<div class="info-panel">Precio ahora: <strong>{round(float(_precio_ahora), 5)}</strong>'
                     f' · Diferencia: <strong>{_dif_txt}</strong>'
                     f' · Vas <strong>{"GANANDO" if _va_ganando else "PERDIENDO"}</strong></div>')

    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    tipo_signal = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"

    contenedor.markdown(
        barra_confianza_html(confianza_actual, "CONFIANZA DE LA OPERACIÓN EN CURSO")
        + nivel_html_de(señal)
        + f"""
        <div class="signal-box {clase_box}">{tipo_signal}</div>
        {entrada_html}
        {vivo_html}
        <div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div><div class="radar-label">EXPIRACIÓN</div></div></div></div>
        """,
        unsafe_allow_html=True,
    )

    # Cierre al final del minuto de operacion (seg 57-59). El rescate de la
    # Fase 1 cubre el caso de que esta ventana se pierda.
    if segundo_actual >= 57 and not st.session_state.resultado_guardado:
        cerrar_operacion(codigo_f, tipo_f)

zona_dinamica(codigo, tipo)
st.markdown('</div>', unsafe_allow_html=True)
