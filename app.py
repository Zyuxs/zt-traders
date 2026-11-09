# ============================================================
# ZT TRADERS INSTITUTIONAL PRO - v2 mejorada
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
# ============================================================
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import requests
import yfinance as yf
import ta
import json
import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

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
    body { background: #000; color: #fff; font-family: Arial, sans-serif; }
    .login-container { max-width: 320px; margin: 60px auto; padding: 25px; background: linear-gradient(135deg, #0a0a1a, #1a0a2e); border-radius: 12px; border: 2px solid #ffd700; text-align: center; }
    .login-title { color: #ffd700; font-size: 24px; font-weight: bold; }
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
if "modelo_ml" not in st.session_state:
    st.session_state.modelo_ml = None
if "modelo_entrenado_con" not in st.session_state:
    st.session_state.modelo_entrenado_con = 0

def resetear_memoria():
    for archivo in (ARCHIVO_HISTORIAL, ARCHIVO_PATRONES):
        if os.path.exists(archivo):
            os.remove(archivo)
    st.session_state.hist_cache = []
    st.session_state.pesos = dict(PATRONES_BASE)
    st.session_state.modelo_ml = None
    st.session_state.modelo_entrenado_con = 0

def aprender(señal, resultado, indicadores, precio_real_entrada):
    """Registra el resultado una sola vez por operacion y actualiza memoria + disco."""
    hist = st.session_state.hist_cache
    pesos = st.session_state.pesos
    hist.append({
        "fecha": ahora_local().strftime("%H:%M:%S"),
        "señal": señal,
        "resultado": resultado,
        "precio_entrada_real": precio_real_entrada,
        "indicadores": indicadores,
    })
    pesos["total_ops"] += 1
    if resultado == "GANADA":
        pesos["total_ganadas"] += 1
    pesos["winrate"] = (pesos["total_ganadas"] / pesos["total_ops"] * 100) if pesos["total_ops"] > 0 else 0.0
    guardar_json_atomico(ARCHIVO_HISTORIAL, hist)
    guardar_json_atomico(ARCHIVO_PATRONES, pesos)

# ---------------- Machine Learning ----------------

def _extraer_dataset(hist):
    X, y = [], []
    for op in hist:
        try:
            ind = op.get("indicadores", {})
            X.append([ind.get("rsi", 50), ind.get("precio_relativo", 0.5), 1 if op["señal"] == "CALL" else 0])
            y.append(1 if op["resultado"] == "GANADA" else 0)
        except Exception:
            continue
    return X, y

def obtener_modelo_ml():
    """Entrena solo si: hay >=10 ops, existen las 2 clases y llegaron >=5 ops nuevas."""
    hist = st.session_state.hist_cache
    modelo = st.session_state.modelo_ml
    if len(hist) < 10:
        return None
    if modelo is not None and len(hist) - st.session_state.modelo_entrenado_con < 5:
        return modelo
    X, y = _extraer_dataset(hist)
    if len(X) < 10 or len(set(y)) < 2:
        return None
    try:
        nuevo = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        nuevo.fit(X, y)
        st.session_state.modelo_ml = nuevo
        st.session_state.modelo_entrenado_con = len(hist)
        return nuevo
    except Exception:
        return modelo

def prob_ganada_ml(modelo, rsi, precio_relativo, es_call):
    """Probabilidad de GANADA segun el modelo (esto corrige el bug de confianza original)."""
    if modelo is None:
        return None
    try:
        proba = modelo.predict_proba([[rsi, precio_relativo, 1 if es_call else 0]])[0]
        clases = list(modelo.classes_)
        return float(proba[clases.index(1)]) if 1 in clases else None
    except Exception:
        return None

# ---------------- Datos de mercado (con cache) ----------------

@st.cache_data(ttl=20, show_spinner=False)
def obtener_datos_binance(symbol, interval="1m", limit=50):
    try:
        r = sesion_http().get(
            "https://api.binance.com/api/v3/klines",
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

@st.cache_data(ttl=20, show_spinner=False)
def _descargar_yahoo(symbol, period, interval="1m"):
    try:
        datos = yf.download(symbol, period=period, interval=interval, progress=False)
        if datos is None or len(datos) == 0:
            return None
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        return datos[["Open", "High", "Low", "Close"]].dropna().copy()
    except Exception:
        return None

def obtener_datos_yahoo(symbol):
    # Primero 1 dia (mucho menos datos que 2d); solo si no alcanza, se pide 2d.
    df = _descargar_yahoo(symbol, "1d")
    if df is None or len(df) < 60:
        df2 = _descargar_yahoo(symbol, "2d")
        if df2 is not None and (df is None or len(df2) > len(df)):
            return df2
    return df

@st.cache_data(ttl=3, show_spinner=False)
def _precio_binance(symbol):
    try:
        return float(sesion_http().get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=5).json()["price"])
    except Exception:
        return None

@st.cache_data(ttl=3, show_spinner=False)
def _precio_yahoo(symbol):
    # Descarga propia con ttl corto: NO reutiliza la descarga de analisis (ttl=20),
    # porque el precio de entrada/cierre necesita ser lo mas fresco posible.
    try:
        datos = yf.download(symbol, period="1d", interval="1m", progress=False)
        if datos is None or len(datos) == 0:
            return None
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        return float(datos["Close"].iloc[-1])
    except Exception:
        return None

def obtener_precio_actual(codigo_par, tipo="crypto"):
    return _precio_binance(codigo_par) if tipo == "crypto" else _precio_yahoo(codigo_par)

# ---------------- Analisis de señal (por puntaje) ----------------

def analizar_par(codigo_par, tipo="crypto"):
    espera = {"señal": "ESPERA", "confianza": 0, "razon": "Datos insuficientes", "indicadores": {"rsi": 50, "precio_relativo": 0.5}}
    df = obtener_datos_binance(codigo_par, interval="1m", limit=50) if tipo == "crypto" else obtener_datos_yahoo(codigo_par)
    if df is None or len(df) < 25:
        return espera
    c_close = "close" if tipo == "crypto" else "Close"
    try:
        df = df.copy()
        df["RSI"] = ta.momentum.RSIIndicator(df[c_close], window=14).rsi()
        bb = ta.volatility.BollingerBands(df[c_close], window=20, window_dev=2)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["SMA20"] = df[c_close].rolling(20).mean()
        df = df.dropna(subset=["RSI", "BB_High", "BB_Low", "SMA20"])
        if len(df) < 5:
            return espera

        u1 = df.iloc[-1]
        precio = float(u1[c_close])
        rsi = float(u1["RSI"])
        bb_high = float(u1["BB_High"])
        bb_low = float(u1["BB_Low"])
        sma = float(u1["SMA20"])
        sma_prev = float(df["SMA20"].iloc[-6]) if len(df) >= 6 else sma
        rango_bb = bb_high - bb_low
        if rango_bb <= 0 or precio <= 0:
            return espera
        precio_relativo = max(0.0, min(1.0, (precio - bb_low) / rango_bb))

        # Puntaje: positivo favorece CALL, negativo favorece PUT.
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

        confianza = 55 + min(30, abs(score) * 6)
        if score == 0:
            confianza = 52

        indicadores = {"rsi": rsi, "precio_relativo": precio_relativo}
        p_ganar = prob_ganada_ml(obtener_modelo_ml(), rsi, precio_relativo, direccion == "CALL")
        if p_ganar is not None:
            confianza_ml = int(p_ganar * 100)
            confianza = int(confianza * 0.7 + confianza_ml * 0.3)
            razones.append(f"ML:{confianza_ml}%")
            if p_ganar < 0.40:
                razones.append("ML en contra")

        return {"señal": direccion, "confianza": max(50, min(int(confianza), 95)), "razon": " | ".join(razones) if razones else "Continuidad", "indicadores": indicadores}
    except Exception:
        return espera

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
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 5px !important; padding-bottom: 0px !important; padding-left: 8px !important; padding-right: 8px !important; max-width: 100% !important; }
body { background: #000; color: #fff; font-family: Arial, sans-serif; }
.main-container { width: 100%; max-width: 450px; margin: 0 auto; display: flex; flex-direction: column; gap: 6px; }
.header { background: linear-gradient(90deg, #1a0a2e, #0a1a2e); padding: 8px; border-radius: 8px; border: 1.5px solid #ffd700; text-align: center; }
.header h1 { color: #ffd700; margin: 0; font-size: 18px; letter-spacing: 2px; }
.header p { color: #aaa; margin: 2px 0 0 0; font-size: 10px; }
.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; }
.stat-box { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 1px solid #ffd700; border-radius: 6px; padding: 6px; text-align: center; }
.stat-box strong { color: #ffd700; font-size: 9px; display: block; margin-bottom: 2px; }
.stat-box span { color: #00ff88; font-size: 14px; font-weight: bold; }
.hora-box { background: linear-gradient(90deg, #0a2a1a, #1a1a2e); border: 2px solid #00ff88; border-radius: 6px; padding: 6px; text-align: center; }
.hora-box span { color: #00ff88; font-size: 20px; font-weight: bold; text-shadow: 0 0 8px #00ff88; }
.signal-box { padding: 10px; border-radius: 8px; text-align: center; font-size: 24px; font-weight: bold; letter-spacing: 2px; margin: 4px 0; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 2px solid #00ff00; color: white; box-shadow: 0 0 20px rgba(0, 255, 0, 0.6); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 2px solid #ff0000; color: white; box-shadow: 0 0 20px rgba(255, 0, 0, 0.6); }
.signal-wait { background: linear-gradient(135deg, #1a1a2e, #0a1a2e); border: 2px solid #ffd700; color: #ffd700; font-size: 16px; padding: 8px; }
.confianza-bar-container { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 2px solid #00ff88; border-radius: 10px; padding: 12px; margin: 8px 0; text-align: center; }
.confianza-bar-label { color: #00ff88; font-size: 12px; font-weight: bold; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 1px; }
.confianza-bar-track { background: #1a1a2e; border-radius: 10px; height: 28px; position: relative; overflow: hidden; border: 1px solid #333; }
.confianza-bar-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
.confianza-alta { background: linear-gradient(90deg, #00cc00, #00ff00); box-shadow: 0 0 15px rgba(0, 255, 0, 0.6); }
.confianza-media { background: linear-gradient(90deg, #ffaa00, #ffcc00); box-shadow: 0 0 15px rgba(255, 170, 0, 0.6); }
.confianza-baja { background: linear-gradient(90deg, #ff4400, #ff6600); box-shadow: 0 0 15px rgba(255, 68, 0, 0.6); }
.radar-container { position: relative; width: 150px; height: 150px; margin: 8px auto; }
.radar { width: 150px; height: 150px; border-radius: 50%; position: relative; overflow: hidden; border: 3px solid; }
.radar-green { border-color: #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 20px rgba(0, 255, 0, 0.5); }
.radar-red { border-color: #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 20px rgba(255, 0, 0, 0.5); }
.radar-yellow { border-color: #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 20px rgba(255, 215, 0, 0.5); }
.radar-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 10; }
.radar-number { font-size: 48px; font-weight: bold; line-height: 1; text-shadow: 0 0 10px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 16px; }
.radar-label { color: #fff; font-size: 9px; margin-top: 3px; letter-spacing: 1px; }
.info-panel { background: linear-gradient(135deg, #0a1a2e, #1a0a2e); border: 1px solid #00ff88; border-radius: 6px; padding: 6px 10px; font-size: 11px; text-align: center; }
.info-panel strong { color: #00ff88; }
.voz-indicator { background: #6a0dad; border: 1px solid #da70d6; border-radius: 6px; padding: 4px 8px; font-size: 10px; color: #fff; text-align: center; margin: 4px 0; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Parte estatica (ya NO se recarga cada segundo) ----------------

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div class="header"><h1>⚡ ZT TRADERS PRO</h1><p>User: {st.session_state.usuario_actual} | IA AUTO-APRENDIZAJE</p></div>', unsafe_allow_html=True)

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
    aprender(señal['señal'], resultado, señal.get('indicadores', {}), entrada)
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
        contenedor.markdown(f"""
        <div class="signal-box signal-wait">📊 ESCANEANDO MERCADO...</div>
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
            contenedor.markdown("""
            <div class="signal-box signal-wait">⏸️ SIN SEÑAL — DATOS INSUFICIENTES</div>
            <div class="info-panel">Este ciclo <strong>no se opera ni se registra</strong>. Antes se inventaba un CALL al 65%.</div>
            """, unsafe_allow_html=True)
            return

        es_call = señal['señal'] == 'CALL'
        confianza_actual = int(señal['confianza'])
        segundos_restantes = 59 - segundo_actual

        voz_html = ""
        if not st.session_state.voz_reproducida:
            if confianza_actual >= 75:
                tipo_texto = "COMPRA" if es_call else "VENTA"
                reproducir_voz(f"Atención. Señal de alta confianza. {tipo_texto}. {confianza_actual} por ciento. Entrar ahora.")
                voz_html = f'<div class="voz-indicator">🔊 VOZ ACTIVADA - {confianza_actual}%</div>'
            else:
                voz_html = f'<div class="voz-indicator">🔇 Confianza {confianza_actual}% - Voz silenciada (mínimo 75%)</div>'
            st.session_state.voz_reproducida = True

        clase_box = "signal-call" if es_call else "signal-put"
        clase_radar = "radar-green" if es_call else "radar-red"
        clase_numero = "radar-number-green" if es_call else "radar-number-red"
        tipo_signal = "🟢 COMPRA (CALL)" if es_call else "🔴 VENTA (PUT)"
        contenedor.markdown(
            voz_html
            + barra_confianza_html(confianza_actual, "NIVEL DE CONFIANZA DE LA SEÑAL")
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

    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    tipo_signal = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"

    contenedor.markdown(
        barra_confianza_html(confianza_actual, "CONFIANZA DE LA OPERACIÓN EN CURSO")
        + f"""
        <div class="signal-box {clase_box}">{tipo_signal}</div>
        {entrada_html}
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
