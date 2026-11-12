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
# --- v3.17: PANEL DE EVOLUCION (pregunta de Jose: "como se si la app esta evolucionando?") ---
# 30) La app guarda sola una "foto" de los indicadores cada 50 operaciones
#     validas (acierto, espejo, precision del cerebro en datos no vistos,
#     etiquetas limpias, juzgadas por Finnhub y postura) en evolucion.json,
#     y el panel nuevo muestra la tabla de fotos y el antes vs ahora con
#     los cambios en puntos. No toca señales, modelo ni aprendizaje: solo
#     mide. La primera foto se guarda al llegar a 50 validas (o con el
#     boton "Guardar foto ahora"); evolucion.json se descarga desde Respaldo
#     y se respalda solo antes de un Reset, como el historial.
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
from datetime import datetime, timezone, timedelta
import html as _html
from zoneinfo import ZoneInfo

import nucleo_ia
import razonamiento_ia

st.set_page_config(page_title="ZT TRADERS INSTITUTIONAL PRO", layout="wide", page_icon="⚡")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_EVOLUCION = "evolucion.json"  # fotos del Panel de Evolucion (v3.17)
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
        submit_key = st.form_submit_button("🚀 CONECTAR", width="stretch")
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
    st.caption("Al conectar, este navegador queda activado con tu enlace personal: guárdalo en Favoritos o en la "
               "pantalla de inicio y entrarás directo, sin escribir la clave cada vez.")
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
    # Red de seguridad (v3.8.1): antes de borrar, deja una copia automatica
    # (historial_respaldo_auto.json / patrones_respaldo_auto.json) para que
    # un Reset accidental se pueda deshacer desde el panel de Respaldo.
    import shutil
    for archivo, copia in ((ARCHIVO_HISTORIAL, "historial_respaldo_auto.json"),
                           (ARCHIVO_PATRONES, "patrones_respaldo_auto.json"),
                           (ARCHIVO_EVOLUCION, "evolucion_respaldo_auto.json")):
        if os.path.exists(archivo):
            try:
                shutil.copy(archivo, copia)
            except Exception:
                pass
            os.remove(archivo)
    st.session_state.hist_cache = []
    st.session_state.pesos = dict(PATRONES_BASE)

def aprender(señal, resultado, indicadores, precio_real_entrada, features=None,
             par=None, tipo=None, nivel=None, confianza=None, fuente=None):
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
        "fuente_datos": fuente,
    })
    pesos["total_ops"] += 1
    if resultado == "GANADA":
        pesos["total_ganadas"] += 1
    pesos["winrate"] = (pesos["total_ganadas"] / pesos["total_ops"] * 100) if pesos["total_ops"] > 0 else 0.0
    guardar_json_atomico(ARCHIVO_HISTORIAL, hist)
    guardar_json_atomico(ARCHIVO_PATRONES, pesos)
    quizas_entrenar_vivo()
    guardar_foto_evolucion_si_toca()  # v3.17: foto cada 50 ops validas

# ---------------- Panel de Evolucion (v3.17): fotos automaticas ----------------
# Responde la pregunta de Jose ("como se si la app esta evolucionando?")
# con numeros guardados por la propia app, sin depender de Muse ni de
# capturas: cada 50 operaciones validas se congela una foto de los 6
# indicadores que ya se miran en "Aprendizaje en vivo". La precision del
# cerebro se lee DESPUES del reentrenamiento (por eso este gancho va
# despues de quizas_entrenar_vivo() en aprender()): la foto refleja el
# cerebro nuevo, no el viejo.

def foto_evolucion_actual():
    """Foto de los indicadores de evolucion con el historial actual.
    Devuelve None si aun no hay operaciones validas para medir."""
    hist = st.session_state.hist_cache
    validas = nucleo_ia.ops_validas_para_vivo(hist)
    n = len(validas)
    if not n:
        return None
    ganadas = sum(1 for h in validas if h["resultado"] == "GANADA")
    precision = None
    vivo = modelo_vivo_info()
    if vivo:
        try:
            precision = float(vivo["metricas"]["winrate_prueba_%"])
        except Exception:
            precision = None
    return {"fecha": ahora_local().strftime("%Y-%m-%d %H:%M"),
            "ops": n,
            "acierto_%": round(ganadas / n * 100, 1),
            "espejo_%": round((n - ganadas) / n * 100, 1),
            "precision_no_visto_%": precision,
            "limpias": sum(1 for h in validas if h.get("fuente_datos") in ("Twelve Data", "Binance")),
            "finnhub": sum(1 for h in validas if h.get("fuente_datos") == "Finnhub"),
            "postura": nucleo_ia.postura_sesion(hist)[0]}

def guardar_foto_evolucion_si_toca():
    """Guarda una foto cuando se cruza un multiplo de 50 operaciones
    validas (50, 100, 150...). Una foto por tramo: si el historial ya
    tiene una foto de ese tramo, no duplica. Nunca rompe el registro de
    la operacion: cualquier error aqui se traga en silencio."""
    try:
        foto = foto_evolucion_actual()
        if not foto or foto["ops"] < 50:
            return
        fotos = leer_json(ARCHIVO_EVOLUCION, [])
        if fotos and fotos[-1].get("ops", 0) // 50 >= foto["ops"] // 50:
            return
        fotos.append(foto)
        guardar_json_atomico(ARCHIVO_EVOLUCION, fotos)
    except Exception:
        pass

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

@st.cache_data(ttl=100, show_spinner=False)
def _serie_twelvedata(codigo_yahoo, outputsize=120):
    # ttl 100s (v3.9.1): el analisis corre en los segundos 40 y 55 de cada
    # ciclo de 2 min; con 45s se pagaban ~2 creditos por ciclo y el plan
    # Basic (800/dia) se agotaba a mitad del dia, cayendo a Yahoo.
    return _td_request(codigo_yahoo, outputsize)

@st.cache_data(ttl=300, show_spinner=False)
def _uso_twelvedata():
    """Creditos Twelve Data consumidos hoy (para vigilar el plan Basic)."""
    key = _twelvedata_key()
    if not key:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/api_usage",
                         params={"apikey": key}, timeout=10)
        d = r.json()
        return (d.get("current_day_usage"), d.get("daily_limit"))
    except Exception:
        return None

@st.cache_data(ttl=25, show_spinner=False)
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

# ---------------- Finnhub: precio juez forex (v3.16) ----------------
# El plan gratuito de Finnhub limita por MINUTO (60 llamadas/min), no
# por dia: no existe el "se acabaron los creditos a mediodia" de Twelve
# Data. Entra como precio JUEZ (congelar la entrada, linea EN VIVO y
# precio de cierre); la serie de velas la sigue trayendo Twelve Data.
# Lleva etiqueta propia "Finnhub": NO cuenta como etiqueta limpia hasta
# verificar en vivo que su precio va parejo con IQ Option (la prueba de
# los 0,2 pips que aprobo Twelve Data). Y desde esta version la etiqueta
# de cada operacion nombra a quien la JUZGO de verdad (antes nombraba
# la fuente de la serie, que no siempre era la misma que el precio).

def _finnhub_key():
    return _secret("finnhub_api_key", "FINNHUB_API_KEY")

def _simbolo_finnhub(codigo_yahoo):
    base = str(codigo_yahoo).replace("=X", "")
    return f"OANDA:{base[:3]}_{base[3:6]}" if len(base) >= 6 else base

@st.cache_data(ttl=8, show_spinner=False)
def _precio_finnhub(codigo_yahoo):
    key = _finnhub_key()
    if not key:
        return None
    try:
        r = sesion_http().get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": _simbolo_finnhub(codigo_yahoo), "token": key},
            timeout=8)
        c = r.json().get("c")
        return float(c) if c else None
    except Exception:
        return None

def obtener_precio_actual_con_fuente(codigo_par, tipo="crypto"):
    """(precio, fuente): la fuente nombra a quien dio el precio de verdad."""
    if tipo == "crypto":
        return _precio_binance(codigo_par), "Binance"
    precio = _precio_finnhub(codigo_par)
    if precio is not None:
        return precio, "Finnhub"
    precio = _precio_twelvedata(codigo_par)
    if precio is not None:
        return precio, "Twelve Data"
    return _precio_yahoo(codigo_par), "Yahoo Finance"

def obtener_precio_actual(codigo_par, tipo="crypto"):
    return obtener_precio_actual_con_fuente(codigo_par, tipo)[0]

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
        # PUERTA DE CONFIANZA (v3.8): el modelo en vivo solo MANDA en la
        # direccion cuando ya demostro precision en operaciones que no vio
        # al entrenar (>=55% en su filtro de probabilidad). Mientras tanto
        # observa y sigue aprendiendo: manda el puntaje tecnico.
        if info_modelo is not None and fuente_modelo.startswith("en vivo"):
            _prec = (info_modelo.get("metricas") or {}).get("precision_filtro_0.55_%", 0) or 0
            if _prec < 55:
                razones.append(f"Cerebro en vivo en observación (precisión en datos no vistos: {_prec}% < 55%): manda el puntaje técnico")
                p_call = p_put = None
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
            nivel, _motivos_lec = nucleo_ia.aplicar_lecciones(
                nivel, fv, direccion, codigo_par,
                nucleo_ia.lecciones_aprendidas(st.session_state.get("hist_cache", [])))
            for _m in _motivos_lec:
                advertencias.append(f"📓 Lección de la experiencia: {_m} — nivel ajustado")
            # TERMOMETRO DE TERRENO (v3.15): si el contexto se parece poco
            # a lo que el cerebro en vivo aprendio (terreno desconocido),
            # el nivel baja un escalón mas, como con las lecciones.
            if fuente_modelo.startswith("en vivo"):
                _dis = nucleo_ia.disimilitud(info_modelo, vector_base)
                if _dis and _dis["desconocido"]:
                    _orden_n = ["Baja", "Media", "Alta"]
                    if nivel in _orden_n:
                        nivel = _orden_n[max(0, _orden_n.index(nivel) - 1)]
                    advertencias.append(f"🌡️ Termómetro: contexto poco parecido a lo que el cerebro aprendió (distancia {_dis['distancia']} > {_dis['umbral']}) — nivel ajustado")
            if direccion != direccion_puntaje:
                razones.append(f"La IA elige {direccion} (el puntaje decía {direccion_puntaje})")
            razones.append(f"IA: SUBE {int(round(p_call * 100))}% vs BAJA {int(round(p_put * 100))}%")
            razones.append(f"Fuente IA: {fuente_modelo}")
            razones.append(f"Régimen: {regimen_nombre} · Confluencia {conf}✓ · Nivel {nivel}")
            razones.append(f"Datos: {fuente_datos}")
            return {"señal": direccion, "confianza": confianza, "razon": " | ".join(razones),
                    "indicadores": indicadores, "features": vector_base + [1 if es_call else 0],
                    "nivel": nivel, "advertencias": advertencias, "fuente": fuente_datos,
                    "escaneo": {"regimen": regimen_nombre, "regimen_cod": int(fv.get("regimen_cod", 0)),
                                "adx": round(float(fv.get("adx", 0)), 1),
                                "dist_res": round(float(fv.get("dist_resistencia", 0)), 2),
                                "dist_sop": round(float(fv.get("dist_soporte", 0)), 2),
                                "confluencia": conf, "bloqueos": len(bloqueos)}}

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
        nivel, _motivos_lec = nucleo_ia.aplicar_lecciones(
            nivel, fv, direccion, codigo_par,
            nucleo_ia.lecciones_aprendidas(st.session_state.get("hist_cache", [])))
        for _m in _motivos_lec:
            advertencias.append(f"📓 Lección de la experiencia: {_m} — nivel ajustado")
        # TERMOMETRO DE TERRENO (v3.15): tambien aplica cuando manda el
        # puntaje tecnico (el cerebro observa): el terreno se mide igual.
        _info_vivo_t = modelo_vivo_info()
        if _info_vivo_t is not None:
            _dis = nucleo_ia.disimilitud(_info_vivo_t, vector_base)
            if _dis and _dis["desconocido"]:
                _orden_n = ["Baja", "Media", "Alta"]
                if nivel in _orden_n:
                    nivel = _orden_n[max(0, _orden_n.index(nivel) - 1)]
                advertencias.append(f"🌡️ Termómetro: contexto poco parecido a lo que el cerebro aprendió (distancia {_dis['distancia']} > {_dis['umbral']}) — nivel ajustado")
        razones.append(f"Régimen: {regimen_nombre} · Confluencia {conf}✓ · Nivel {nivel}")
        if tipo != "crypto" and modelo_vivo_info() is None:
            _validas = len(nucleo_ia.ops_validas_para_vivo(st.session_state.hist_cache))
            razones.append(f"Modelo en vivo aprendiendo: {_validas}/{nucleo_ia.MIN_OPS_MODELO_VIVO} operaciones reales")
        else:
            razones.append("Sin modelo IA entrenado")
        razones.append(f"Datos: {fuente_datos}")
        return {"señal": direccion, "confianza": confianza,
                "razon": " | ".join(razones), "indicadores": indicadores,
                "features": vector_base + [1 if direccion == "CALL" else 0],
                "nivel": nivel, "advertencias": advertencias, "fuente": fuente_datos,
                "escaneo": {"regimen": regimen_nombre, "regimen_cod": int(fv.get("regimen_cod", 0)),
                            "adx": round(float(fv.get("adx", 0)), 1),
                            "dist_res": round(float(fv.get("dist_resistencia", 0)), 2),
                            "dist_sop": round(float(fv.get("dist_soporte", 0)), 2),
                            "confluencia": conf, "bloqueos": len(bloqueos)}}
    except Exception:
        return espera

def indice_operabilidad(res):
    """Indice de Operabilidad REAL (0-100) para el escaneo global (v3.11):
    puntua las CONDICIONES del par con sus datos de verdad — regimen,
    fuerza de la tendencia (ADX), espacio libre en la direccion de la
    señal hasta el soporte/resistencia, confluencia y bloqueos. No es la
    señal: es si el par esta en condiciones de operarse ahora."""
    esc = res.get("escaneo") or {}
    if res.get("señal") not in ("CALL", "PUT") or not esc:
        return None
    idx = 50
    rc = esc.get("regimen_cod", 0)
    if rc in (1, 2):
        idx += 15
    elif rc == 3:
        idx -= 25
    adx = esc.get("adx", 0)
    if adx >= 30:
        idx += 10
    elif adx >= 20:
        idx += 5
    elif adx < 15:
        idx -= 10
    esp = esc.get("dist_res", 0) if res["señal"] == "CALL" else esc.get("dist_sop", 0)
    if esp >= 2:
        idx += 15
    elif esp >= 1:
        idx += 8
    elif esp < 0.5:
        idx -= 20
    if esc.get("confluencia", 0) >= 2:
        idx += 10
    if esc.get("bloqueos", 0):
        idx -= 15
    idx += max(0, int(res.get("confianza", 50)) - 50) // 4
    return max(1, min(99, idx))

def estado_operabilidad(idx):
    if idx is None:
        return "⚪ SIN DATOS"
    if idx >= 70:
        return "🟢 ARMÓNICO"
    if idx >= 45:
        return "🟡 NEUTRO"
    return "🔴 EVITAR"

def _motivo_escaner(res):
    esc = res.get("escaneo") or {}
    if res.get("señal") not in ("CALL", "PUT"):
        return str(res.get("razon", ""))[:80]
    esp = esc.get("dist_res", 0) if res["señal"] == "CALL" else esc.get("dist_sop", 0)
    m = f"{esc.get('regimen', '-')} · ADX {esc.get('adx', 0):.0f} · espacio {esp:.1f} ATR"
    if esc.get("bloqueos"):
        m += " · ⚠ con bloqueo"
    return m

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
    # v3.17.4: components.html, NO st.html — st.html no ejecuta <script> y
    # la voz moria en silencio (igual que el grafico). El aviso de
    # deprecacion en el log es solo eso, un aviso: el iframe sigue andando.
    components.html(js_code, height=0, width=0)

PARES = {
    'Bitcoin': {'codigo': 'BTCUSDT', 'tipo': 'crypto'}, 'Ethereum': {'codigo': 'ETHUSDT', 'tipo': 'crypto'},
    'Solana': {'codigo': 'SOLUSDT', 'tipo': 'crypto'}, 'Binance Coin': {'codigo': 'BNBUSDT', 'tipo': 'crypto'},
    'Ripple': {'codigo': 'XRPUSDT', 'tipo': 'crypto'}, 'Cardano': {'codigo': 'ADAUSDT', 'tipo': 'crypto'},
    'Dogecoin': {'codigo': 'DOGEUSDT', 'tipo': 'crypto'}, 'Polkadot': {'codigo': 'DOTUSDT', 'tipo': 'crypto'},
    'Litecoin': {'codigo': 'LTCUSDT', 'tipo': 'crypto'}, 'Polygon': {'codigo': 'MATICUSDT', 'tipo': 'crypto'},
    'Avalanche': {'codigo': 'AVAXUSDT', 'tipo': 'crypto'}, 'Chainlink': {'codigo': 'LINKUSDT', 'tipo': 'crypto'},
    'EUR/USD': {'codigo': 'EURUSD=X', 'tipo': 'forex'}, 'GBP/USD': {'codigo': 'GBPUSD=X', 'tipo': 'forex'},
    'USD/JPY': {'codigo': 'USDJPY=X', 'tipo': 'forex'}, 'AUD/USD': {'codigo': 'AUDUSD=X', 'tipo': 'forex'},
    'USD/CAD': {'codigo': 'USDCAD=X', 'tipo': 'forex'}, 'USD/CHF': {'codigo': 'USDCHF=X', 'tipo': 'forex'},
    'NZD/USD': {'codigo': 'NZDUSD=X', 'tipo': 'forex'}, 'EUR/GBP': {'codigo': 'EURGBP=X', 'tipo': 'forex'},
    'EUR/JPY': {'codigo': 'EURJPY=X', 'tipo': 'forex'}, 'GBP/JPY': {'codigo': 'GBPJPY=X', 'tipo': 'forex'},
    'AUD/JPY': {'codigo': 'AUDJPY=X', 'tipo': 'forex'}, 'EUR/AUD': {'codigo': 'EURAUD=X', 'tipo': 'forex'},
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
.main-container { width: 100%; max-width: 1560px; margin: 0 auto; display: flex; flex-direction: column; gap: 8px; font-family: 'Inter', Arial, sans-serif; }
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
.signal-call { background: linear-gradient(135deg, #00b359, #00e676); border: 1px solid rgba(0,255,150,0.7); color: #ffffff; box-shadow: 0 8px 28px rgba(0,230,118,0.35), inset 0 1px 0 rgba(255,255,255,0.35); text-shadow: 0 1px 3px rgba(0,60,25,0.5); }
.signal-put { background: linear-gradient(135deg, #d81b3a, #ff4d5e); border: 1px solid rgba(255,90,110,0.7); color: #ffffff; box-shadow: 0 8px 28px rgba(255,60,90,0.32), inset 0 1px 0 rgba(255,255,255,0.3); text-shadow: 0 1px 3px rgba(70,0,10,0.5); }
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
/* v3.10: vista compacta tipo "una sola pantalla" (inspirada en el panel
   del amigo de Jose): menos aire entre bloques y tarjetas mas bajas,
   para ver señal + estadisticas sin bajar tanto. */
.stVerticalBlock { gap: 0.45rem !important; }
.stat-box { padding: 6px 2px !important; }
.hora-box { margin: 2px 0 !important; }
/* v3.11: tarjeta en vivo estilo panel del amigo — radar a la izquierda y
   filas de datos (etiqueta / valor) a la derecha, barra de progreso del
   ciclo y pastillas de marcador (W-L) y EN VIVO. */
.brand-bar { display: flex; justify-content: space-between; align-items: center; margin: 2px 0 6px 0; }
.pill-record { background: rgba(250,204,21,0.12); border: 1px solid rgba(250,204,21,0.45); color: #fde68a; font-weight: 800; font-size: 12px; padding: 4px 12px; border-radius: 99px; letter-spacing: 1px; }
.pill-live { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.55); color: #fca5a5; font-weight: 800; font-size: 12px; padding: 4px 12px; border-radius: 99px; letter-spacing: 1px; }
.live-grid { display: flex; gap: 12px; align-items: center; margin: 6px 0; }
.live-grid .radar-container { flex: 0 0 auto; margin: 0 !important; padding: 0 !important; }
.data-rows { flex: 1; display: flex; flex-direction: column; gap: 6px; min-width: 0; }
.data-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.012)); border: 1px solid rgba(255,255,255,0.10); border-radius: 10px; padding: 7px 12px; }
.data-row span { color: #8b96b8; font-size: 10.5px; font-weight: 700; letter-spacing: 1.5px; white-space: nowrap; }
.data-row strong { color: #eef2ff; font-size: 14.5px; font-weight: 800; text-align: right; }
.data-row strong.amarillo { color: #facc15; }
.data-row strong.verde { color: #00e676; }
.data-row strong.rojo { color: #ff6b74; }
.progress-track { height: 8px; background: rgba(255,255,255,0.07); border-radius: 99px; overflow: hidden; margin: 6px 0; }
.progress-fill { height: 100%; background: linear-gradient(90deg, #facc15, #f59e0b); border-radius: 99px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Parte estatica (ya NO se recarga cada segundo) ----------------

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div class="header"><span class="badge-version">v3.17.5</span><h1>⚡ ZT TRADERS PRO</h1><p>User: {st.session_state.usuario_actual} · IA AUTO-APRENDIZAJE</p></div>', unsafe_allow_html=True)
if st.query_params.get("zt_token"):
    st.caption("✅ Este dispositivo está activado: guarda ESTA página en Favoritos o en la pantalla de inicio "
               "(con la dirección completa, tal como está arriba) y la próxima vez entrarás directo, sin escribir la clave.")

# v3.13: distribucion ANCHA para PC — a la izquierda el selector de par y
# el grafico (grande), a la derecha la zona de señales en vivo, lado a
# lado para ver todo sin bajar. En telefono las columnas se apilan solas
# (grafico primero, señal debajo), como antes.
col_izq, col_der = st.columns([1.3, 1], gap="medium")
with col_izq:
    col1, col2 = st.columns([3, 1])
    with col1:
        _nombres_pares = list(PARES.keys())
        _idx_par = _nombres_pares.index(st.session_state.par_sel) if st.session_state.get("par_sel") in _nombres_pares else 0
        par_nombre = st.selectbox("Par", _nombres_pares, index=_idx_par, label_visibility="collapsed")
    with col2:
        # v3.10.1 (seguridad): el Reset borra el aprendizaje COMPARTIDO de
        # todos los usuarios; solo lo ve el Admin Master.
        if st.session_state.usuario_actual.startswith("👑"):
            if st.button("🗑️ Reset", key="reset_btn", width="stretch"):
                st.session_state.confirmar_reset = True
            if st.session_state.get("confirmar_reset"):
                st.warning("⚠️ El Reset borra el historial de aprendizaje y los contadores. "
                           "Quedará una copia automática de seguridad, pero el aprendizaje vuelve a cero.")
                _cc1, _cc2 = st.columns(2)
                with _cc1:
                    if st.button("✅ Sí, borrar todo", key="reset_confirmar", width="stretch"):
                        st.session_state.confirmar_reset = False
                        resetear_memoria()
                        reiniciar_ciclo()
                        st.rerun()
                with _cc2:
                    if st.button("❌ Cancelar", key="reset_cancelar", width="stretch"):
                        st.session_state.confirmar_reset = False
                        st.rerun()

if par_nombre != st.session_state.par_sel:
    st.session_state.par_sel = par_nombre
    reiniciar_ciclo()

codigo_info = PARES[st.session_state.par_sel]
codigo = codigo_info['codigo']
tipo = codigo_info['tipo']

tv_symbol = f"BINANCE:{codigo}" if tipo == 'crypto' else f"FX:{codigo.replace('=X', '')}"
widget_html = f"""
<div class="tradingview-widget-container" style="height:340px;width:100%;margin-bottom:6px;">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {{"width": "100%", "height": "340", "symbol": "{tv_symbol}", "interval": "1", "timezone": "Etc/UTC", "theme": "dark", "style": "1", "locale": "es", "enable_publishing": false, "hide_top_toolbar": true, "hide_legend": true, "save_image": false, "calendar": false, "hide_volume": true, "support_host": "https://www.tradingview.com"}}
  </script>
</div>
"""
with col_izq:
    # v3.17.4: el widget de TradingView necesita su <script> y st.html no
    # ejecuta scripts: por eso el grafico salia en blanco. Vuelve al iframe.
    components.html(widget_html, height=345)

# La zona de señales en vivo se ancla en la columna DERECHA (v3.13; en
# v3.10 iba bajo el grafico). Los paneles quedan debajo, en expanders.
with col_der:
    slot_zona = st.container()

# ---------------- Escaneo global de pares (v3.11) ----------------
# La idea del panel del amigo de Jose (elegir el mejor par del momento),
# pero con el Indice de Operabilidad calculado de datos reales.
with st.expander("🌐 Escaneo global — ¿qué par conviene operar ahora?"):
    st.caption("Índice de Operabilidad calculado con datos reales de cada par (régimen, fuerza ADX, espacio libre "
               "hasta soporte/resistencia, confluencia y bloqueos). 🟢 ARMÓNICO = condiciones limpias · "
               "🟡 NEUTRO = operable con cuidado · 🔴 EVITAR = volátil, pegado a un nivel o con bloqueo. "
               "Escanear forex consume créditos de Twelve Data (1 por par): úsalo cuando lo necesites.")
    if st.button("🔍 Escanear los 24 pares ahora", width="stretch"):
        _resultados = []
        _barra = st.progress(0.0)
        for _i, (_nombre, _info) in enumerate(PARES.items()):
            try:
                _r = analizar_par(_info["codigo"], _info["tipo"])
            except Exception:
                _r = {"señal": "ESPERA", "confianza": 0, "razon": "Error de análisis", "nivel": "", "escaneo": None}
            _idx = indice_operabilidad(_r)
            _resultados.append({"Par": _nombre, "Estado": estado_operabilidad(_idx),
                                "Índice": _idx, "Señal tentativa": _r.get("señal"),
                                "Confianza": _r.get("confianza"), "Nivel": _r.get("nivel") or "-",
                                "Lectura": _motivo_escaner(_r)})
            _barra.progress((_i + 1) / len(PARES))
        _barra.empty()
        st.session_state.escaneo_global = _resultados
        st.session_state.escaneo_ts = ahora_local().strftime("%H:%M:%S")
    _esc = st.session_state.get("escaneo_global")
    if _esc:
        _con_idx = [r for r in _esc if r["Índice"] is not None]
        if _con_idx:
            _mejor = max(_con_idx, key=lambda r: r["Índice"])
            st.success(f"🏆 Mejor par ahora: {_mejor['Par']} — {_mejor['Estado']} (índice {_mejor['Índice']}) · "
                       f"señal tentativa {_mejor['Señal tentativa']} · {_mejor['Lectura']} · escaneo de las {st.session_state.get('escaneo_ts', '')}")
        _orden = sorted(_esc, key=lambda r: (r["Índice"] is None, -(r["Índice"] or 0)))
        st.dataframe(pd.DataFrame(_orden), width="stretch", hide_index=True)
        _c1, _c2 = st.columns([3, 1])
        _ir = _c1.selectbox("Ir a operar en:", [r["Par"] for r in _orden], key="escaneo_ir")
        if _c2.button("➡️ Cambiar par", width="stretch"):
            st.session_state.par_sel = _ir
            st.rerun()

# ---------------- Panel IA v3 (solo Admin): backtesting + entrenamiento ----------------
_info_modelo = modelo_v3_info()
if st.session_state.usuario_actual.startswith("👑"):
    with st.expander("🧠 IA v3 — Entrenar con backtesting"):
        if _info_modelo:
            _m = _info_modelo["metricas"]
            st.success(f"Modelo activo — entrenado el {_m['fecha_entrenamiento']} con {_m['ejemplos_entrenamiento']} ejemplos. Brier: {_m['brier']}")
            st.dataframe(pd.DataFrame(_m["por_umbral"]), width="stretch", hide_index=True)
            if _m.get("por_regimen"):
                st.caption("Precisión por régimen de mercado (umbral 0.62):")
                st.dataframe(pd.DataFrame(_m["por_regimen"]), width="stretch", hide_index=True)
            st.caption("Precisión = % de acierto de las señales emitidas en datos que el modelo nunca vio. Cobertura = % de ciclos en los que sí emite señal.")
        else:
            st.info("Todavía no hay modelo entrenado. Sin modelo, la app usa solo el filtro de puntaje técnico.")
        _codigos_crypto = [v["codigo"] for v in PARES.values() if v["tipo"] == "crypto"]
        _sel = st.multiselect("Pares crypto para el backtest", _codigos_crypto,
                              default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        _dias = st.slider("Días de historial (velas 1m)", 7, 30, 14)
        if st.button("🚀 Entrenar modelo con backtest", width="stretch"):
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
                st.dataframe(pd.DataFrame(_info["metricas"]["por_umbral"]), width="stretch", hide_index=True)
                if _info["metricas"].get("por_regimen"):
                    st.dataframe(pd.DataFrame(_info["metricas"]["por_regimen"]), width="stretch", hide_index=True)
                st.success("Modelo activo. A partir de ahora las señales crypto se filtran con su probabilidad calibrada (mínimo 62%).")
            except Exception as _e:
                st.error(f"No se pudo entrenar: {_e}")
    with st.expander("💬 Diario inteligente — IA que razona (v3.4)"):
        if not ia_configurar_desde_secrets():
            st.info("Sin clave de IA configurada. En los Secrets de Streamlit agrega: ia_api_key = \"tu clave gratis\" e ia_proveedor = \"gemini\" (o \"groq\" / \"openrouter\"). Claves gratis: Google AI Studio (aistudio.google.com) o Groq (console.groq.com).")
        else:
            st.caption(f"IA conectada (proveedor: {_secret('ia_proveedor', 'IA_PROVEEDOR') or 'gemini'}). Segunda opinión por señal: {'ACTIVADA' if _ia_bool('ia_segunda_opinion') else 'apagada — actívala con ia_segunda_opinion = true en Secrets'}. Clima de sesión: {'ACTIVADO' if _ia_bool('ia_clima') else 'apagado — ia_clima = true'}.")
            if st.button("🧠 Analizar mi historial con IA", width="stretch"):
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
                    _aut = nucleo_ia.autopsia_perdidas(_hist)
                    if _aut:
                        _tot_perd = sum(_aut.values())
                        _lineas.append("Autopsia de pérdidas (causa probable): " +
                                       "; ".join(f"{c}: {n} de {_tot_perd}" for c, n in _aut.items()))
                    _pos_d = nucleo_ia.postura_sesion(_hist)
                    _lineas.append(f"Postura de la sesión según resultados recientes: {_pos_d[0]} "
                                   f"(acierto {_pos_d[1]}% en las últimas {_pos_d[2]} operaciones)")
                with st.spinner("La IA está leyendo tu historial..."):
                    _informe = razonamiento_ia.analizar_historial("\n".join(_lineas))
                if _informe:
                    st.markdown(_informe)
                else:
                    # v3.17.5: el modulo ahora dice LA CAUSA (sobrecarga,
                    # limite, clave...). Si el modulo del repo es viejo y no
                    # trae ultimo_error(), se muestra el mensaje de siempre.
                    _motivo = ""
                    try:
                        _motivo = razonamiento_ia.ultimo_error() or ""
                    except Exception:
                        _motivo = ""
                    st.error(_motivo or "La IA no respondió. Revisa tu clave de API o inténtalo de nuevo.")
    with st.expander("💾 Respaldo del aprendizaje — guarda tus operaciones"):
        st.caption("El historial y el cerebro en vivo viven solo en el servidor de Streamlit y se pueden perder al reiniciar o volver a subir la app. Descarga una copia antes de subir una versión nueva y guárdala junto a los archivos de la app: si se pierde, subir ese archivo con la app lo restaura.")
        _hist_respaldo = st.session_state.hist_cache
        st.download_button("⬇️ Descargar historial.json (operaciones reales)",
                           data=json.dumps(_hist_respaldo, indent=2, ensure_ascii=False),
                           file_name=f"historial_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", width="stretch", key="dl_historial_respaldo")
        st.download_button("⬇️ Descargar patrones.json (contadores)",
                           data=json.dumps(st.session_state.pesos, indent=2, ensure_ascii=False),
                           file_name=f"patrones_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", width="stretch")
        if os.path.exists(ARCHIVO_EVOLUCION):
            with open(ARCHIVO_EVOLUCION, "r", encoding="utf-8") as _f:
                st.download_button("⬇️ Descargar evolucion.json (fotos del Panel de Evolución)",
                                   data=_f.read(),
                                   file_name=f"evolucion_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                                   mime="application/json", width="stretch")
            st.caption("Si evolucion.json se pierde, el panel vuelve a empezar desde la próxima foto; el historial y el cerebro no se pierden.")
        if os.path.exists(ARCHIVO_MODELO_VIVO):
            with open(ARCHIVO_MODELO_VIVO, "rb") as _f:
                st.download_button("⬇️ Descargar modelo_vivo.pkl (cerebro en vivo)",
                                   data=_f.read(),
                                   file_name=f"modelo_vivo_{ahora_local().strftime('%Y%m%d_%H%M')}.pkl",
                                   mime="application/octet-stream", width="stretch")
        else:
            st.caption("Todavía no hay modelo en vivo (se crea solo al llegar a 60 operaciones reales).")
        st.caption(f"Operaciones en el historial ahora: {len(_hist_respaldo)}.")
        _uso_td = _uso_twelvedata()
        if _uso_td and _uso_td[0] is not None:
            st.caption(f"📊 Créditos Twelve Data usados hoy: {_uso_td[0]} de {_uso_td[1]}. "
                       f"Si se acaban, la app cae a Yahoo (precio retrasado) hasta el día siguiente.")
        st.caption("📡 Finnhub (precio juez forex): " + ("clave configurada ✓ — juzga entradas y cierres sin gastar créditos Twelve."
                   if _finnhub_key() else "sin clave en Secrets (finnhub_api_key); el precio juez sigue siendo Twelve/Yahoo."))
        # Prueba de fuentes (v3.16.1): un toque y la app muestra el precio
        # del par seleccionado segun las tres fuentes, casi al mismo
        # instante, con las diferencias en pips. Es la prueba de los
        # 0,2 pips sin salir de la app: Jose compara esos numeros con el
        # precio de IQ Option en el mismo momento.
        _par_prueba = PARES.get(st.session_state.get("par_sel", ""), {})
        _cod_prueba = _par_prueba.get("codigo", "EURUSD=X")
        _nombre_prueba = st.session_state.get("par_sel", "")
        if _par_prueba.get("tipo", "forex") != "forex":
            # v3.17.3: el boton ya no se esconde en cripto; la prueba usa EUR/USD.
            _cod_prueba = "EURUSD=X"
            _nombre_prueba = "EUR/USD (prueba)"
        if st.button("🧪 Probar Finnhub contra Twelve y Yahoo (par seleccionado)", width="stretch"):
            _pf = _precio_finnhub(_cod_prueba)
            _pt = _precio_twelvedata(_cod_prueba)
            _py = _precio_yahoo(_cod_prueba)
            _pip = 0.01 if "JPY" in _cod_prueba else 0.0001
            def _fmt(v):
                return f"{v:.5f}" if v else "sin respuesta"
            def _dif(a, b):
                return f"{abs(a - b) / _pip:.1f} pips" if (a and b) else "—"
            st.caption(f"🧪 {_nombre_prueba} · {ahora_local().strftime('%H:%M:%S')} — "
                       f"Finnhub: {_fmt(_pf)} · Twelve Data: {_fmt(_pt)} · Yahoo: {_fmt(_py)} · "
                       f"Finnhub↔Twelve: {_dif(_pf, _pt)} · Finnhub↔Yahoo: {_dif(_pf, _py)}. "
                       f"Compáralos con IQ Option en este mismo momento: la fuente buena queda a 1 pip o menos de IQ.")
        if os.path.exists("historial_respaldo_auto.json"):
            with open("historial_respaldo_auto.json", "r", encoding="utf-8") as _f:
                st.download_button("🛟 Descargar copia automática pre-Reset (historial)",
                                   data=_f.read(),
                                   file_name="historial_respaldo_auto.json",
                                   mime="application/json", width="stretch")
            st.caption("Esa copia se guardó sola justo antes del último Reset. Para deshacerlo: "
                       "renómbrala como historial.json, súbela junto a los archivos de la app y reinicia la app.")
    with st.expander("🔐 Seguridad y licencias — crear claves y revocar accesos"):
        st.caption("Las claves generadas aqui son aleatorias (imposibles de adivinar por patron) y la licencia "
                   "empieza a contar desde hoy. Si el enlace personal de un cliente se filtra (una captura, un "
                   "reenvio), revoca su dispositivo: ese enlace muere y el cliente vuelve a entrar con su clave.")
        _dias_lic = st.slider("Días de vigencia de la nueva licencia", 7, 365, 30)
        if st.button("🎫 Generar licencia nueva", width="stretch"):
            import secrets as _secrets
            _alf = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
            _nueva = "ZT-" + "-".join("".join(_secrets.choice(_alf) for _ in range(6)) for _ in range(3))
            _lics = cargar_licencias()
            _vence = (ahora_local().date() + timedelta(days=int(_dias_lic))).strftime("%Y-%m-%d")
            _lics[_nueva] = {"vence": _vence, "device_id": None}
            guardar_licencias(_lics)
            st.session_state.ultima_licencia = (_nueva, _vence)
        if st.session_state.get("ultima_licencia"):
            _k, _v = st.session_state.ultima_licencia
            st.success(f"Licencia creada — vence el {_v}. Copiala ahora y enviasela a tu cliente:")
            st.code(_k)
        _lics2 = cargar_licencias()
        _con_dev = [(_k, _v) for _k, _v in _lics2.items() if _v.get("device_id")]
        if _con_dev:
            st.markdown("**Dispositivos con acceso guardado (enlace personal activo):**")
            for _k, _v in _con_dev:
                _c1, _c2 = st.columns([3, 1])
                _c1.caption(f"🔑 {_k[:12]}... · vence {_v.get('vence')}")
                if _c2.button("Revocar", key=f"rev_{_k[:16]}"):
                    _v["device_id"] = None
                    _lics2[_k] = _v
                    guardar_licencias(_lics2)
                    st.rerun()
        else:
            st.caption("Ninguna licencia tiene dispositivo vinculado todavia.")
elif _info_modelo:
    st.caption(f"🧠 IA v3 activa — modelo entrenado el {_info_modelo['metricas']['fecha_entrenamiento']}")


# ---------------- Panel de aprendizaje en vivo (visible para todos) ----------------
# v3.17.3: Aprendizaje en vivo solo lo ve el Admin Master; los clientes no ven conteos ni rendimiento.
if st.session_state.usuario_actual.startswith("👑"):
    with st.expander("📈 Aprendizaje en vivo — así va aprendiendo la app"):
        _hist = st.session_state.hist_cache
        # v3.17.1: la descarga del historial solo existia en el Respaldo, que
        # solo se muestra con sesion de Admin Master; con sesion de licencia
        # no se veia. Ahora tambien esta aqui, visible para todos.
        # v3.17.2: key distinta obligatoria — sin ella, en sesion Admin este
        # boton chocaba con el identico del Respaldo (StreamlitDuplicateElementId).
        st.download_button("⬇️ Descargar historial.json (operaciones reales)",
                           data=json.dumps(_hist, indent=2, ensure_ascii=False),
                           file_name=f"historial_{ahora_local().strftime('%Y%m%d_%H%M')}.json",
                           mime="application/json", width="stretch", key="dl_historial_aprendizaje")
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
            _n_empates = sum(1 for _h in _hist if _h.get("resultado") == "EMPATE")
            if _n_empates:
                st.caption(f"↔️ Empates (el precio de cierre fue igual al de entrada; en IQ Option te devuelven el monto): "
                           f"{_n_empates}. No cuentan como pérdida ni entran al aprendizaje del cerebro.")
            if _n_val >= 20:
                _espejo = round((_n_val - _ganadas) / _n_val * 100, 1)
                st.caption(f"🔁 Señal espejo: operando TODO al revés de lo que dijo la app, el acierto habría sido {_espejo}%. "
                           f"Si el espejo supera al derecho por mucho, hay un sesgo que corregir, no mala suerte.")
                for _dir in ("CALL", "PUT"):
                    _sub = [_h for _h in _validas if _h.get("señal") == _dir]
                    if len(_sub) >= 20:
                        _wr = sum(1 for _h in _sub if _h["resultado"] == "GANADA") / len(_sub) * 100
                        if _wr < 40:
                            st.warning(f"⚠️ Sesgo {_dir}: solo {_wr:.1f}% de acierto en {len(_sub)} operaciones. "
                                       f"Esa dirección está rindiendo muy por debajo del azar; el cerebro lo tiene en cuenta al entrenar.")
                        elif _wr >= 58:
                            st.success(f"✅ {_dir} viene fuerte: {_wr:.1f}% de acierto en {len(_sub)} operaciones.")
            _limpias = sum(1 for _h in _validas if _h.get("fuente_datos") in ("Twelve Data", "Binance"))
            st.caption(f"🎯 Operaciones con etiqueta limpia (juzgadas con precio sincronizado de Twelve Data o Binance): "
                       f"{_limpias} de {_n_val}. Al llegar a {nucleo_ia.MIN_OPS_MODELO_VIVO} limpias, el cerebro entrenará solo con esas "
                       f"y dejará fuera las que se juzgaron con el precio retrasado de Yahoo.")
            _n_finnhub = sum(1 for _h in _validas if _h.get("fuente_datos") == "Finnhub")
            if _n_finnhub:
                st.caption(f"📡 Operaciones juzgadas por Finnhub (precio en vivo sin techo diario): {_n_finnhub} de {_n_val}. "
                           f"Su etiqueta está EN VERIFICACIÓN: contará como limpia cuando se confirme en vivo que su precio "
                           f"va parejo con IQ Option (como se hizo con Twelve Data).")
            _postura, _post_wr, _post_n = nucleo_ia.postura_sesion(_hist)
            if _postura == "REDUCIR":
                st.warning(f"🧭 Postura de la sesión: **REDUCIR** — el acierto reciente es {_post_wr}% en las últimas {_post_n} operaciones. "
                           f"La recomendación del entrenador de exposición: opera solo señales de Nivel Alta o descansa el par un rato.")
            elif _postura == "NEUTRA":
                st.info(f"🧭 Postura de la sesión: **NEUTRA** — acierto reciente {_post_wr}% en las últimas {_post_n} operaciones. Sé selectivo: prioriza Nivel Alta y Media.")
            elif _postura == "PLENA":
                st.success(f"🧭 Postura de la sesión: **PLENA** — acierto reciente {_post_wr}% en las últimas {_post_n} operaciones. El sistema viene rindiendo; mantén la disciplina igual.")
            _autopsia = nucleo_ia.autopsia_perdidas(_hist)
            _n_perdidas = sum(_autopsia.values())
            if _n_perdidas:
                st.caption("🔬 Autopsia de pérdidas — de qué están muriendo las operaciones:")
                if _n_perdidas >= 10:
                    _filas_aut = [{"causa probable": c, "pérdidas": n,
                                   "%": round(n / _n_perdidas * 100, 1)} for c, n in _autopsia.items()]
                else:
                    _filas_aut = [{"causa probable": c, "pérdidas": n,
                                   "%": "muestra chica"} for c, n in _autopsia.items()]
                st.dataframe(pd.DataFrame(_filas_aut), width="stretch", hide_index=True)
            _lecciones = nucleo_ia.lecciones_aprendidas(_hist)
            if _lecciones:
                st.caption("📓 Lecciones de la experiencia — lo que la app aprendió de sus propias pérdidas. "
                           "Ya se aplican solas a las señales nuevas (bajan el nivel en los contextos que castigan):")
                for _l in _lecciones:
                    st.info(f"**{_l['titulo']}**\n\n🔍 Evidencia: {_l['evidencia']}.\n\n🛠 Ajuste activo: {_l['efecto']}")
            elif _n_perdidas >= 10:
                st.caption("📓 Todavía sin lecciones automáticas: se activan cuando una causa de pérdida domina "
                           "con 20+ pérdidas registradas (muestra suficiente para no aprender de ruido).")
            _c1, _c2 = st.columns(2)
            with _c1:
                st.caption("Acierto por par")
                st.dataframe(_tabla_vivo("par", "par"), width="stretch", hide_index=True)
                st.caption("Acierto por dirección")
                st.dataframe(_tabla_vivo("señal", "dirección"), width="stretch", hide_index=True)
            with _c2:
                st.caption("Acierto por nivel de señal")
                st.dataframe(_tabla_vivo("nivel", "nivel"), width="stretch", hide_index=True)
                st.caption("Acierto por tipo")
                st.dataframe(_tabla_vivo("tipo", "tipo"), width="stretch", hide_index=True)

# ---------------- Panel de Evolucion (v3.17) ----------------
# La evolucion no se siente, se mide: tabla de fotos cada 50 ops y el
# antes vs ahora con los cambios en puntos. Desde v3.17.3 solo lo ve el
# Admin Master: los clientes no ven conteos ni rendimiento.
# v3.17.3: Panel de Evolucion solo lo ve el Admin Master; los clientes no ven conteos ni rendimiento.
if st.session_state.usuario_actual.startswith("👑"):
    with st.expander("📈 Panel de Evolución — antes vs ahora (v3.17)", expanded=False):
        st.caption("Cada 50 operaciones válidas la app guarda sola una foto de los indicadores "
                   "(acierto, espejo, precisión del cerebro en datos no vistos, etiquetas limpias, "
                   "juzgadas por Finnhub y postura). La primera foto es 'hoy'; a partir de ahí el "
                   "'antes vs ahora' se construye solo. Metas: precisión no visto 55% · acierto "
                   "limpio 53,8% (IQ Option) / 51,3% (Exnova).")
        _fotos_ev = leer_json(ARCHIVO_EVOLUCION, [])
        if st.button("📸 Guardar foto ahora", width="stretch", key="evo_foto_ahora"):
            _f_ahora = foto_evolucion_actual()
            if not _f_ahora or _f_ahora["ops"] < 50:
                st.info("Todavía no hay 50 operaciones válidas para la primera foto.")
            elif _fotos_ev and _fotos_ev[-1].get("ops", 0) // 50 >= _f_ahora["ops"] // 50:
                st.info("Ya hay una foto de este tramo de 50 operaciones; la próxima se guarda sola al cruzar el siguiente múltiplo de 50.")
            else:
                _fotos_ev.append(_f_ahora)
                guardar_json_atomico(ARCHIVO_EVOLUCION, _fotos_ev)
                st.success(f"Foto guardada: {_f_ahora['ops']} operaciones, acierto {_f_ahora['acierto_%']}%.")
            _fotos_ev = leer_json(ARCHIVO_EVOLUCION, [])
        if _fotos_ev:
            _filas_ev = [{"fecha": f.get("fecha", "—"), "operaciones": f.get("ops", "—"),
                          "acierto %": f.get("acierto_%", "—"), "espejo %": f.get("espejo_%", "—"),
                          "precisión cerebro (no visto) %": f.get("precision_no_visto_%", "—"),
                          "limpias": f.get("limpias", "—"), "Finnhub": f.get("finnhub", "—"),
                          "postura": f.get("postura", "—")} for f in _fotos_ev]
            st.dataframe(pd.DataFrame(_filas_ev), width="stretch", hide_index=True)
            if len(_fotos_ev) >= 2:
                _prim, _ult = _fotos_ev[0], _fotos_ev[-1]
                def _cambio(clave, bueno_subir=True, unidad="%"):
                    _a, _b = _prim.get(clave), _ult.get(clave)
                    if _a is None or _b is None:
                        return "sin dato"
                    _d = round(_b - _a, 1)
                    _signo = "+" if _d > 0 else ""
                    _bien = "✅" if (_d > 0) == bueno_subir and _d != 0 else ("⚠️" if _d != 0 else "＝")
                    return f"{_a}{unidad} → {_b}{unidad} ({_signo}{_d} pts) {_bien}"
                st.caption(f"📊 Antes vs ahora — de {_prim.get('fecha')} ({_prim.get('ops')} ops) a {_ult.get('fecha')} ({_ult.get('ops')} ops):")
                st.caption(f"· Acierto: {_cambio('acierto_%', True)}")
                st.caption(f"· Espejo (bajar es bueno): {_cambio('espejo_%', False)}")
                st.caption(f"· Precisión del cerebro en datos no vistos: {_cambio('precision_no_visto_%', True)}")
                st.caption(f"· Etiquetas limpias: {_cambio('limpias', True, '')}")
                _sube_acierto = (_ult.get("acierto_%") or 0) > (_prim.get("acierto_%") or 0)
                _sube_cerebro = (_ult.get("precision_no_visto_%") is not None and _prim.get("precision_no_visto_%") is not None
                                 and _ult["precision_no_visto_%"] > _prim["precision_no_visto_%"])
                if _sube_acierto and _sube_cerebro:
                    st.success("✅ Evolucionando: suben a la vez el acierto y la precisión del cerebro en datos que nunca vio. Eso es aprendizaje real, no suerte de una sesión.")
                elif _sube_acierto or _sube_cerebro:
                    st.info("🟡 A medias: uno de los dos sube y el otro no. Mira cuál se mueve cuando lleguen 50 operaciones más.")
                else:
                    st.warning("⚠️ Estancada por ahora: ni el acierto ni la precisión del cerebro suben entre fotos. Si las limpias tampoco crecen, el freno es la falta de datos limpios, no la estrategia.")
            else:
                st.caption("Con una sola foto todavía no hay 'antes vs ahora': la comparación aparece sola cuando se guarde la segunda foto (al cruzar el siguiente múltiplo de 50 operaciones).")
        else:
            _f_ahora = foto_evolucion_actual()
            if _f_ahora and _f_ahora["ops"] >= 50:
                st.caption(f"Ya llevas {_f_ahora['ops']} operaciones válidas y aún no hay fotos: pulsa 'Guardar foto ahora' para tomar la primera, y las siguientes se guardan solas cada 50.")
            else:
                st.caption("Aún no hay fotos: la primera se guarda sola al llegar a 50 operaciones válidas.")

# ---------------- Catalogador ZT (v3.14) ----------------
# La idea del panel de DAM Trader en version honesta: efectividad MEDIDA
# por par x direccion (con las perdidas visibles), filtros de etiquetas
# limpias y "solo con tendencia", y estados con muestra minima.
# v3.17.3: Catalogador ZT solo lo ve el Admin Master; los clientes no ven conteos ni rendimiento.
if st.session_state.usuario_actual.startswith("👑"):
    with st.expander("📚 Catalogador ZT — qué está funcionando, medido", expanded=False):
        _nombres_par = {v["codigo"]: k for k, v in PARES.items()}
        _cf1, _cf2 = st.columns(2)
        with _cf1:
            _cat_limpias = st.checkbox("🧼 Solo etiquetas limpias (Twelve Data/Binance)", value=True, key="cat_solo_limpias")
        with _cf2:
            _cat_tend = st.checkbox("📈 Solo con tendencia", value=False, key="cat_solo_tendencia")
        _cat = nucleo_ia.catalogo(st.session_state.hist_cache, solo_limpias=_cat_limpias, solo_tendencia=_cat_tend)
        if _cat["total"] < 10:
            st.caption(f"Muestra insuficiente con estos filtros: {_cat['total']} operaciones. El catálogo se llena solo, operando.")
        else:
            _m, _p = _cat["mejor"], _cat["peor"]
            if _m:
                _mn = _nombres_par.get(_m["par"], _m["par"])
                if _m["estado"].startswith("🟢"):
                    st.success(f"🏆 Tu mejor ficha medida: **{_mn} {_m['direccion']}** — {_m['acierto']}% en {_m['ops']} operaciones. Supera el 53,8% de equilibrio.")
                else:
                    st.info(f"🏆 La ficha menos mala por ahora: **{_mn} {_m['direccion']}** — {_m['acierto']}% en {_m['ops']} operaciones. Ninguna supera aún el 53,8% con muestra suficiente.")
                if _mn in PARES and st.button(f"🎯 Operar {_mn} ahora", key="cat_operar_mejor", width="stretch"):
                    st.session_state.par_sel = _mn
                    reiniciar_ciclo()
                    st.rerun()
            if _p and _p["estado"].startswith("🔴"):
                st.warning(f"🔴 Para descansar: **{_nombres_par.get(_p['par'], _p['par'])} {_p['direccion']}** — {_p['acierto']}% en {_p['ops']} operaciones.")
            _filas_cat = [{"Par": _nombres_par.get(f["par"], f["par"]), "Dirección": f["direccion"], "Ops": f["ops"],
                           "W": f["ganadas"], "L": f["perdidas"], "Acierto %": f["acierto"],
                           "Limpias": f["limpias"], "Estado": f["estado"]} for f in _cat["filas"]]
            st.dataframe(pd.DataFrame(_filas_cat), width="stretch", hide_index=True)
            if _cat["regimenes"]:
                st.caption("Acierto por régimen de mercado (con los mismos filtros):")
                st.dataframe(pd.DataFrame([{"Régimen": r["regimen"], "Ops": r["ops"], "Acierto %": r["acierto"]} for r in _cat["regimenes"]]),
                             width="stretch", hide_index=True)
            st.caption("🟢 OPERABLE = ≥20 ops y acierto ≥53,8% · 🔴 DESCANSAR = ≥15 ops y acierto <45% · "
                       "🟡 EN OBSERVACIÓN = muestra corta o acierto intermedio · ⚪ SIN MUESTRA = <10 ops (no concluir).")

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

def _fmt_precio(p, codigo):
    """Formato de precio segun el par (JPY 3 decimales, cripto grande 2)."""
    if p is None:
        return "--.-----"
    p = float(p)
    if "JPY" in str(codigo):
        return f"{p:,.3f}"
    if abs(p) >= 100:
        return f"{p:,.2f}"
    return f"{p:.5f}"

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
        html += f'<div class="info-panel">🧠 IA ({op["veredicto"]}): {_html.escape(str(op["motivo"]))}</div>'
    return html

def cerrar_operacion(codigo_op, tipo_op):
    """Evalua la operacion abierta con el precio actual y la registra una sola vez."""
    señal = st.session_state.senal
    entrada = st.session_state.precio_real_entrada
    if señal is None or señal.get('señal') == 'ESPERA' or not entrada or st.session_state.resultado_guardado:
        st.session_state.resultado_guardado = True
        return
    precio_actual, _fuente_precio = obtener_precio_actual_con_fuente(codigo_op, tipo_op)
    resultado = 'PERDIDA'
    if precio_actual is not None:
        # EMPATE (v3.9.1): si el cierre es IGUAL a la entrada, en IQ Option
        # te devuelven el monto: no es perdida. Contarlo como PERDIDA
        # envenenaba las estadisticas y el entrenamiento, sobre todo en
        # rangos (precio casi inmovil) y con los precios escalonados de
        # Yahoo. El empate no entra al aprendizaje (ops_validas lo excluye).
        _dec = 3 if "JPY" in str(codigo_op) else 5
        if round(float(precio_actual), _dec) == round(float(entrada), _dec):
            resultado = 'EMPATE'
        else:
            es_call = señal['señal'] == 'CALL'
            if (es_call and precio_actual > entrada) or (not es_call and precio_actual < entrada):
                resultado = 'GANADA'
    aprender(señal['señal'], resultado, señal.get('indicadores', {}), entrada, señal.get('features'),
             par=codigo_op, tipo=tipo_op, nivel=señal.get('nivel'), confianza=señal.get('confianza'),
             fuente=(_fuente_precio or señal.get('fuente')))
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

    # Estadisticas + reloj (v3.17.3: marcador y estadisticas solo los ve
    # el Admin Master; los clientes ven solo el indicador EN VIVO).
    pesos = st.session_state.pesos
    if st.session_state.usuario_actual.startswith("👑"):
        _hw = sum(1 for _h in st.session_state.hist_cache if _h.get('resultado') == 'GANADA')
        _hl = sum(1 for _h in st.session_state.hist_cache if _h.get('resultado') == 'PERDIDA')
        st.markdown(f'<div class="brand-bar"><span class="pill-record">🔥 {_hw}W - {_hl}L</span><span class="pill-live">● EN VIVO</span></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box"><strong>Winrate</strong><span>{round(pesos['winrate'], 1)}%</span></div>
            <div class="stat-box"><strong>Ops</strong><span>{pesos['total_ops']}</span></div>
            <div class="stat-box"><strong>Win</strong><span>{pesos['total_ganadas']}</span></div>
            <div class="stat-box"><strong>ML</strong><span style="font-size:11px;">{len(st.session_state.hist_cache)} ops</span></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="brand-bar"><span class="pill-live">● EN VIVO</span></div>', unsafe_allow_html=True)
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
                        _postura_txt = nucleo_ia.postura_sesion(st.session_state.hist_cache)
                        op = razonamiento_ia.segunda_opinion({
                            'par': codigo_f, 'tipo': tipo_f, 'direccion': pre['señal'],
                            'confianza': pre['confianza'], 'nivel': pre.get('nivel') or '-',
                            'razon': pre['razon'], 'regimen': '',
                            'postura': f"{_postura_txt[0]} (acierto reciente: {_postura_txt[1]}% en {_postura_txt[2]} ops)",
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
                    _pos = nucleo_ia.postura_sesion(st.session_state.hist_cache)
                    txt = razonamiento_ia.clima_sesion(codigo_f, tipo_f, round(var, 2),
                        f"segun el analisis del ciclo; postura de la sesión: {_pos[0]} (acierto reciente {_pos[1]}% en {_pos[2]} operaciones)")
                    if txt:
                        st.session_state.clima_ia = txt
            except Exception:
                pass

        clima_html = f'<div class="info-panel">🧠 Clima IA: {_html.escape(str(st.session_state.clima_ia))}</div>' if st.session_state.clima_ia else ''
        try:
            _precio_vivo = obtener_precio_actual(codigo_f, tipo_f)
        except Exception:
            _precio_vivo = None
        _precio_txt = _fmt_precio(_precio_vivo, codigo_f)
        _entrada_txt = (ahora + timedelta(seconds=(60 - segundo_actual))).strftime("%H:%M:%S")
        _pct_ciclo = min(100, int(segundo_actual / 55 * 100))
        contenedor.markdown(f"""
        <div class="signal-box signal-wait">📊 ESCANEANDO MERCADO...</div>
        {clima_html}
        <div class="live-grid">
          <div class="radar-container"><div class="radar radar-yellow"><div class="radar-center"><div class="radar-number radar-number-yellow">IA</div><div class="radar-label">ACTIVA</div></div></div></div>
          <div class="data-rows">
            <div class="data-row"><span>⏱ ENTRADA EXACTA</span><strong class="amarillo">{_entrada_txt}</strong></div>
            <div class="data-row"><span>PUNTO DE ENTRADA</span><strong>{_precio_txt}</strong></div>
            <div class="data-row"><span>● EN VIVO</span><strong class="verde">{_precio_txt}</strong></div>
            <div class="data-row"><span>EXPIRACIÓN</span><strong>SIGUIENTE VELA (:00)</strong></div>
            <div class="data-row"><span>HORA SEÑAL</span><strong>{hora_actual}</strong></div>
          </div>
        </div>
        <div class="progress-track"><div class="progress-fill" style="width:{_pct_ciclo}%"></div></div>
        <div class="info-panel">⏱️ Próxima señal en: <strong>{55 - segundo_actual}s</strong></div>
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
        _entrada_txt2 = (ahora + timedelta(seconds=(60 - segundo_actual))).strftime("%H:%M:%S")
        contenedor.markdown(
            voz_html
            + barra_confianza_html(confianza_actual, "NIVEL DE CONFIANZA DE LA SEÑAL")
            + nivel_html_de(señal)
            + f"""
            <div class="signal-box {clase_box}">{tipo_signal}</div>
            <div class="live-grid">
              <div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes}</div><div class="radar-label">SEGUNDOS</div></div></div></div>
              <div class="data-rows">
                <div class="data-row"><span>⏱ ENTRADA EXACTA</span><strong class="amarillo">{_entrada_txt2}</strong></div>
                <div class="data-row"><span>EXPIRACIÓN</span><strong>SIGUIENTE VELA (:00)</strong></div>
                <div class="data-row"><span>HORA SEÑAL</span><strong>{hora_actual}</strong></div>
              </div>
            </div>
            <div class="info-panel">Razón: <strong>{señal["razon"]}</strong></div>
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

    # Precio EN VIVO durante la operacion (v3.6.1): la Entrada es la foto
    # del segundo :00 y no se mueve; este precio si se mueve, para que el
    # usuario compare contra IQ Option en el mismo instante y vea si va
    # ganando o perdiendo y por cuanto. En v3.11.1 va en filas de datos
    # junto al radar (misma tarjeta compacta de las fases 1 y 2).
    _ent = st.session_state.precio_real_entrada
    _precio_ahora = obtener_precio_actual(codigo_f, tipo_f)
    _vivo_txt = "--.-----"
    _estado_txt = "--"
    _estado_clase = ""
    if _precio_ahora and _ent:
        _dif = float(_precio_ahora) - float(_ent)
        _va_ganando = (_dif > 0) if es_call else (_dif < 0)
        if tipo_f == "forex":
            _pip = 0.01 if "JPY" in str(codigo_f) else 0.0001
            _dif_txt = f"{_dif / _pip:+.1f} pips"
        else:
            _dif_txt = f"{_dif / float(_ent) * 100:+.3f}%"
        _vivo_txt = _fmt_precio(_precio_ahora, codigo_f)
        _estado_txt = f"{'GANANDO' if _va_ganando else 'PERDIENDO'} · {_dif_txt}"
        _estado_clase = "verde" if _va_ganando else "rojo"

    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    tipo_signal = "🟢 EN CURSO (CALL)" if es_call else "🔴 EN CURSO (PUT)"

    contenedor.markdown(
        barra_confianza_html(confianza_actual, "CONFIANZA DE LA OPERACIÓN EN CURSO")
        + nivel_html_de(señal)
        + f"""
        <div class="signal-box {clase_box}">{tipo_signal}</div>
        <div class="live-grid">
          <div class="radar-container"><div class="radar {clase_radar}"><div class="radar-center"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div><div class="radar-label">EXPIRACIÓN</div></div></div></div>
          <div class="data-rows">
            <div class="data-row"><span>PUNTO DE ENTRADA</span><strong>{_fmt_precio(_ent, codigo_f)}</strong></div>
            <div class="data-row"><span>● EN VIVO</span><strong>{_vivo_txt}</strong></div>
            <div class="data-row"><span>Vas</span><strong class="{_estado_clase}">{_estado_txt}</strong></div>
            <div class="data-row"><span>HORA SEÑAL</span><strong>{hora_actual}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cierre al final del minuto de operacion (seg 57-59). El rescate de la
    # Fase 1 cubre el caso de que esta ventana se pierda.
    if segundo_actual >= 57 and not st.session_state.resultado_guardado:
        cerrar_operacion(codigo_f, tipo_f)

with slot_zona:
    zona_dinamica(codigo, tipo)
st.markdown('</div>', unsafe_allow_html=True)
