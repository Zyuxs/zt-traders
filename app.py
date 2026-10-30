import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import time
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="ZT TRADERS", layout="centered", page_icon="")

ARCHIVO_HISTORIAL = "historial.json"
ARCHIVO_PATRONES = "patrones.json"
ARCHIVO_USUARIOS = "usuarios.json"

def cargar_usuarios():
    if os.path.exists(ARCHIVO_USUARIOS):
        try:
            with open(ARCHIVO_USUARIOS, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def verificar_usuario(usuario, clave):
    usuarios = cargar_usuarios()
    if usuario not in usuarios:
        return False, "Usuario no encontrado"
    datos = usuarios[usuario]
    if datos['clave'] != clave:
        return False, "Contraseña incorrecta"
    fecha_vence = datetime.strptime(datos['vence'], '%Y-%m-%d').date()
    hoy = datetime.now().date()
    if hoy > fecha_vence:
        return False, f"Tu membresia vencio el {datos['vence']}. Contacta al administrador."
    return True, datos.get('nombre', usuario)

if 'logueado' not in st.session_state:
    st.session_state.logueado = False
if 'usuario_actual' not in st.session_state:
    st.session_state.usuario_actual = ""

if not st.session_state.logueado:
    st.markdown("""
    <style>
    body { background: #000; color: #fff; font-family: Arial, sans-serif; }
    .login-container { max-width: 400px; margin: 80px auto; padding: 30px; background: linear-gradient(90deg, #1a1a2e, #16213e); border-radius: 15px; border: 2px solid #ffd700; box-shadow: 0 0 40px rgba(255, 215, 0, 0.3); }
    .login-title { color: #ffd700; text-align: center; font-size: 28px; font-weight: bold; margin-bottom: 10px; letter-spacing: 3px; }
    .login-subtitle { color: #aaa; text-align: center; font-size: 13px; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">ZT TRADERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="login-subtitle">Sistema de Acceso Privado</div>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        usuario_input = st.text_input("Usuario", placeholder="Ingresa tu usuario")
        clave_input = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
        submit = st.form_submit_button("ENTRAR", use_container_width=True)
        if submit:
            if usuario_input and clave_input:
                ok, mensaje = verificar_usuario(usuario_input, clave_input)
                if ok:
                    st.session_state.logueado = True
                    st.session_state.usuario_actual = mensaje
                    st.rerun()
                else:
                    st.error(mensaje)
            else:
                st.warning("Por favor completa todos los campos")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

def cargar_patrones():
    if os.path.exists(ARCHIVO_PATRONES):
        try:
            with open(ARCHIVO_PATRONES, 'r') as f:
                return json.load(f)
        except:
            os.remove(ARCHIVO_PATRONES)
    return {'total_ops': 0, 'total_ganadas': 0, 'winrate': 0.0}

def guardar_patrones(pesos):
    try:
        with open(ARCHIVO_PATRONES, 'w') as f:
            json.dump(pesos, f, indent=2)
    except:
        pass

def cargar_historial():
    if os.path.exists(ARCHIVO_HISTORIAL):
        try:
            with open(ARCHIVO_HISTORIAL, 'r') as f:
                return json.load(f)
        except:
            os.remove(ARCHIVO_HISTORIAL)
    return []

def guardar_historial(hist):
    try:
        with open(ARCHIVO_HISTORIAL, 'w') as f:
            json.dump(hist, f, indent=2)
    except:
        pass

def aprender_en_segundo_plano(señal, resultado, indicadores):
    hist = cargar_historial()
    pesos = cargar_patrones()
    registro = {
        'fecha': datetime.now().strftime('%H:%M:%S'),
        'señal': señal,
        'resultado': resultado,
        'indicadores': indicadores
    }
    hist.append(registro)
    pesos['total_ops'] += 1
    if resultado == 'GANADA':
        pesos['total_ganadas'] += 1
    pesos['winrate'] = (pesos['total_ganadas'] / pesos['total_ops'] * 100) if pesos['total_ops'] > 0 else 0.0
    guardar_historial(hist)
    guardar_patrones(pesos)

def entrenar_modelo_ml():
    hist = cargar_historial()
    X = []
    y = []
    for op in hist:
        if 'indicadores' in op:
            ind = op['indicadores']
            features = [
                ind.get('rsi', 50), ind.get('stoch', 50), ind.get('macd', 0),
                ind.get('adx', 20), ind.get('bb_position', 0.5),
                ind.get('sma_cross', 0), 1 if op['señal'] == 'CALL' else 0
            ]
            X.append(features)
            y.append(1 if op['resultado'] == 'GANADA' else 0)
    if len(X) < 20:
        return None
    try:
        modelo = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, min_samples_split=5)
        scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')
        accuracy_real = scores.mean()
        modelo.fit(X, y)
        return {'modelo': modelo, 'accuracy': accuracy_real, 'muestras': len(X)}
    except:
        return None

def predecir_con_ml(modelo, indicadores):
    features = [
        indicadores.get('rsi', 50), indicadores.get('stoch', 50),
        indicadores.get('macd', 0), indicadores.get('adx', 20),
        indicadores.get('bb_position', 0.5), indicadores.get('sma_cross', 0),
        1 if indicadores.get('direccion', 'CALL') == 'CALL' else 0
    ]
    try:
        probabilidad = modelo.predict_proba([features])[0]
        return probabilidad[1] * 100
    except:
        return 50.0

def señal_por_defecto(codigo_par, rsi=50.0):
    dirección = 'CALL' if rsi < 50 else 'PUT'
    indicadores = {'rsi': rsi, 'stoch': 50, 'macd': 0, 'adx': 20, 'bb_position': 0.5, 'sma_cross': 0, 'direccion': dirección}
    return {'señal': dirección, 'precio': 0.0, 'rsi': rsi, 'confianza': 10, 'razon': f"Señal por defecto (RSI {rsi:.0f})", 'indicadores': indicadores}

def analizar_par(codigo_par):
    try:
        datos_1m = yf.download(codigo_par, period="2d", interval="1m", progress=False)
        datos_5m = yf.download(codigo_par, period="5d", interval="5m", progress=False)
        if isinstance(datos_1m.columns, pd.MultiIndex):
            datos_1m.columns = datos_1m.columns.get_level_values(0)
            datos_5m.columns = datos_5m.columns.get_level_values(0)
        datos_1m = datos_1m[['Open', 'High', 'Low', 'Close']].copy()
        datos_5m = datos_5m[['Open', 'High', 'Low', 'Close']].copy()
        if len(datos_1m) < 30 or len(datos_5m) < 30:
            return señal_por_defecto(codigo_par)

        datos_1m['RSI'] = ta.momentum.RSIIndicator(datos_1m['Close'], window=14).rsi()
        datos_1m['STOCH'] = ta.momentum.StochasticOscillator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).stoch()
        datos_1m['MACD'] = ta.trend.MACD(datos_1m['Close']).macd()
        datos_1m['MACD_signal'] = ta.trend.MACD(datos_1m['Close']).macd_signal()
        datos_1m['ADX'] = ta.trend.ADXIndicator(datos_1m['High'], datos_1m['Low'], datos_1m['Close']).adx()
        bb = ta.volatility.BollingerBands(datos_1m['Close'], window=20, window_dev=2)
        datos_1m['BB_High'] = bb.bollinger_hband()
        datos_1m['BB_Low'] = bb.bollinger_lband()
        datos_1m['SMA_10'] = datos_1m['Close'].rolling(window=10).mean()
        datos_5m['SMA_20'] = datos_5m['Close'].rolling(window=20).mean()
        datos_5m['RSI'] = ta.momentum.RSIIndicator(datos_5m['Close'], window=14).rsi()

        u1 = datos_1m.iloc[-1]
        a1 = datos_1m.iloc[-2]
        u5 = datos_5m.iloc[-1]
        rsi = 50.0 if pd.isna(u1['RSI']) else float(u1['RSI'])
        stoch = 50.0 if pd.isna(u1['STOCH']) else float(u1['STOCH'])
        macd = 0.0 if pd.isna(u1['MACD']) else float(u1['MACD'])
        macd_sig = 0.0 if pd.isna(u1['MACD_signal']) else float(u1['MACD_signal'])
        adx = 20.0 if pd.isna(u1['ADX']) else float(u1['ADX'])
        precio = float(u1['Close'])
        bb_high = float(u1['BB_High'])
        bb_low = float(u1['BB_Low'])
        sma_1m = float(u1['SMA_10'])
        sma_5m = float(u5['SMA_20'])

        dirección = 'CALL' if rsi < 50 else 'PUT'
        if precio <= bb_low: dirección = 'CALL'
        elif precio >= bb_high: dirección = 'PUT'
        if precio > sma_1m and a1['Close'] <= a1['SMA_10']: dirección = 'CALL'
        elif precio < sma_1m and a1['Close'] >= a1['SMA_10']: dirección = 'PUT'

        bb_position = (precio - bb_low) / (bb_high - bb_low) if bb_high != bb_low else 0.5
        bb_position = max(0, min(1, bb_position))
        sma_cross = 0
        if precio > sma_1m and a1['Close'] <= a1['SMA_10']: sma_cross = 1
        elif precio < sma_1m and a1['Close'] >= a1['SMA_10']: sma_cross = -1

        indicadores = {'rsi': rsi, 'stoch': stoch, 'macd': macd - macd_sig, 'adx': adx, 'bb_position': bb_position, 'sma_cross': sma_cross, 'direccion': dirección}

        modelo_info = entrenar_modelo_ml()
        if modelo_info and modelo_info['modelo']:
            probabilidad_ml = predecir_con_ml(modelo_info['modelo'], indicadores)
            if probabilidad_ml >= 50:
                return {'señal': dirección, 'precio': precio, 'rsi': rsi, 'confianza': probabilidad_ml, 'razon': f"ML: {probabilidad_ml:.1f}% (Precisión: {modelo_info['accuracy']*100:.0f}%, Muestras: {modelo_info['muestras']})", 'indicadores': indicadores}

        confianza = 0
        razones = []
        if dirección == 'CALL' and rsi < 35: confianza += 25; razones.append("RSI sobreventa")
        elif dirección == 'PUT' and rsi > 65: confianza += 25; razones.append("RSI sobrecompra")
        if dirección == 'CALL' and precio <= bb_low: confianza += 25; razones.append("Toca banda inferior")
        elif dirección == 'PUT' and precio >= bb_high: confianza += 25; razones.append("Toca banda superior")
        if dirección == 'CALL' and macd > macd_sig and a1['MACD'] <= a1['MACD_signal']: confianza += 20; razones.append("MACD cruza alcista")
        elif dirección == 'PUT' and macd < macd_sig and a1['MACD'] >= a1['MACD_signal']: confianza += 20; razones.append("MACD cruza bajista")
        if adx > 25: confianza += 15; razones.append(f"ADX fuerte ({adx:.0f})")
        if dirección == 'CALL' and stoch < 20: confianza += 15; razones.append("Estocástico sobreventa")
        elif dirección == 'PUT' and stoch > 80: confianza += 15; razones.append("Estocástico sobrecompra")

        if razones and confianza >= 15:
            return {'señal': dirección, 'precio': precio, 'rsi': rsi, 'confianza': confianza, 'razon': " | ".join(razones) + " (Clásico)", 'indicadores': indicadores}

        return {'señal': dirección, 'precio': precio, 'rsi': rsi, 'confianza': 10, 'razon': f"Señal básica (RSI {rsi:.0f})", 'indicadores': indicadores}
    except:
        return señal_por_defecto(codigo_par)

def obtener_precio_actual(codigo_par):
    try:
        datos = yf.download(codigo_par, period="1d", interval="1m", progress=False)
        if isinstance(datos.columns, pd.MultiIndex):
            datos.columns = datos.columns.get_level_values(0)
        if len(datos) < 1: return None
        return float(datos['Close'].iloc[-1])
    except:
        return None

PARES = {
    'EUR/USD': 'EURUSD=X', 'GBP/USD': 'GBPUSD=X', 'USD/JPY': 'USDJPY=X',
    'AUD/USD': 'AUDUSD=X', 'EUR/JPY': 'EURJPY=X', 'AUD/CAD': 'AUDCAD=X',
    'EUR/GBP': 'EURGBP=X', 'GBP/JPY': 'GBPJPY=X', 'NZD/USD': 'NZDUSD=X',
    'USD/CAD': 'USDCAD=X', 'USD/CHF': 'USDCHF=X', 'AUD/JPY': 'AUDJPY=X',
    'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD'
}

# Inicialización
if 'estado' not in st.session_state: st.session_state.estado = 'analizando'
if 'par_sel' not in st.session_state: st.session_state.par_sel = 'EUR/USD'
if 'senal' not in st.session_state: st.session_state.senal = None
if 'pesos' not in st.session_state: st.session_state.pesos = cargar_patrones()
if 'inicio_ciclo' not in st.session_state: st.session_state.inicio_ciclo = None
if 'resultado_guardado' not in st.session_state: st.session_state.resultado_guardado = False

CSS = """
<style>
body { background: #000; color: #fff; font-family: Arial, sans-serif; }
.main-header { background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 20px; border-radius: 10px; border: 2px solid #ffd700; text-align: center; margin-bottom: 20px; }
.main-header h1 { color: #ffd700; margin: 0; font-size: 32px; letter-spacing: 3px; }
.main-header p { color: #aaa; margin: 8px 0 0 0; font-size: 13px; }
.signal-box { padding: 35px; border-radius: 15px; text-align: center; font-size: 36px; font-weight: bold; margin: 20px 0; letter-spacing: 2px; }
.signal-call { background: linear-gradient(135deg, #00cc00, #00ff00); border: 4px solid #00ff00; color: white; box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.signal-put { background: linear-gradient(135deg, #cc0000, #ff0000); border: 4px solid #ff0000; color: white; box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.signal-wait { background: #1a1a2e; border: 3px solid #ffd700; color: #ffd700; }
.signal-neutral { background: #1a1a2e; border: 3px solid #00ff88; color: #00ff88; }
.radar { width: 240px; height: 240px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 25px auto; }
.radar-green { border: 4px solid #00ff00; background: radial-gradient(circle, #0a2a1a, #000); box-shadow: 0 0 40px rgba(0, 255, 0, 0.6); }
.radar-red { border: 4px solid #ff0000; background: radial-gradient(circle, #2a0a0a, #000); box-shadow: 0 0 40px rgba(255, 0, 0, 0.6); }
.radar-yellow { border: 4px solid #ffd700; background: radial-gradient(circle, #2a2a1a, #000); box-shadow: 0 0 40px rgba(255, 215, 0, 0.6); }
.radar-number { font-size: 90px; font-weight: bold; text-shadow: 0 0 25px currentColor; }
.radar-number-green { color: #00ff00; }
.radar-number-red { color: #ff0000; }
.radar-number-yellow { color: #ffd700; font-size: 24px; text-align: center; }
.info-panel { background: #1a1a2e; border: 2px solid #00ff88; border-radius: 10px; padding: 15px; margin: 12px 0; }
.info-panel strong { color: #00ff88; }
.stat-box { background: #1a1a2e; border: 2px solid #ffd700; border-radius: 10px; padding: 12px; text-align: center; }
.footer { text-align: center; color: #666; font-size: 11px; margin-top: 25px; padding: 15px; border-top: 1px solid #333; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

col_logout, col_reset = st.columns([3, 1])
with col_logout:
    if st.button(" Cerrar Sesión", key="logout_btn"):
        st.session_state.logueado = False
        st.session_state.usuario_actual = ""
        st.rerun()
with col_reset:
    if st.button(" Reset", key="reset_btn"):
        if os.path.exists(ARCHIVO_HISTORIAL): os.remove(ARCHIVO_HISTORIAL)
        if os.path.exists(ARCHIVO_PATRONES): os.remove(ARCHIVO_PATRONES)
        st.session_state.pesos = cargar_patrones()
        st.session_state.estado = 'analizando'
        st.session_state.senal = None
        st.session_state.inicio_ciclo = None
        st.session_state.resultado_guardado = False
        st.success("Contadores reseteados")
        time.sleep(1)
        st.rerun()

st.markdown(f'<div class="main-header"><h1>ZT TRADERS</h1><p>Bienvenido: {st.session_state.usuario_actual}</p></div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    par_nombre = st.selectbox("Par:", list(PARES.keys()), index=list(PARES.keys()).index(st.session_state.par_sel))
    st.session_state.par_sel = par_nombre
with col2:
    ahora = datetime.now()
    hora_actual = ahora.strftime('%H:%M:%S')
    st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">HORA</strong><br><span style="color:#00ff88; font-size:22px;">{hora_actual}</span></div>', unsafe_allow_html=True)

codigo = PARES[par_nombre]

hist = cargar_historial()
modelo_info = entrenar_modelo_ml()
ops_con_indicadores = sum(1 for op in hist if 'indicadores' in op)

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Winrate</strong><br><span style="color:#00ff88; font-size:22px;">{round(st.session_state.pesos["winrate"], 1)}%</span></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Operaciones</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ops"]}</span></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">Ganadas</strong><br><span style="color:#00ff88; font-size:22px;">{st.session_state.pesos["total_ganadas"]}</span></div>', unsafe_allow_html=True)
with c4: 
    if modelo_info and ops_con_indicadores >= 20:
        precision = modelo_info["accuracy"] * 100
        color_precision = "#00ff88" if precision >= 60 else "#ffaa00" if precision >= 50 else "#ff4444"
        st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">ML Precisión</strong><br><span style="color:{color_precision}; font-size:22px;">{precision:.0f}%</span><br><small style="color:#aaa;">({ops_con_indicadores} muestras)</small></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="stat-box"><strong style="color:#ffd700;">ML Status</strong><br><span style="color:#ff8800; font-size:18px;">Entrenando...<br>({ops_con_indicadores}/20)</span></div>', unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# LÓGICA SIMPLE Y DIRECTA - SIN SINCRONIZACIÓN VISIBLE
# ==========================================
# Al abrir la app: empieza inmediatamente en "ANALIZANDO"
# Ciclo: 55s análisis → 5s señal → 60s operación → repite
# Cronómetro SOLO en señal (5s) y operación (60s)
# ==========================================

ahora = datetime.now()

# Si no hay inicio de ciclo, crearlo ahora mismo
if st.session_state.inicio_ciclo is None:
    st.session_state.inicio_ciclo = ahora

# Calcular tiempo transcurrido desde el inicio del ciclo
tiempo_transcurrido = (ahora - st.session_state.inicio_ciclo).total_seconds()

# FASE 1: ANÁLISIS (0-54 segundos) - SIN CRONÓMETRO
if tiempo_transcurrido < 55:
    if st.session_state.estado != 'analizando':
        st.session_state.estado = 'analizando'
        st.session_state.senal = None
        st.session_state.resultado_guardado = False
    
    if st.session_state.senal is None:
        st.session_state.senal = analizar_par(codigo)
    
    st.markdown('<div class="signal-box signal-wait">ANALIZANDO...</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-panel" style="text-align:center;"><strong>La IA está analizando el mercado...</strong></div>', unsafe_allow_html=True)
    time.sleep(1)
    st.rerun()

# FASE 2: SEÑAL (55-59 segundos) - CON CRONÓMETRO
elif tiempo_transcurrido < 60:
    if st.session_state.estado != 'senal':
        st.session_state.estado = 'senal'
    
    señal = st.session_state.senal
    es_call = señal['señal'] == 'CALL'
    segundos_restantes = 59 - int(tiempo_transcurrido)
    
    tipo = "CALL" if es_call else "PUT"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-panel"><strong>Punto:</strong> {round(señal["precio"], 5)} | <strong>Confianza:</strong> {señal["confianza"]:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes}</div></div>', unsafe_allow_html=True)
    time.sleep(0.3)
    st.rerun()

# FASE 3: OPERACIÓN (60-119 segundos) - CON CRONÓMETRO
elif tiempo_transcurrido < 120:
    if st.session_state.estado != 'operando':
        st.session_state.estado = 'operando'
    
    señal = st.session_state.senal
    es_call = señal['señal'] == 'CALL'
    segundos_restantes_op = 119 - int(tiempo_transcurrido)
    
    tipo = "CALL" if es_call else "PUT"
    clase_box = "signal-call" if es_call else "signal-put"
    clase_radar = "radar-green" if es_call else "radar-red"
    clase_numero = "radar-number-green" if es_call else "radar-number-red"
    
    st.markdown(f'<div class="signal-box {clase_box}">{tipo}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-panel"><strong>Entrada:</strong> {round(señal["precio"], 5)} | <strong>Confianza:</strong> {señal["confianza"]:.1f}%</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="radar {clase_radar}"><div class="radar-number {clase_numero}">{segundos_restantes_op}</div></div>', unsafe_allow_html=True)
    st.warning("No cierres esta pagina")
    
    # Evaluar resultado al final
    if tiempo_transcurrido >= 119 and not st.session_state.resultado_guardado:
        precio_entrada = señal['precio']
        precio_actual = obtener_precio_actual(codigo)
        resultado_interno = 'PERDIDA'
        if precio_actual is not None and precio_entrada > 0:
            if es_call and precio_actual > precio_entrada:
                resultado_interno = 'GANADA'
            elif not es_call and precio_actual < precio_entrada:
                resultado_interno = 'GANADA'
        indicadores = señal.get('indicadores', {})
        aprender_en_segundo_plano(señal['señal'], resultado_interno, indicadores)
        st.session_state.pesos = cargar_patrones()
        st.session_state.resultado_guardado = True
    
    time.sleep(0.5)
    st.rerun()

# FASE 4: REINICIAR CICLO (después de 120 segundos)
else:
    st.session_state.inicio_ciclo = ahora
    st.session_state.estado = 'analizando'
    st.session_state.senal = None
    st.session_state.resultado_guardado = False
    st.rerun()

st.markdown("---")
st.markdown("### Historial")
hist = cargar_historial()
if hist:
    for op in reversed(hist[-10:]):
        emoji = "✅" if op['resultado'] == 'GANADA' else "❌"
        st.write(f"{emoji} {op['fecha']} - {op['señal']} - {op['resultado']}")
else:
    st.info("La IA está empezando a entrenar desde cero.")

st.markdown(f'<div class="footer">ZT TRADERS | Usuario: {st.session_state.usuario_actual}</div>', unsafe_allow_html=True)
