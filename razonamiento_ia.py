# -*- coding: utf-8 -*-
"""
razonamiento_ia.py — Capa de razonamiento con un modelo de lenguaje (LLM)
para ZT TRADERS INSTITUTIONAL PRO (v3.4).

Papel de esta capa (acordado con Jose):
  - El cerebro cuantitativo (nucleo_ia.py) decide la direccion (CALL/PUT)
    y su probabilidad. Esta capa NO sustituye eso y NUNCA veta una señal.
  - El LLM actua como un trader senior que revisa la señal: confirma,
    sube o rebaja su NIVEL, y explica el por que en una linea.
  - Tambien resume el "clima" de la sesion y analiza el historial del
    usuario (diario inteligente) cuando el Admin lo pide.

v3.9: los prompts incorporan frameworks destilados del proyecto abierto
"claude-trading-skills" de tradermonty (licencia MIT,
github.com/tradermonty/claude-trading-skills): el checklist sistematico
del analista tecnico (tendencia, niveles, alineacion, volumen/velas,
conflictos), la disciplina pre-operacion (postura de exposicion y rachas)
y la autopsia de señales (atribuir cada perdida a una causa probable,
con muestra minima antes de concluir).

Proveedores GRATUITOS para la fase de prueba (clave propia del usuario):
  - gemini     -> Google AI Studio (aistudio.google.com), nivel gratuito.
  - groq       -> console.groq.com, nivel gratuito, muy rapido.
  - openrouter -> openrouter.ai, modelos con sufijo :free.
Todos hablan el mismo formato (chat/completions estilo OpenAI), por eso
un solo cliente sirve para los tres, y el dia que la app sea comercial
se cambia de proveedor/modelo SOLO desde los Secrets, sin tocar codigo:
  ia_proveedor = "gemini" | "groq" | "openrouter" | "personalizado"
  ia_api_key   = "..."           (obligatoria)
  ia_modelo    = "..."           (opcional: usa el del preset)
  ia_base_url  = "https://..."   (solo para "personalizado")

Respaldo automatico (v3.17.6): si el proveedor principal devuelve 429
(limite) o 503 (sobrecarga) o falla, la capa reintenta con un SEGUNDO
proveedor sin que la app se entere. Cada proveedor tiene su propio cupo
gratuito, asi que el limite de uno no tumba la capa. Para Qwen por Groq
(gratis, console.groq.com, registro solo con correo):
  ia_proveedor_respaldo = "groq"
  ia_api_key_respaldo   = "gsk_..."      (la clave de Groq)
  ia_modelo_respaldo    = "qwen3-32b"   (opcional: Qwen 3 de 32B en Groq)
Sin respaldo configurado, la capa se comporta como antes.

Si no hay clave, o la API falla, o responde tarde: TODO devuelve
None / "" y la app sigue funcionando exactamente igual sin esta capa.
Nada en este modulo lanza excepciones hacia la app.
"""

import json
import time

import requests

PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "modelo": "gemini-3.8-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "modelo": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "modelo": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

_CONFIG = {"proveedor": "gemini", "api_key": "", "modelo": "", "base_url": "",
           "proveedor_respaldo": "", "api_key_respaldo": "",
           "modelo_respaldo": "", "base_url_respaldo": ""}
_ULTIMO_PROVEEDOR = {"nombre": ""}

# v3.17.5: causa del ultimo fallo, para que la app pueda decirle al
# usuario QUE paso (sobrecarga, limite, clave, modelo...) en vez de un
# "no respondio" generico que no permite hacer nada. Nunca guarda la clave.
_ULTIMO_ERROR = {"estado": None, "detalle": ""}


def ultimo_error():
    """Frase en español con la causa del último fallo del LLM ("" si no hubo)."""
    e = _ULTIMO_ERROR["estado"]
    if e is None:
        return ""
    if e == 503:
        return ("Gemini está sobrecargado en este momento (nivel gratuito). "
                "No es tu clave: inténtalo de nuevo en unos minutos.")
    if e == 429:
        return ("Se alcanzó el límite de uso del nivel gratuito. "
                "Espera unos minutos e inténtalo de nuevo.")
    if e in (401, 403):
        return ("La clave de API no fue aceptada. Revisa ia_api_key en Secrets: "
                "debe ser la clave vigente, sin espacios ni caracteres de más.")
    if e == 404:
        return ("El modelo de IA no está disponible en este momento. "
                "Avísale a Muse para actualizarlo.")
    if e == "timeout":
        return "La IA tardó demasiado en responder. Inténtalo de nuevo."
    if e == "sin_respuesta":
        return "La IA respondió pero sin texto aprovechable. Inténtalo de nuevo."
    if e in ("sin_clave", "sin_config"):
        return "No hay clave de IA configurada. Revisa ia_api_key en Secrets."
    if e == "error":
        return "No se pudo conectar con la IA (red). Inténtalo de nuevo."
    return f"La IA no respondió (error {e}). Inténtalo de nuevo."

SISTEMA = (
    "Eres un trader profesional senior especializado en opciones binarias de "
    "1 minuto. Revisas las señales de un sistema cuantitativo. Reglas fijas: "
    "(1) NUNCA propones cambiar la direccion de la señal ni vetarla; solo "
    "ajustas su nivel de confianza (Alta/Media/Baja). "
    "(2) Piensas como los mejores traders: tendencia manda, no se compra "
    "pegado a resistencia ni se vende pegado a soporte, se desconfia de "
    "entradas tardias tras velas extendidas y de operar en rango sin "
    "confirmacion. "
    "(3) Trabajas con el metodo de un analista tecnico sistematico: "
    "tendencia y su fuerza, soportes/resistencias y espacio libre, "
    "alineacion de medias, impulso y volumen, patrones de velas; y declaras "
    "los conflictos cuando las piezas no concuerdan. "
    "(4) Disciplina de riesgo: tras rachas perdedoras o con postura de "
    "sesion REDUCIR, tu vara es mas exigente, no mas blanda. "
    "(5) Honestidad estadistica: con muestras chicas (menos de 20 datos) "
    "hablas de indicios, no de certezas. "
    "(6) Respondes breve y directo, en español."
)


def configurar(proveedor=None, api_key=None, modelo=None, base_url=None,
               proveedor_respaldo=None, api_key_respaldo=None,
               modelo_respaldo=None, base_url_respaldo=None):
    """Recibe los valores de los Secrets de Streamlit (los pasa la app)."""
    if proveedor:
        _CONFIG["proveedor"] = str(proveedor).strip().lower()
    if api_key is not None:
        _CONFIG["api_key"] = str(api_key).strip()
    if modelo:
        _CONFIG["modelo"] = str(modelo).strip()
    if base_url:
        _CONFIG["base_url"] = str(base_url).strip().rstrip("/")
    if proveedor_respaldo:
        _CONFIG["proveedor_respaldo"] = str(proveedor_respaldo).strip().lower()
    if api_key_respaldo is not None:
        _CONFIG["api_key_respaldo"] = str(api_key_respaldo).strip()
    if modelo_respaldo:
        _CONFIG["modelo_respaldo"] = str(modelo_respaldo).strip()
    if base_url_respaldo:
        _CONFIG["base_url_respaldo"] = str(base_url_respaldo).strip().rstrip("/")


def _perfiles():
    """Perfiles utilizables en orden: principal y, si hay clave, respaldo."""
    lista = []
    if _CONFIG["api_key"]:
        lista.append({"proveedor": _CONFIG["proveedor"],
                      "api_key": _CONFIG["api_key"],
                      "modelo": _CONFIG["modelo"],
                      "base_url": _CONFIG["base_url"]})
    if _CONFIG["api_key_respaldo"]:
        lista.append({"proveedor": _CONFIG["proveedor_respaldo"] or "groq",
                      "api_key": _CONFIG["api_key_respaldo"],
                      "modelo": _CONFIG["modelo_respaldo"],
                      "base_url": _CONFIG["base_url_respaldo"]})
    return lista


def disponible():
    return bool(_perfiles())


def respaldo_configurado():
    """True si hay un segundo proveedor con clave listo para relevar."""
    return len(_perfiles()) > 1


def ultimo_proveedor():
    """Proveedor que respondio la ultima llamada exitosa ("" si ninguno)."""
    return _ULTIMO_PROVEEDOR["nombre"]


def _intentar(perfil, sistema, usuario, mensajes, max_tokens, timeout):
    """Un intento contra UN perfil. Devuelve (texto|None, estado).

    Gemini usa su endpoint NATIVO (generateContent con x-goog-api-key):
    verificado con clave real; su ruta compatible-OpenAI respondia con
    timeouts. Los demas proveedores usan chat/completions estilo OpenAI.
    """
    proveedor = perfil["proveedor"]
    preset = PRESETS.get(proveedor, {})
    base = (perfil["base_url"] or preset.get("base_url", "")).rstrip("/")
    modelo = perfil["modelo"] or preset.get("modelo", "")
    if proveedor == "gemini" and not perfil["base_url"]:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
                headers={"x-goog-api-key": perfil["api_key"],
                         "Content-Type": "application/json"},
                json={"system_instruction": {"parts": [{"text": sistema}]},
                      "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                      "generationConfig": {"temperature": 0.2,
                                           "maxOutputTokens": max_tokens}},
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            return None, "timeout"
        except Exception:
            return None, "error"
        if r.status_code != 200:
            return None, r.status_code
        try:
            partes = r.json()["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError, ValueError):
            return None, "sin_respuesta"
        return "".join(p.get("text", "") for p in partes), None
    url = (base + "/chat/completions") if base else ""
    if not url or not modelo:
        return None, "sin_config"
    try:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {perfil['api_key']}",
                     "Content-Type": "application/json"},
            json={"model": modelo, "messages": mensajes,
                  "temperature": 0.2, "max_tokens": max_tokens},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        return None, "timeout"
    except Exception:
        return None, "error"
    if r.status_code != 200:
        return None, r.status_code
    try:
        texto = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None, "sin_respuesta"
    return texto, None


def _chat(mensajes, max_tokens=220, timeout=14):
    """Una llamada al LLM; devuelve el texto o None ante cualquier fallo.

    v3.17.5: cada fallo queda registrado en _ULTIMO_ERROR (ver
    ultimo_error()).
    v3.17.6: relevo automatico — si un perfil falla (429, 503, clave,
    red...), se intenta con el siguiente perfil SIN esperar: cada proveedor
    tiene su propio cupo gratuito. Solo el ULTIMO perfil, y solo en llamadas
    LARGAS (Diario, timeout>=30), reintenta una vez tras 15 s ante 503/429.
    Las llamadas por señal (timeout corto) no duermen: no deben frenar el
    ciclo de 2 minutos.
    """
    perfiles = _perfiles()
    if not perfiles:
        _ULTIMO_ERROR.update(estado="sin_clave", detalle="")
        return None
    sistema = ""
    usuario = ""
    for m in mensajes:
        if m.get("role") == "system":
            sistema = m.get("content", "")
        elif m.get("role") == "user":
            usuario = m.get("content", "")
    for idx, perfil in enumerate(perfiles):
        es_ultimo = idx == len(perfiles) - 1
        intentos = 2 if (timeout >= 30 and es_ultimo) else 1
        for intento in range(intentos):
            texto, estado = _intentar(perfil, sistema, usuario, mensajes,
                                      max_tokens, timeout)
            if texto is not None:
                _ULTIMO_ERROR.update(estado=None, detalle="")
                _ULTIMO_PROVEEDOR["nombre"] = perfil["proveedor"]
                return texto
            _ULTIMO_ERROR.update(estado=estado, detalle="")
            if estado in (503, 429) and intento + 1 < intentos:
                time.sleep(15)
                continue
            break
    return None


def _extraer_json(texto):
    if not texto:
        return None
    ini = texto.find("{")
    fin = texto.rfind("}")
    if ini == -1 or fin == -1 or fin <= ini:
        return None
    try:
        return json.loads(texto[ini:fin + 1])
    except Exception:
        return None


def segunda_opinion(resumen):
    """Revisa UNA señal. `resumen`: dict con par, tipo, direccion, confianza,
    nivel, razon, regimen y (si hay) probabilidades. Devuelve
    {"veredicto": CONFIRMA|SUBE|REBAJA, "motivo": "..."} o None."""
    prompt = (
        "Revisa esta señal del sistema cuantitativo con este CHECKLIST de "
        "analista tecnico (en silencio, punto por punto): "
        "1) Tendencia: direccion y fuerza; ¿la señal va con ella o contra ella? "
        "2) Niveles: ¿hay espacio libre hasta el soporte/resistencia o entra "
        "pegado a un nivel? 3) Alineacion: medias e impulso (MACD/ROC) ¿apoyan? "
        "4) Volumen y velas: ¿hay confirmacion o es una vela extendida y tardia? "
        "5) Conflictos: ¿que pieza NO concuerda? 6) Disciplina: postura de la "
        "sesion y racha reciente ¿permiten ser generoso con el nivel?\n"
        "Responde SOLO con un JSON asi: {\"veredicto\": "
        "\"CONFIRMA\"|\"SUBE\"|\"REBAJA\", \"motivo\": \"una linea, maximo 25 "
        "palabras\"}\n"
        "CONFIRMA = el nivel esta bien. SUBE = merece un nivel mas. "
        "REBAJA = merece un nivel menos (contra tendencia, pegado a nivel, "
        "entrada tardia, conflicto entre piezas, postura REDUCIR...). "
        "Nunca cambies la direccion.\n\n"
        f"Par: {resumen.get('par')} ({resumen.get('tipo')})\n"
        f"Señal: {resumen.get('direccion')} | Confianza del sistema: "
        f"{resumen.get('confianza')}% | Nivel actual: {resumen.get('nivel')}\n"
        f"Régimen de mercado: {resumen.get('regimen')}\n"
        f"Postura de la sesión (por resultados recientes): {resumen.get('postura') or 'sin datos'}\n"
        f"Análisis del sistema: {resumen.get('razon')}\n"
        f"Advertencias: {', '.join(resumen.get('advertencias') or []) or 'ninguna'}"
    )
    datos = _extraer_json(_chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=120, timeout=12))
    if not datos:
        return None
    veredicto = str(datos.get("veredicto", "")).upper().strip()
    if veredicto not in ("CONFIRMA", "SUBE", "REBAJA"):
        return None
    return {"veredicto": veredicto,
            "motivo": str(datos.get("motivo", "")).strip()[:200]}


def clima_sesion(par, tipo, variacion_60m_pct, regimen):
    """Frase corta con el clima del par para la sesion (o "" si no hay IA)."""
    prompt = (
        f"En UNA frase breve (maximo 22 palabras), describe el clima actual "
        f"para operar {par} ({tipo}) en operaciones de 1 minuto, como un "
        f"entrenador de exposicion: si el contexto invita a operar normal, "
        f"a ser selectivo o a reducir. "
        f"Variación de la última hora: {variacion_60m_pct}%. "
        f"Contexto (régimen y postura de la sesión): {regimen}. "
        f"Sin prometer resultados."
    )
    texto = _chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=80, timeout=12)
    return (texto or "").strip().strip('"')[:220]


def analizar_historial(resumen_texto):
    """Diario inteligente: informe en Markdown a partir de un resumen del
    historial y las metricas del usuario. Devuelve "" si no hay IA."""
    prompt = (
        "Analiza el historial de operaciones de este usuario como su mentor "
        "de trading, con método de autopsia de señales (postmortem): cada "
        "pérdida se atribuye a una causa probable — contra la tendencia del "
        "régimen, pegada a soporte/resistencia, en rango sin confirmación, "
        "en volatilidad extrema, o sin causa clara — y cada patrón se juzga "
        "solo si tiene muestra suficiente (20+ casos; con menos, es indicio). "
        "Escribe un informe breve en español con estas secciones: "
        "## Qué está funcionando, ## Qué está fallando (con la causa dominante "
        "de las pérdidas), ## Ajustes concretos. "
        "Básate SOLO en los datos dados; si un dato no alcanza, dilo. "
        "Recuerda el equilibrio: con payout 86% necesita acertar mas del "
        "53,8% para no perder.\n\nDATOS:\n" + resumen_texto
    )
    texto = _chat([
        {"role": "system", "content": SISTEMA},
        {"role": "user", "content": prompt},
    ], max_tokens=700, timeout=45)
    return (texto or "").strip()
