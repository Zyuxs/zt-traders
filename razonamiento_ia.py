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

_CONFIG = {"proveedor": "gemini", "api_key": "", "modelo": "", "base_url": ""}

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


def configurar(proveedor=None, api_key=None, modelo=None, base_url=None):
    """Recibe los valores de los Secrets de Streamlit (los pasa la app)."""
    if proveedor:
        _CONFIG["proveedor"] = str(proveedor).strip().lower()
    if api_key is not None:
        _CONFIG["api_key"] = str(api_key).strip()
    if modelo:
        _CONFIG["modelo"] = str(modelo).strip()
    if base_url:
        _CONFIG["base_url"] = str(base_url).strip().rstrip("/")


def disponible():
    return bool(_CONFIG["api_key"])


def _endpoint_y_modelo():
    preset = PRESETS.get(_CONFIG["proveedor"], {})
    base = _CONFIG["base_url"] or preset.get("base_url", "")
    modelo = _CONFIG["modelo"] or preset.get("modelo", "")
    return (base.rstrip("/") + "/chat/completions") if base else "", modelo


def _chat(mensajes, max_tokens=220, timeout=14):
    """Una llamada al LLM; devuelve el texto o None ante cualquier fallo.

    Gemini usa su endpoint NATIVO (generateContent con x-goog-api-key):
    verificado con clave real; su ruta compatible-OpenAI respondia con
    timeouts. Los demas proveedores usan chat/completions estilo OpenAI.

    v3.17.5: cada fallo queda registrado en _ULTIMO_ERROR (ver
    ultimo_error()) y las llamadas LARGAS (Diario, timeout>=30) reintentan
    UNA vez tras 15 s ante 503/429, que es la sobrecarga tipica del nivel
    gratuito de Gemini. Las llamadas por señal (timeout corto) no reintentan:
    no deben frenar el ciclo de 2 minutos.
    """
    if not disponible():
        _ULTIMO_ERROR.update(estado="sin_clave", detalle="")
        return None
    sistema = ""
    usuario = ""
    for m in mensajes:
        if m.get("role") == "system":
            sistema = m.get("content", "")
        elif m.get("role") == "user":
            usuario = m.get("content", "")
    intentos = 2 if timeout >= 30 else 1
    if _CONFIG["proveedor"] == "gemini" and not _CONFIG["base_url"]:
        modelo = _CONFIG["modelo"] or PRESETS["gemini"]["modelo"]
        for intento in range(intentos):
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
                    headers={"x-goog-api-key": _CONFIG["api_key"],
                             "Content-Type": "application/json"},
                    json={"system_instruction": {"parts": [{"text": sistema}]},
                          "contents": [{"role": "user", "parts": [{"text": usuario}]}],
                          "generationConfig": {"temperature": 0.2,
                                               "maxOutputTokens": max_tokens}},
                    timeout=timeout,
                )
            except requests.exceptions.Timeout:
                _ULTIMO_ERROR.update(estado="timeout", detalle="")
                return None
            except Exception as exc:
                _ULTIMO_ERROR.update(estado="error", detalle=type(exc).__name__)
                return None
            if r.status_code != 200:
                _ULTIMO_ERROR.update(estado=r.status_code, detalle=str(r.text)[:120])
                if r.status_code in (503, 429) and intento + 1 < intentos:
                    time.sleep(15)
                    continue
                return None
            try:
                partes = r.json()["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError, TypeError, ValueError):
                _ULTIMO_ERROR.update(estado="sin_respuesta", detalle="")
                return None
            _ULTIMO_ERROR.update(estado=None, detalle="")
            return "".join(p.get("text", "") for p in partes)
        return None
    url, modelo = _endpoint_y_modelo()
    if not url or not modelo:
        _ULTIMO_ERROR.update(estado="sin_config", detalle="")
        return None
    for intento in range(intentos):
        try:
            r = requests.post(
                url,
                headers={"Authorization": f"Bearer {_CONFIG['api_key']}",
                         "Content-Type": "application/json"},
                json={"model": modelo, "messages": mensajes,
                      "temperature": 0.2, "max_tokens": max_tokens},
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            _ULTIMO_ERROR.update(estado="timeout", detalle="")
            return None
        except Exception as exc:
            _ULTIMO_ERROR.update(estado="error", detalle=type(exc).__name__)
            return None
        if r.status_code != 200:
            _ULTIMO_ERROR.update(estado=r.status_code, detalle=str(r.text)[:120])
            if r.status_code in (503, 429) and intento + 1 < intentos:
                time.sleep(15)
                continue
            return None
        try:
            texto = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            _ULTIMO_ERROR.update(estado="sin_respuesta", detalle="")
            return None
        _ULTIMO_ERROR.update(estado=None, detalle="")
        return texto
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
