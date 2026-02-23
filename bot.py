import os
import json
import io
import re
import time
import base64
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google.oauth2 import service_account
from googleapiclient.discovery import build

from openai import OpenAI


# =========================
# LOGGING
# =========================
logger = logging.getLogger("facturas-bot")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)


# =========================
# ENV VARS
# =========================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")

SHEET_ID = os.environ.get("SHEET_ID")  # puede ser ID o URL (lo limpiamos)
SHEET_NAME = os.environ.get("SHEET_NAME", "Facturas")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_SECONDS = float(os.environ.get("RETRY_BASE_SECONDS", "1.5"))

# Para limitar tamaño de imagen enviada a OpenAI (Telegram manda fotos grandes a veces)
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(4 * 1024 * 1024)))  # 4MB

if not BOT_TOKEN:
    raise RuntimeError("Falta BOT_TOKEN.")
if not GOOGLE_CREDENTIALS_JSON:
    raise RuntimeError("Falta GOOGLE_CREDENTIALS_JSON.")
if not SHEET_ID:
    raise RuntimeError("Falta SHEET_ID.")
if not OPENAI_API_KEY:
    raise RuntimeError("Falta OPENAI_API_KEY.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# GOOGLE SCOPES (SOLO SHEETS)
# =========================
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


def get_sheets_client():
    creds_info = json.loads(GOOGLE_CREDENTIALS_JSON)
    creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return sheets


def clean_sheet_id(sheet_id: str) -> str:
    if not sheet_id:
        return sheet_id
    if "docs.google.com" in sheet_id:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_id)
        if m:
            return m.group(1)
    return sheet_id


# =========================
# PROMPTS (VISION: ChatGPT analiza la imagen)
# =========================
SYSTEM_PROMPT = """
Eres un Auditor Fiscal Senior especializado en facturación y comprobantes fiscales de la República Dominicana (DGII).

Tu misión es analizar IMÁGENES de facturas dominicanas con formatos completamente variables y extraer datos estructurados con máxima precisión.

IMPORTANTE:
Las facturas pueden tener:
- Formato POS térmico
- Formato A4
- Facturas electrónicas
- Tickets de supermercado
- Recibos pequeños
- Variaciones en encabezados
- Campos en distinto orden
- Información repetida
- Texto mal reconocido

Debes adaptarte a cualquier estructura.

REGLAS ABSOLUTAS:
1) Devuelve ÚNICAMENTE JSON válido.
2) No incluyas explicaciones ni texto adicional.
3) No inventes datos.
4) Si no hay evidencia clara, devuelve "".
5) Prioriza exactitud sobre completitud.
6) Si existe duda significativa, marca Estado = "REVISAR".

FORMATO OBLIGATORIO (EXACTAMENTE 17 CAMPOS):

{
  "Timestamp_UTC": "",
  "Telegram_User": "",
  "Proveedor": "",
  "RNC": "",
  "NCF": "",
  "Fecha_Factura": "",
  "Subtotal_Gravado": "",
  "ITBIS": "",
  "Total": "",
  "Moneda": "",
  "Metodo_Pago": "",
  "Categoria_Gasto": "",
  "Proyecto_CentroCosto": "",
  "Drive_File_Link": "",
  "OCR_Text": "",
  "Estado": "",
  "Observaciones": ""
}

==========================
RECONOCIMIENTO DEL PROVEEDOR (CRÍTICO)
==========================
El nombre del proveedor puede estar en:
- Primera línea del documento
- Encabezado centrado
- En mayúsculas grandes
- Junto al RNC
- Antes de la dirección
- En el pie de página
- Repetido varias veces
- Dentro de un logo (texto parcial)

REGLAS:
1) Prioriza el nombre cerca del RNC del emisor, encabezado y en mayúsculas. Valora SRL, S.A., S.A.S., EIRL.
2) NO uses: nombre del cliente, “Consumidor Final”, sección “Cliente”, banco.
3) Si aparecen múltiples empresas: el proveedor es quien EMITE la factura (RNC asociado al comprobante).
4) Si no es claro: Proveedor="" y explica breve en Observaciones.

==========================
VALIDACIONES FISCALES RD
==========================
RNC:
- 9 dígitos normalmente. Si es cédula, no usar como RNC empresarial.
- Si hay múltiples, usar el del emisor.

NCF:
- Puede iniciar con B, E u otra letra válida.
- Debe estar asociado a “NCF” o “Comprobante Fiscal”.
- Si está incompleto → "".

Fecha_Factura:
- Fecha de emisión, formato YYYY-MM-DD.
- Si hay varias fechas, no usar vencimiento.

Montos:
- Remover símbolos (RD$, $, €).
- Formato 1234.56.
- Manejar 1,234.56 y 1.234,56.
- No asumir separador decimal si ambiguo.

ITBIS:
- Puede inferirse SOLO si Total y Subtotal_Gravado existen y la diferencia es consistente.
- Si hay Propina legal 10%, NO mezclar con ITBIS; anotarlo.

Total:
- Priorizar “TOTAL”, “Total a pagar”, “Monto Total”, “Total General”.
- Si hay múltiples totales, elige el final a pagar y explica.

Moneda:
- RD$, DOP → DOP
- USD, US$, Dólares → USD
- € → EUR
- Si solo “$” sin contexto → ""

Método_Pago:
- Detecta: Efectivo, Tarjeta, Crédito, Débito, Transferencia, Cheque.
- Si no está claro → ""

Categoria_Gasto / Proyecto_CentroCosto:
- Solo si está explícito. No inventar.

ESTADO:
- "OK" SOLO si Total no vacío y (RNC o NCF no vacío) y sin inconsistencias fuertes.
- En cualquier otro caso: "REVISAR".

CHEQUEO FINAL:
- JSON con EXACTAMENTE 17 campos.
- No agregues campos.
- Timestamp_UTC, Telegram_User, Drive_File_Link, OCR_Text deben quedar "".
- Solo JSON válido.
""".strip()

USER_PROMPT = """
Analiza la imagen de esta factura dominicana (formato variable) y extrae los datos estructurados conforme a las reglas del sistema.
Devuelve SOLO el JSON.
""".strip()

EXPECTED_KEYS = [
    "Timestamp_UTC",
    "Telegram_User",
    "Proveedor",
    "RNC",
    "NCF",
    "Fecha_Factura",
    "Subtotal_Gravado",
    "ITBIS",
    "Total",
    "Moneda",
    "Metodo_Pago",
    "Categoria_Gasto",
    "Proyecto_CentroCosto",
    "Drive_File_Link",
    "OCR_Text",
    "Estado",
    "Observaciones",
]


def _normalize_ai_output(data: Dict[str, Any]) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for k in EXPECTED_KEYS:
        v = data.get(k, "")
        if v is None:
            v = ""
        cleaned[k] = str(v).strip()
    return cleaned


def compute_estado_fallback(ai: Dict[str, str]) -> str:
    total = (ai.get("Total") or "").strip()
    rnc = (ai.get("RNC") or "").strip()
    ncf = (ai.get("NCF") or "").strip()
    if total and (rnc or ncf):
        return "OK"
    return "REVISAR"


def _sleep_backoff(attempt: int):
    # 1.5s, 3s, 6s...
    time.sleep(RETRY_BASE_SECONDS * (2 ** attempt))


def parse_invoice_from_image_with_gpt(image_bytes: bytes) -> Dict[str, str]:
    """
    Usa ChatGPT con visión: analiza la imagen directamente (sin Google Vision).
    Devuelve dict con las 17 llaves.
    """
    # Telegram a veces manda imágenes grandes; cortamos si excede MAX_IMAGE_BYTES
    if len(image_bytes) > MAX_IMAGE_BYTES:
        image_bytes = image_bytes[:MAX_IMAGE_BYTES]

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    last_err: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            # Compatible con openai==1.63.0: Chat Completions con contenido multimodal
            completion = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": USER_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )

            out = (completion.choices[0].message.content or "").strip()
            data = json.loads(out)
            return _normalize_ai_output(data)

        except Exception as e:
            last_err = e
            logger.warning(f"OpenAI intento {attempt+1}/{MAX_RETRIES} falló: {e}")
            if attempt < MAX_RETRIES - 1:
                _sleep_backoff(attempt)

    # Si todo falla, propagamos el error
    raise RuntimeError(f"Falló OpenAI tras {MAX_RETRIES} intentos: {last_err}")


# =========================
# SHEETS
# =========================
def append_to_sheet(sheets, sheet_id: str, row_values):
    range_name = f"{SHEET_NAME}!A1"
    body = {"values": [row_values]}
    sheets.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()


# =========================
# TELEGRAM HANDLERS
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "✅ Bot activo.\n"
        "Envíame una FOTO de una factura.\n"
        "Flujo: ChatGPT (visión) → Google Sheets."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    sheets = get_sheets_client()

    user = update.effective_user.username or str(update.effective_user.id)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)

    buf = io.BytesIO()
    await tg_file.download_to_memory(out=buf)
    img_bytes = buf.getvalue()

    await update.message.reply_text("🧠 Analizando factura con ChatGPT (visión)…")

    ai = parse_invoice_from_image_with_gpt(img_bytes)

    estado = (ai.get("Estado") or "").strip().upper()
    if estado not in ("OK", "REVISAR"):
        estado = compute_estado_fallback(ai)

    sheet_id_clean = clean_sheet_id(SHEET_ID)

    # 17 columnas exactas según tu sheet
    # Nota: aunque el prompt dice que algunos deben venir "", la FILA sí los llena.
    row = [
        timestamp,                                 # A Timestamp_UTC
        user,                                      # B Telegram_User
        ai.get("Proveedor", "") or "",             # C Proveedor
        ai.get("RNC", "") or "",                   # D RNC
        ai.get("NCF", "") or "",                   # E NCF
        ai.get("Fecha_Factura", "") or "",         # F Fecha_Factura
        ai.get("Subtotal_Gravado", "") or "",      # G Subtotal_Gravado
        ai.get("ITBIS", "") or "",                 # H ITBIS
        ai.get("Total", "") or "",                 # I Total
        ai.get("Moneda", "") or "",                # J Moneda
        ai.get("Metodo_Pago", "") or "",           # K Metodo_Pago
        ai.get("Categoria_Gasto", "") or "",       # L Categoria_Gasto
        ai.get("Proyecto_CentroCosto", "") or "",  # M Proyecto_CentroCosto
        "",                                        # N Drive_File_Link (no usado)
        "",                                        # O OCR_Text (no usado)
        estado,                                    # P Estado
        ai.get("Observaciones", "") or "",         # Q Observaciones
    ]

    append_to_sheet(sheets, sheet_id_clean, row)

    await update.message.reply_text(
        "✅ Registrado en Google Sheets.\n\n"
        f"Proveedor: {ai.get('Proveedor') or '—'}\n"
        f"RNC: {ai.get('RNC') or '—'}\n"
        f"NCF: {ai.get('NCF') or '—'}\n"
        f"Fecha: {ai.get('Fecha_Factura') or '—'}\n"
        f"Subtotal: {ai.get('Subtotal_Gravado') or '—'}\n"
        f"ITBIS: {ai.get('ITBIS') or '—'}\n"
        f"Total: {ai.get('Total') or '—'}\n"
        f"Moneda: {ai.get('Moneda') or '—'}\n"
        f"Método pago: {ai.get('Metodo_Pago') or '—'}\n"
        f"Estado: {estado}\n"
        f"Obs: {ai.get('Observaciones') or '—'}"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.exception("❌ Error no manejado:", exc_info=context.error)
    try:
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "⚠️ Ocurrió un error procesando la factura. "
                "Reintenta enviando la foto."
            )
    except Exception:
        pass


def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(on_error)

    logger.info("Iniciando bot con polling (sin webhook)...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
