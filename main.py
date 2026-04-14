from __future__ import annotations
import asyncio, ipaddress, logging, re, socket, httpx, os, base64, json
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("juice")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

@asynccontextmanager
async def lifespan(app):
    log.info("Juice ready.")
    if ANTHROPIC_API_KEY:
        log.info("Claude AI features enabled.")
    else:
        log.warning("ANTHROPIC_API_KEY not set — AI features disabled.")
    yield

app = FastAPI(title="Juice — Smart Electricity", version="1.0.0", lifespan=lifespan)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://ocr-service-4e7i.onrender.com,http://localhost:8000,http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

ALLOWED = {"image/jpeg","image/png","image/webp","image/tiff","image/bmp","image/heic","image/heif","application/pdf"}
MAX_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_private_url(url: str) -> bool:
    """Block requests to private/internal network addresses (SSRF protection)."""
    from urllib.parse import urlparse
    hostname = urlparse(str(url)).hostname
    if not hostname:
        return True
    try:
        resolved = socket.getaddrinfo(hostname, None)
        for _, _, _, _, addr in resolved:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except socket.gaierror:
        return True
    return False

def _clean_text(raw):
    lines = raw.splitlines()
    cleaned = [re.sub(r"[^\S\n]+"," ",l).strip() for l in lines]
    return re.sub(r"\n{3,}","\n\n","\n".join(cleaned)).strip()

def _detect_type(data: bytes) -> str:
    if data[:4] == b'%PDF':
        return "application/pdf"
    if data[:4] in (b'\x00\x00\x00\x18', b'\x00\x00\x00\x1c', b'\x00\x00\x00\x20') or b'heic' in data[:20].lower() or b'heix' in data[:20].lower():
        return "image/heic"
    if data[:2] == b'\xff\xd8':
        return "image/jpeg"
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:4] == b'RIFF':
        return "image/webp"
    return "image/jpeg"

async def _ocr_from_bytes(data: bytes) -> str:
    if not ANTHROPIC_API_KEY:
        return "[ANTHROPIC_API_KEY not set]"

    detected_type = _detect_type(data)

    # For images: fix EXIF rotation and resize large images
    if detected_type != "application/pdf":
        try:
            from PIL import Image, ImageOps
            import io as _io
            pil_img = Image.open(_io.BytesIO(data))
            pil_img = ImageOps.exif_transpose(pil_img)
            # Resize if too large (max 2000px on longest side)
            max_dim = 2000
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                log.info(f"Resized image to {new_size}")
            buf = _io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            detected_type = "image/jpeg"
        except Exception as e:
            log.warning(f"Image processing failed: {e}")

    # For HEIC: convert to JPEG first
    if detected_type == "image/heic":
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as f:
                f.write(data); tmp_in = f.name
            tmp_out = tmp_in.replace('.heic', '.jpg')
            subprocess.run(['convert', tmp_in, tmp_out], check=True, timeout=10)
            data = open(tmp_out, 'rb').read()
            detected_type = "image/jpeg"
            for p in [tmp_in, tmp_out]:
                try: os.unlink(p)
                except: pass
        except Exception as e:
            log.warning(f"HEIC conversion failed: {e}")

    b64 = base64.b64encode(data).decode()

    if detected_type == "application/pdf":
        try:
            import fitz  # pymupdf
            doc = fitz.open(stream=data, filetype="pdf")
            # Try local text extraction first (instant, works for digital PDFs)
            texts = [page.get_text() for page in doc]
            combined = "\n\n".join(t.strip() for t in texts if t.strip())
            if len(combined.split()) > 20:
                log.info(f"PDF text extracted locally ({len(combined.split())} words)")
                return _clean_text(combined)
            # Fallback: scanned PDF — render to images and OCR via API
            log.info("PDF has no embedded text, falling back to Vision OCR")
            page_images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csRGB)
                page_images.append(pix.tobytes("jpeg"))
            texts = await asyncio.gather(*[_ocr_from_bytes(img) for img in page_images])
            return _clean_text("\n\n".join(texts))
        except Exception as e:
            log.warning(f"PDF processing failed: {e}")
            return f"[PDF processing failed: {e}]"

    content_block = {"type": "image", "source": {"type": "base64", "media_type": detected_type, "data": b64}}

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={"model": "claude-sonnet-4-6", "max_tokens": 4096,
                      "messages": [{"role": "user", "content": [
                          content_block,
                          {"type": "text", "text": "Extract ALL text from this image with perfect accuracy. This text is likely in Hebrew, Arabic, English, or mixed.\n\nCRITICAL RULES:\n1. Hebrew must be spelled EXACTLY as a native speaker would write it. Fix any OCR artifacts. Use proper Hebrew spelling for known names, places, organizations, teams, etc.\n2. For tabular or grid data: output as a clean markdown table with proper Hebrew column headers.\n3. Preserve all numbers, dates, times, currency amounts, and reference numbers exactly.\n4. Use markdown headings (# ##) for sections.\n5. Read RIGHT-TO-LEFT for Hebrew text — ensure word order is correct.\n6. Output ONLY the extracted content. No explanations or commentary."}
                      ]}]}
            )
        if resp.status_code == 200:
            return _clean_text(resp.json()["content"][0]["text"].strip())
        log.warning(f"Claude Vision {resp.status_code}: {resp.text[:300]}")
        return f"[Error {resp.status_code}]"
    except Exception as e:
        log.warning(f"Claude Vision failed: {type(e).__name__}: {e}")
        import traceback; log.warning(traceback.format_exc())
        return f"[Error: {type(e).__name__}: {e}]"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TextResult(BaseModel):
    filename: str
    text: str
    structured: dict | None = None

class BatchResponse(BaseModel):
    results: list[TextResult]
    total: int

class UrlRequest(BaseModel):
    url: HttpUrl

class AIRequest(BaseModel):
    text: str
    language: str = ""

class AIResponse(BaseModel):
    result: str

OutputMode = Literal["text", "structured"]

def _to_structured(text):
    lines = [l for l in text.splitlines() if l.strip()]
    kv = {}
    for line in lines:
        m = re.match(r"^([^:\n]{1,60}):\s*(.+)$", line)
        if m: kv[m.group(1).strip()] = m.group(2).strip()
    words = text.split()
    return {"lines": lines, "key_value_pairs": kv,
            "stats": {"line_count": len(lines), "word_count": len(words), "char_count": len(text)}}

def _build(filename, text, mode):
    return TextResult(filename=filename, text=text,
                      structured=_to_structured(text) if mode=="structured" else None)

# ---------------------------------------------------------------------------
# OCR routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
async def health():
    return {"status":"ok","model":"claude-vision","app":"Juice",
            "ai": bool(ANTHROPIC_API_KEY), "version":"1.0.0"}

@app.post("/ocr/upload", response_model=TextResult, tags=["ocr"])
async def ocr_single(file: Annotated[UploadFile, File()], mode: OutputMode = Query("text")):
    if file.content_type and file.content_type not in ALLOWED and not (file.filename and file.filename.lower().endswith('.pdf')):
        raise HTTPException(415, f"סוג קובץ לא נתמך: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "הקובץ גדול מדי (מקסימום 20MB).")
    text = await _ocr_from_bytes(data)
    return _build(file.filename or "upload", text, mode)

@app.post("/ocr/batch", response_model=BatchResponse, tags=["ocr"])
async def ocr_batch(files: Annotated[list[UploadFile], File()], mode: OutputMode = Query("text")):
    if not files: raise HTTPException(400, "לא נבחרו קבצים.")
    if len(files) > 20: raise HTTPException(400, "מקסימום 20 קבצים.")
    async def process(f):
        if f.content_type and f.content_type not in ALLOWED and not (f.filename and f.filename.lower().endswith('.pdf')):
            return TextResult(filename=f.filename or "?", text="[דלג] סוג לא נתמך")
        data = await f.read()
        if len(data) > MAX_BYTES:
            return TextResult(filename=f.filename or "?", text="[דלג] קובץ גדול מדי")
        text = await _ocr_from_bytes(data)
        return _build(f.filename or "?", text, mode)
    results = await asyncio.gather(*[process(f) for f in files])
    return BatchResponse(results=list(results), total=len(results))

@app.post("/ocr/url", response_model=TextResult, tags=["ocr"])
async def ocr_url(body: UrlRequest, mode: OutputMode = Query("text")):
    if _is_private_url(str(body.url)):
        raise HTTPException(400, "URLs pointing to private or internal networks are not allowed.")
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(str(body.url)); r.raise_for_status()
    except httpx.HTTPStatusError as e: raise HTTPException(502, f"שגיאת שרת מרוחק: {e.response.status_code}")
    except httpx.RequestError as e: raise HTTPException(502, str(e))
    if len(r.content) > MAX_BYTES: raise HTTPException(413, "הקובץ גדול מדי.")
    first = r.content[:20].lower()
    if b"<html" in first or b"<!doc" in first:
        raise HTTPException(422, "הקישור מוביל לדף אינטרנט ולא לתמונה.")
    text = await _ocr_from_bytes(r.content)
    return _build(str(body.url).split("/")[-1] or "remote", text, mode)

# ---------------------------------------------------------------------------
# AI helper
# ---------------------------------------------------------------------------
async def _claude(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 2048,
                  "messages": [{"role": "user", "content": prompt}]}
        )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"].strip()

# ---------------------------------------------------------------------------
# AI routes
# ---------------------------------------------------------------------------
@app.post("/ai/eli12", response_model=AIResponse, tags=["ai"])
async def explain_eli12(body: AIRequest):
    if not body.text.strip(): raise HTTPException(400, "הטקסט ריק.")
    result = await _claude(
        f"""אתה עוזר שמסביר מסמכים מורכבים בשפה פשוטה וידידותית.
המשימה: הסבר את הטקסט הבא כאילו אתה מדבר עם אדם מבוגר שלא מכיר עניינים בירוקרטיים/משפטיים.
כללים: השתמש בשפה פשוטה וברורה, הסבר מונחים מורכבים, ציין מה חשוב ומה הפעולות הנדרשות, כתוב בעברית, פרק למשפטים קצרים.
טקסט לפישוט:
{body.text}"""
    )
    return AIResponse(result=result)

@app.post("/ai/translate", response_model=AIResponse, tags=["ai"])
async def translate(body: AIRequest):
    if not body.text.strip(): raise HTTPException(400, "הטקסט ריק.")
    lang = body.language or "English"
    result = await _claude(
        f"""תרגם את הטקסט הבא ל{lang}.
כללים: תרגום מדויק ובהיר, שמור על המבנה המקורי, פלט את התרגום בלבד ללא הסברים.
טקסט לתרגום:
{body.text}"""
    )
    return AIResponse(result=result)

@app.post("/ai/analyze", response_model=AIResponse, tags=["ai"])
async def analyze(body: AIRequest):
    if not body.text.strip(): raise HTTPException(400, "Empty text.")
    result = await _claude(
        """Analyze this text and extract fields. The text can be anything: a bill, receipt, bank transfer, legal document, medical document, sports data, menu, schedule, article, etc.
Return a JSON object with null for missing fields:
- document_type: clear, user-friendly description e.g. "Water & Sewage Bill", "Money Transfer", "Sports Betting Odds", "Restaurant Menu", "Medical Referral", "News Article"
- amount: main payment amount as string e.g. "2,478.80"
- currency: currency symbol e.g. "₪"
- due_date: due date as written e.g. "05/04/2026"
- deadline_date: due date ISO format YYYY-MM-DD or null
- deadline_title: e.g. "Payment due"
- clearing_id: clearing/payment reference number
- account_number: account or customer number
- reference_number: reference or case number
- period: billing period e.g. "03-04/2026"
- sender: organization that sent this
- recipient: recipient name
- barcode: full barcode or payment slip number if present
- payment_required: true or false
- reply_address: postal reply address or null
- contact_details: object with phone, fax, email, website keys if found
- property_details: object with address, block, parcel, size, type, description keys if found

Return ONLY valid JSON. No markdown.

Document:
""" + body.text
    )
    clean = result.replace('```json', '').replace('```', '').strip()
    return AIResponse(result=json.dumps(_safe_parse(clean)))

def _safe_parse(text: str) -> dict:
    """Parse JSON, fixing common issues like unescaped quotes in Hebrew (e.g. בע"מ)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fixing unescaped inner quotes: replace " that aren't JSON structural
    import re
    fixed = re.sub(r'(?<=\w)"(?=\w)', '\\"', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # Last resort: extract key-value pairs with regex
    result = {}
    for m in re.finditer(r'"(\w+)":\s*("(?:[^"\\]|\\.)*"|null|true|false|[\d.]+|\{[^}]*\})', text):
        key, val = m.group(1), m.group(2)
        try:
            result[key] = json.loads(val)
        except:
            result[key] = val.strip('"')
    return result if result else {"raw": text}

ANALYZE_PROMPT = """Analyze this Israeli electricity bill image (חשבון חשמל).

Extract these fields. Return a JSON object with null for missing fields:
- document_type: "Electricity Bill"
- amount: total payment amount as string e.g. "1,421.25"
- currency: "₪"
- due_date: payment deadline as written
- deadline_date: same date in ISO format YYYY-MM-DD or null
- period: billing period e.g. "21/01/2026 - 19/03/2026"
- period_days: number of days in billing period as integer
- sender: electricity supplier name
- recipient: customer name
- account_number: contract or account number
- meter_number: electricity meter number
- barcode: payment barcode number if visible
- payment_required: true/false
- tariff_type: "flat" if תעריף אחיד, "taoz" if תע״ז time-of-use, or other
- consumption_kwh: total kWh consumed as number e.g. 2061
- rate_per_kwh: rate in agorot per kWh as number e.g. 54.51
- kva: KVA capacity as number e.g. 17.32
- kva_charge: KVA charge amount in NIS as number
- fixed_charges: total fixed charges in NIS as number
- connection_type: "single" or "three" phase
- breaker_amps: breaker size in amps as number
- consumption_breakdown: if Taoz bill, object with {peak_kwh, shoulder_kwh, offpeak_kwh} else null
- contact_details: object with phone, website keys if found

Return ONLY valid JSON. No markdown."""

# ---------------------------------------------------------------------------
# Electricity Savings Calculator
# ---------------------------------------------------------------------------
# 2026 tariff rates (agorot per kWh, before VAT)
TARIFFS_2026 = {
    "flat": 54.51,
    "taoz": {
        "winter": {"peak": 62.88, "shoulder": 57.96, "offpeak": 54.23},
        "transition": {"peak": 34.13, "shoulder": 29.86, "offpeak": 26.86},
        "summer": {"peak": 145.97, "shoulder": 46.55, "offpeak": 41.82},
    }
}
SUPPLIER_DISCOUNTS = {
    "cellcom": {"name": "Cellcom Energy", "discount": 0.08},
    "bezeq": {"name": "Bezeq Energy", "discount": 0.06},
    "hot": {"name": "Hot Energy", "discount": 0.07},
    "partner": {"name": "Partner Power", "discount": 0.07},
    "pazgas": {"name": "Pazgas Electric", "discount": 0.06},
    "electra": {"name": "Super Power (Electra)", "discount": 0.08},
    "amisragas": {"name": "Amisragas Electric", "discount": 0.06},
}
VAT_RATE = 0.18

def calculate_savings(bill_data: dict) -> dict:
    """Calculate potential savings from tariff switch, supplier switch, and smart scheduling."""
    consumption = bill_data.get("consumption_kwh") or 0
    period_days = bill_data.get("period_days") or 60
    rate = bill_data.get("rate_per_kwh") or 54.51
    amount = 0
    try:
        amount = float(str(bill_data.get("amount", "0")).replace(",", ""))
    except (ValueError, TypeError):
        amount = consumption * rate / 100 * (1 + VAT_RATE)

    tariff = bill_data.get("tariff_type") or "flat"
    monthly_kwh = consumption / period_days * 30 if period_days > 0 else consumption / 2
    monthly_cost = amount / period_days * 30 if period_days > 0 else amount / 2

    savings = []
    total_monthly_saving = 0

    # 1. Supplier switch savings
    best_supplier = max(SUPPLIER_DISCOUNTS.items(), key=lambda x: x[1]["discount"])
    supplier_saving = monthly_cost * best_supplier[1]["discount"]
    savings.append({
        "type": "supplier_switch",
        "title": "Switch supplier",
        "title_he": "החלפת ספק",
        "description": f"Switch to {best_supplier[1]['name']} for {best_supplier[1]['discount']:.0%} discount",
        "monthly_saving": round(supplier_saving, 0),
        "annual_saving": round(supplier_saving * 12, 0),
        "effort": "easy",
        "effort_he": "קל",
    })
    total_monthly_saving += supplier_saving

    # 2. Taoz optimization (if currently on flat tariff)
    if tariff == "flat" and monthly_kwh > 500:
        # Estimate: if 30% of usage can shift to off-peak (water heater, dryer, dishwasher at night)
        shiftable_pct = 0.30
        shiftable_kwh = monthly_kwh * shiftable_pct
        # Average saving: difference between flat rate and off-peak rate (weighted across seasons)
        avg_offpeak = (TARIFFS_2026["taoz"]["winter"]["offpeak"] +
                       TARIFFS_2026["taoz"]["transition"]["offpeak"] +
                       TARIFFS_2026["taoz"]["summer"]["offpeak"]) / 3
        avg_peak = (TARIFFS_2026["taoz"]["winter"]["peak"] +
                    TARIFFS_2026["taoz"]["transition"]["peak"] +
                    TARIFFS_2026["taoz"]["summer"]["peak"]) / 3
        # But remaining 70% may cost more during peak/shoulder
        remaining_kwh = monthly_kwh * 0.7
        # Assume 40% of remaining is peak, 60% shoulder
        remaining_cost = remaining_kwh * (0.4 * avg_peak + 0.6 * ((avg_peak + avg_offpeak) / 2)) / 100
        shifted_cost = shiftable_kwh * avg_offpeak / 100
        taoz_total = remaining_cost + shifted_cost
        flat_total = monthly_kwh * TARIFFS_2026["flat"] / 100
        taoz_saving = flat_total - taoz_total
        if taoz_saving > 0:
            savings.append({
                "type": "taoz_switch",
                "title": "Switch to Taoz tariff",
                "title_he": "מעבר לתעריף תע״ז",
                "description": f"Shift {shiftable_pct:.0%} of usage to off-peak hours",
                "monthly_saving": round(taoz_saving, 0),
                "annual_saving": round(taoz_saving * 12, 0),
                "effort": "medium",
                "effort_he": "בינוני",
            })
            total_monthly_saving += taoz_saving

    # 3. Smart scheduling (water heater, AC)
    # Water heater: ~15% of residential bill, shift to night = ~40% saving on that portion
    water_heater_saving = monthly_cost * 0.15 * 0.40
    savings.append({
        "type": "smart_scheduling",
        "title": "Smart water heater",
        "title_he": "דוד חשמל חכם",
        "description": "Heat water only during off-peak hours (Switcher device)",
        "monthly_saving": round(water_heater_saving, 0),
        "annual_saving": round(water_heater_saving * 12, 0),
        "effort": "easy",
        "effort_he": "קל",
        "device": "Switcher",
        "device_cost": 150,
    })
    total_monthly_saving += water_heater_saving

    # 4. AC optimization (summer months)
    if monthly_kwh > 800:
        ac_saving = monthly_cost * 0.10  # ~10% from smart AC scheduling
        savings.append({
            "type": "ac_optimization",
            "title": "Smart AC scheduling",
            "title_he": "ניהול מזגן חכם",
            "description": "Pre-cool before peak hours, auto-adjust temperature",
            "monthly_saving": round(ac_saving, 0),
            "annual_saving": round(ac_saving * 12, 0),
            "effort": "medium",
            "effort_he": "בינוני",
            "device": "Sensibo",
            "device_cost": 250,
        })
        total_monthly_saving += ac_saving

    # 5. Solar recommendation (for high consumers)
    solar = None
    if monthly_kwh > 800:
        solar_saving = monthly_cost * 0.65  # Solar can offset ~65% of bill
        solar = {
            "monthly_saving": round(solar_saving, 0),
            "annual_saving": round(solar_saving * 12, 0),
            "estimated_cost": 30000,
            "payback_years": round(30000 / (solar_saving * 12), 1) if solar_saving > 0 else 0,
        }

    return {
        "current_bill": {
            "monthly_cost": round(monthly_cost, 0),
            "annual_cost": round(monthly_cost * 12, 0),
            "monthly_kwh": round(monthly_kwh, 0),
            "tariff_type": tariff,
            "rate_per_kwh": rate,
            "supplier": bill_data.get("sender") or "IEC",
        },
        "savings": savings,
        "total_monthly_saving": round(total_monthly_saving, 0),
        "total_annual_saving": round(total_monthly_saving * 12, 0),
        "saving_percentage": round(total_monthly_saving / monthly_cost * 100, 0) if monthly_cost > 0 else 0,
        "optimized_monthly_cost": round(monthly_cost - total_monthly_saving, 0),
        "solar": solar,
    }

@app.post("/electricity/analyze", response_model=AIResponse, tags=["electricity"])
async def analyze_electricity(file: Annotated[UploadFile, File()]):
    """Analyze an electricity bill and return savings recommendations."""
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "File too large.")
    detected_type = _detect_type(data)

    if detected_type == "application/pdf":
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            page_images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
                page_images.append(pix.tobytes("jpeg"))
            data = page_images[0] if page_images else data
            detected_type = "image/jpeg"
        except Exception as e:
            raise HTTPException(500, f"PDF rendering failed: {e}")

    # HEIC -> JPEG
    if detected_type == "image/heic":
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.heic', delete=False) as f:
                f.write(data); tmp_in = f.name
            tmp_out = tmp_in.replace('.heic', '.jpg')
            subprocess.run(['convert', tmp_in, tmp_out], check=True, timeout=10)
            data = open(tmp_out, 'rb').read()
            detected_type = "image/jpeg"
            for p in [tmp_in, tmp_out]:
                try: os.unlink(p)
                except: pass
        except Exception as e:
            log.warning(f"HEIC conversion failed: {e}")

    # Normalize image: fix EXIF orientation, resize if too large (Claude limit ~5MB)
    if detected_type and detected_type.startswith("image/"):
        try:
            from PIL import Image, ImageOps
            import io as _io
            pil_img = Image.open(_io.BytesIO(data))
            pil_img = ImageOps.exif_transpose(pil_img)
            if pil_img.mode not in ("RGB", "L"):
                pil_img = pil_img.convert("RGB")
            max_dim = 2000
            if max(pil_img.size) > max_dim:
                ratio = max_dim / max(pil_img.size)
                pil_img = pil_img.resize((int(pil_img.size[0]*ratio), int(pil_img.size[1]*ratio)), Image.LANCZOS)
            buf = _io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            detected_type = "image/jpeg"
        except Exception as e:
            log.warning(f"Image normalization failed: {e}")

    b64 = base64.b64encode(data).decode()
    content_block = {"type": "image", "source": {"type": "base64", "media_type": detected_type, "data": b64}}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": 2048,
                      "messages": [{"role": "user", "content": [content_block, {"type": "text", "text": ANALYZE_PROMPT}]}]}
            )
        if resp.status_code != 200:
            log.warning(f"Claude Vision {resp.status_code}: {resp.text[:300]}")
            raise HTTPException(502, f"Vision API returned {resp.status_code}. Make sure the image is a clear photo of an Israeli electricity bill (חשבון חשמל).")
        raw = resp.json()["content"][0]["text"].strip()
    except httpx.TimeoutException:
        raise HTTPException(504, "Analysis timed out")

    clean = raw.replace('```json', '').replace('```', '').strip()
    bill_data = _safe_parse(clean)

    # Validate this is actually an electricity bill
    if not bill_data.get("consumption_kwh") and not bill_data.get("rate_per_kwh") and not bill_data.get("meter_number"):
        raise HTTPException(422, "This doesn't look like an electricity bill. Please upload an Israeli חשבון חשמל that shows kWh consumption, the meter number, and tariff details — not a credit card statement, payment confirmation, or other document.")

    savings = calculate_savings(bill_data)
    result = {"bill": bill_data, "savings": savings}
    return AIResponse(result=json.dumps(result))

@app.post("/ai/analyze-vision", response_model=AIResponse, tags=["ai"])
async def analyze_vision(file: Annotated[UploadFile, File()]):
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "File too large.")
    detected_type = _detect_type(data)

    # For PDFs, render first page to image
    if detected_type == "application/pdf":
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csRGB)
            data = pix.tobytes("jpeg")
            detected_type = "image/jpeg"
        except Exception as e:
            raise HTTPException(500, f"PDF rendering failed: {e}")

    b64 = base64.b64encode(data).decode()
    content_block = {"type": "image", "source": {"type": "base64", "media_type": detected_type, "data": b64}}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 2048,
                      "messages": [{"role": "user", "content": [content_block, {"type": "text", "text": ANALYZE_PROMPT}]}]}
            )
        if resp.status_code != 200:
            raise HTTPException(502, f"Claude API error: {resp.status_code}")
        result = resp.json()["content"][0]["text"].strip()
    except httpx.TimeoutException:
        raise HTTPException(504, "Analysis timed out")

    clean = result.replace('```json', '').replace('```', '').strip()
    return AIResponse(result=json.dumps(_safe_parse(clean)))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
async def _serve_icon():
    from fastapi.responses import FileResponse
    icon = os.path.join(os.path.dirname(__file__), "static", "icon-180.png")
    return FileResponse(icon, media_type="image/png", headers={"Cache-Control": "no-cache, must-revalidate"})

@app.get("/apple-touch-icon.png", include_in_schema=False)
async def apple_icon(): return await _serve_icon()

@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
async def apple_icon_pre(): return await _serve_icon()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return await _serve_icon()

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    p = os.path.join(os.path.dirname(__file__), "ui.html")
    return HTMLResponse(open(p).read())

@app.get("/install", include_in_schema=False, response_class=HTMLResponse)
async def serve_install():
    p = os.path.join(os.path.dirname(__file__), "install.html")
    return HTMLResponse(open(p).read())

@app.get("/landing", include_in_schema=False, response_class=HTMLResponse)
async def serve_landing():
    p = os.path.join(os.path.dirname(__file__), "landing.html")
    if not os.path.exists(p):
        return HTMLResponse("<h1>Landing page not found</h1>", status_code=404)
    return HTMLResponse(open(p).read())
