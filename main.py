from __future__ import annotations
import asyncio, ipaddress, logging, re, socket, httpx, os, base64, json
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("ocr_service")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

@asynccontextmanager
async def lifespan(app):
    log.info("SimpliScan ready.")
    if ANTHROPIC_API_KEY:
        log.info("Claude AI features enabled.")
    else:
        log.warning("ANTHROPIC_API_KEY not set — AI features disabled.")
    yield

app = FastAPI(title="SimpliScan OCR", version="4.0.0", lifespan=lifespan)
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://ocr-service-4e7i.onrender.com,http://localhost:8000,http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

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

    # For images: fix EXIF rotation
    if detected_type != "application/pdf":
        try:
            from PIL import Image, ImageOps
            import io as _io
            pil_img = Image.open(_io.BytesIO(data))
            pil_img = ImageOps.exif_transpose(pil_img)
            buf = _io.BytesIO()
            fmt = "JPEG" if detected_type != "image/png" else "PNG"
            pil_img.save(buf, format=fmt)
            data = buf.getvalue()
            detected_type = "image/jpeg" if fmt == "JPEG" else "image/png"
        except Exception as e:
            log.warning(f"EXIF rotate failed: {e}")

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
        # Convert PDF to images using pymupdf
        try:
            import fitz  # pymupdf
            import io as _io
            doc = fitz.open(stream=data, filetype="pdf")
            texts = []
            for page in doc:
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("jpeg")
                texts.append(await _ocr_from_bytes(img_data))
            return _clean_text("\n\n".join(texts))
        except Exception as e:
            log.warning(f"PDF to image failed: {e}")
            return f"[PDF conversion failed: {e}]"

    content_block = {"type": "image", "source": {"type": "base64", "media_type": detected_type, "data": b64}}
    extra_headers = {}

    try:
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json={"model": "claude-sonnet-4-6", "max_tokens": 4096,
                      "messages": [{"role": "user", "content": [
                          content_block,
                          {"type": "text", "text": "This document may be rotated or tilted. Detect the correct reading orientation and extract ALL text exactly as it appears. The text is likely Hebrew, English, or mixed Hebrew/English. Preserve line breaks, numbers, and structure. Output ONLY the extracted text, nothing else."}
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
    return {"status":"ok","model":"claude-vision","languages":["en","he","any"],
            "ai": bool(ANTHROPIC_API_KEY), "version":"4.0.0"}

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
    results = []
    for f in files: results.append(await process(f))
    return BatchResponse(results=results, total=len(results))

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
            json={"model": "claude-sonnet-4-6", "max_tokens": 2048,
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
        """Extract ONLY these specific fields from this document. Return a JSON object with null for missing fields:
- document_type: e.g. "Property Tax Bill"
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
    try:
        data = json.loads(clean)
        return AIResponse(result=json.dumps(data))
    except:
        return AIResponse(result=clean)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    p = os.path.join(os.path.dirname(__file__), "ui.html")
    return HTMLResponse(open(p).read())

@app.get("/landing", include_in_schema=False, response_class=HTMLResponse)
async def serve_landing():
    p = os.path.join(os.path.dirname(__file__), "landing.html")
    if not os.path.exists(p):
        return HTMLResponse("<h1>Landing page not found</h1>", status_code=404)
    return HTMLResponse(open(p).read())
