from __future__ import annotations
import asyncio, logging, re, cv2, httpx, numpy as np, os
from contextlib import asynccontextmanager
from typing import Annotated, Literal
import pytesseract
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("ocr_service")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

@asynccontextmanager
async def lifespan(app):
    log.info("Tesseract OCR service ready (en+he).")
    if ANTHROPIC_API_KEY:
        log.info("Claude AI features enabled.")
    else:
        log.warning("ANTHROPIC_API_KEY not set — AI features disabled.")
    yield

app = FastAPI(title="Simplified Access OCR", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED = {"image/jpeg","image/png","image/webp","image/tiff","image/bmp"}
MAX_BYTES = 20 * 1024 * 1024

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def _preprocess(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("לא ניתן לפענח את התמונה.")
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        scale = 2.0 if max(h, w) < 800 else 1.5
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    return gray

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def _fix_rtl_line(line):
    he_chars = sum(1 for c in line if "\u05d0" <= c <= "\u05ea")
    if he_chars > 2:
        return " ".join(reversed(line.split()))
    return line

def _clean_text(raw):
    lines = raw.splitlines()
    cleaned = [_fix_rtl_line(re.sub(r"[^\S\n]+"," ",l).strip()) for l in lines]
    return re.sub(r"\n{3,}","\n\n","\n".join(cleaned)).strip()

def _run_ocr(image):
    best = ""
    for psm in [6, 3, 4]:
        try:
            r = pytesseract.image_to_string(
                image, lang="heb+eng",
                config=f"--psm {psm} --oem 1 -c preserve_interword_spaces=1"
            )
            if len(r.strip()) > len(best.strip()):
                best = r
        except Exception as e:
            log.warning(f"Tesseract PSM {psm} failed: {e}")
    return best.strip()

async def _ocr_from_bytes(data):
    loop = asyncio.get_event_loop()
    gray = await loop.run_in_executor(None, _preprocess, data)
    raw = await loop.run_in_executor(None, _run_ocr, gray)
    cleaned = _clean_text(raw)
    enhanced = await _ai_correct(cleaned)
    return enhanced

# ---------------------------------------------------------------------------
# Claude AI helpers
# ---------------------------------------------------------------------------
async def _claude(prompt: str, max_tokens: int = 2048) -> str:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(503, "מפתח API לא מוגדר — פנה למנהל המערכת.")
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-6", "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]}
        )
    if resp.status_code != 200:
        raise HTTPException(502, f"שגיאת Claude API: {resp.status_code}")
    return resp.json()["content"][0]["text"].strip()

async def _ai_correct(text: str) -> str:
    if not ANTHROPIC_API_KEY or not text.strip():
        return text
    try:
        return await _claude(
            f"""אתה עוזר תיקון OCR המתמחה בעברית. תקן את הטקסט הבא שנסרק על ידי OCR.

הבעיה הנפוצה ביותר: מילים עברית מתמזגות יחד ללא רווח.
לדוגמה: "מהפכתהאינטרנט" צריך להיות "מהפכת האינטרנט"
לדוגמה: "ביותרבישראל" צריך להיות "ביותר בישראל"
לדוגמה: "מטעםהפרקליטות" צריך להיות "מטעם הפרקליטות"

כללים:
1. הפרד מילים שמוזגו יחד — זה התיקון הכי חשוב
2. תקן אותיות שגויות בעברית
3. אל תשנה מספרים כלל
4. שמור על מעברי שורה בדיוק
5. פלט את הטקסט המתוקן בלבד, ללא הסברים או תוספות

טקסט לתיקון:
{text}"""
        )
    except Exception as e:
        log.warning(f"AI correction failed: {e}")
        return text

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
    return {"status":"ok","model":"tesseract","languages":["en","he"],
            "ai": bool(ANTHROPIC_API_KEY), "version":"3.0.0"}

@app.post("/ocr/upload", response_model=TextResult, tags=["ocr"])
async def ocr_single(file: Annotated[UploadFile, File()], mode: OutputMode = Query("text")):
    if file.content_type not in ALLOWED: raise HTTPException(415, f"סוג קובץ לא נתמך: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "הקובץ גדול מדי (מקסימום 20MB).")
    try: text = await _ocr_from_bytes(data)
    except Exception as e: raise HTTPException(500, str(e))
    return _build(file.filename or "upload", text, mode)

@app.post("/ocr/batch", response_model=BatchResponse, tags=["ocr"])
async def ocr_batch(files: Annotated[list[UploadFile], File()], mode: OutputMode = Query("text")):
    if not files: raise HTTPException(400, "לא נבחרו קבצים.")
    if len(files) > 20: raise HTTPException(400, "מקסימום 20 קבצים.")
    async def process(f):
        if f.content_type not in ALLOWED:
            return TextResult(filename=f.filename or "?", text=f"[דלג] סוג לא נתמך")
        data = await f.read()
        if len(data) > MAX_BYTES:
            return TextResult(filename=f.filename or "?", text="[דלג] קובץ גדול מדי")
        try: text = await _ocr_from_bytes(data)
        except Exception as e: return TextResult(filename=f.filename or "?", text=f"[שגיאה] {e}")
        return _build(f.filename or "?", text, mode)
    results = []
    for f in files: results.append(await process(f))
    return BatchResponse(results=results, total=len(results))

@app.post("/ocr/url", response_model=TextResult, tags=["ocr"])
async def ocr_url(body: UrlRequest, mode: OutputMode = Query("text")):
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(str(body.url)); r.raise_for_status()
    except httpx.HTTPStatusError as e: raise HTTPException(502, f"שגיאת שרת מרוחק: {e.response.status_code}")
    except httpx.RequestError as e: raise HTTPException(502, str(e))
    if len(r.content) > MAX_BYTES: raise HTTPException(413, "הקובץ גדול מדי.")
    first = r.content[:20].lower()
    if b"<html" in first or b"<!doc" in first:
        raise HTTPException(422, "הקישור מוביל לדף אינטרנט ולא לתמונה. לחצו ימני על התמונה → \'פתח תמונה בכרטיסייה חדשה\' והעתיקו את הקישור.")
    try: text = await _ocr_from_bytes(r.content)
    except Exception as e: raise HTTPException(422, str(e))
    return _build(str(body.url).split("/")[-1] or "remote", text, mode)

# ---------------------------------------------------------------------------
# AI routes
# ---------------------------------------------------------------------------
@app.post("/ai/eli12", response_model=AIResponse, tags=["ai"])
async def explain_eli12(body: AIRequest):
    if not body.text.strip(): raise HTTPException(400, "הטקסט ריק.")
    result = await _claude(
        f"""אתה עוזר שמסביר מסמכים מורכבים בשפה פשוטה וידידותית.

המשימה: הסבר את הטקסט הבא כאילו אתה מדבר עם אדם מבוגר שלא מכיר עניינים בירוקרטיים/משפטיים.

כללים:
- השתמש בשפה פשוטה וברורה
- הסבר מונחים מורכבים במילים פשוטות
- ציין מה חשוב ומה הפעולות הנדרשות
- כתוב בעברית
- פרק למשפטים קצרים

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

כללים:
- תרגום מדויק ובהיר
- שמור על המבנה המקורי
- פלט את התרגום בלבד ללא הסברים

טקסט לתרגום:
{body.text}"""
    )
    return AIResponse(result=result)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    import os
    p = os.path.join(os.path.dirname(__file__), "ui.html")
    return HTMLResponse(open(p).read())


@app.get("/landing", include_in_schema=False, response_class=HTMLResponse)
async def serve_landing():
    import os
    p = os.path.join(os.path.dirname(__file__), "landing.html")
    if not os.path.exists(p):
        return HTMLResponse("<h1>Landing page not found</h1>", status_code=404)
    return HTMLResponse(open(p).read())
