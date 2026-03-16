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
            f"""אתה עוזר תיקון OCR. תקן שגיאות זיהוי טקסט בטקסט הבא.
כללים:
- תקן אותיות שגויות בעברית ובאנגלית
- אל תשנה מספרים שנראים כמו סיכויים (1.50, 3.95 וכו\')
- שמור על מעברי שורה
- פלט את הטקסט המתוקן בלבד, ללא הסברים

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

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SimpliScan — Document Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0c0c0c;
  --bg2: #141414;
  --bg3: #1c1c1c;
  --border: #2a2a2a;
  --border2: #383838;
  --text: #f0f0f0;
  --text2: #999;
  --text3: #555;
  --green: #00c875;
  --green-dim: #003d21;
  --orange: #ff6b35;
  --blue: #4d9fff;
  --radius: 12px;
  --font-display: 'Syne', sans-serif;
  --font-body: 'DM Sans', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font-body); font-size: 15px; min-height: 100vh; }

/* NAV */
nav {
  display: flex; align-items: center; padding: 0 40px; height: 60px;
  border-bottom: 1px solid var(--border); background: var(--bg);
  position: sticky; top: 0; z-index: 100;
}
.nav-logo { font-family: var(--font-display); font-size: 18px; font-weight: 800; color: var(--text); letter-spacing: -.3px; }
.nav-logo em { font-style: normal; color: var(--green); }
.nav-status { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text2); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--text3); transition: all .3s; }
.status-dot.online { background: var(--green); box-shadow: 0 0 8px var(--green); }
.nav-app-link { margin-left: 24px; font-size: 12px; color: var(--text2); text-decoration: none; padding: 6px 14px; border: 1px solid var(--border2); border-radius: 6px; transition: all .15s; }
.nav-app-link:hover { border-color: var(--green); color: var(--green); }

/* LAYOUT */
.app-layout { display: grid; grid-template-columns: 420px 1fr; min-height: calc(100vh - 60px); }

/* LEFT PANEL */
.left-panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--bg2); }

.panel-section { border-bottom: 1px solid var(--border); padding: 24px; }
.panel-section:last-of-type { border-bottom: none; flex: 1; display: flex; flex-direction: column; }

.section-label {
  font-family: var(--font-display); font-size: 10px; font-weight: 600;
  letter-spacing: .15em; text-transform: uppercase; color: var(--text3);
  margin-bottom: 16px;
}

/* Mode tabs */
.mode-tabs { display: flex; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 3px; gap: 2px; }
.mode-tab {
  flex: 1; padding: 8px 10px; text-align: center;
  font-family: var(--font-body); font-size: 12px; font-weight: 500;
  color: var(--text2); cursor: pointer; border-radius: 6px;
  border: none; background: transparent; transition: all .15s; letter-spacing: .02em;
}
.mode-tab:hover { color: var(--text); }
.mode-tab.active { background: var(--bg3); color: var(--green); border: 1px solid var(--border2); }

/* Drop zone */
.drop-zone {
  flex: 1; border: 1px dashed var(--border2); border-radius: var(--radius);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 10px; cursor: pointer; transition: all .2s; position: relative;
  min-height: 180px; background: var(--bg);
}
.drop-zone:hover, .drop-zone.drag-over { border-color: var(--green); background: rgba(0,200,117,.03); }
.drop-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.drop-icon { font-size: 32px; opacity: .5; pointer-events: none; }
.drop-text { font-family: var(--font-display); font-size: 13px; font-weight: 600; color: var(--text2); pointer-events: none; }
.drop-hint { font-size: 11px; color: var(--text3); pointer-events: none; }
.thumb-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
.thumb { width: 60px; height: 60px; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); }
.thumb img { width: 100%; height: 100%; object-fit: cover; }

/* URL input */
.url-input {
  width: 100%; padding: 10px 14px; background: var(--bg);
  border: 1px solid var(--border); border-radius: 8px;
  font-family: var(--font-body); font-size: 13px; color: var(--text);
  outline: none; transition: border-color .15s;
}
.url-input:focus { border-color: var(--green); }
.url-input::placeholder { color: var(--text3); }

/* Feature toggles */
.feature-toggles { display: flex; flex-direction: column; gap: 8px; }
.toggle-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 14px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; cursor: pointer; transition: all .15s;
}
.toggle-row:hover { border-color: var(--border2); }
.toggle-row.active { border-color: var(--green); background: rgba(0,200,117,.04); }
.toggle-label { display: flex; align-items: center; gap: 10px; font-size: 13px; font-weight: 500; }
.toggle-icon { font-size: 15px; }
.toggle-switch {
  width: 36px; height: 20px; border-radius: 10px;
  background: var(--bg3); border: 1px solid var(--border2);
  position: relative; transition: all .2s; flex-shrink: 0;
}
.toggle-switch::after {
  content: ''; position: absolute; width: 14px; height: 14px;
  background: var(--text3); border-radius: 50%; top: 2px; left: 2px; transition: all .2s;
}
.toggle-row.active .toggle-switch { background: var(--green-dim); border-color: var(--green); }
.toggle-row.active .toggle-switch::after { background: var(--green); left: 18px; }
.toggle-desc { font-size: 11px; color: var(--text3); margin-top: 2px; }

/* Language select */
.lang-row { display: flex; align-items: center; gap: 10px; margin-top: 12px; padding: 10px 14px; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; }
.lang-label { font-size: 12px; color: var(--text2); white-space: nowrap; }
.lang-select {
  flex: 1; background: transparent; border: none; color: var(--text);
  font-family: var(--font-body); font-size: 13px; outline: none; cursor: pointer;
}
.lang-select option { background: var(--bg2); }

/* Run button */
.run-btn {
  width: 100%; padding: 14px; background: var(--green); color: #000;
  border: none; border-radius: var(--radius); font-family: var(--font-display);
  font-size: 14px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
  cursor: pointer; transition: all .15s; display: flex; align-items: center;
  justify-content: center; gap: 8px;
}
.run-btn:hover { background: #00e688; transform: translateY(-1px); }
.run-btn:disabled { background: var(--bg3); color: var(--text3); cursor: not-allowed; transform: none; }
.run-btn.loading { background: var(--green-dim); color: var(--green); cursor: wait; }

.action-btns { display: flex; gap: 8px; margin-top: 8px; }
.action-btn {
  flex: 1; padding: 10px; background: transparent; color: var(--text2);
  border: 1px solid var(--border); border-radius: 8px; font-family: var(--font-body);
  font-size: 12px; cursor: pointer; transition: all .15s; letter-spacing: .03em;
}
.action-btn:hover { border-color: var(--border2); color: var(--text); }

/* RIGHT PANEL */
.right-panel { display: flex; flex-direction: column; background: var(--bg); }

.output-header {
  border-bottom: 1px solid var(--border); padding: 0 32px;
  display: flex; align-items: center; height: 48px; gap: 0;
}
.output-tab {
  height: 100%; padding: 0 16px; font-family: var(--font-body);
  font-size: 12px; font-weight: 500; color: var(--text3);
  border: none; background: transparent; cursor: pointer;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
  transition: all .15s; letter-spacing: .04em; text-transform: uppercase;
}
.output-tab:hover { color: var(--text2); }
.output-tab.active { color: var(--green); border-bottom-color: var(--green); }
.output-tab[style*="none"] { display: none !important; }

.output-meta { margin-left: auto; font-size: 11px; color: var(--text3); }

.output-body { flex: 1; overflow-y: auto; padding: 32px; }
.output-body::-webkit-scrollbar { width: 4px; }
.output-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.output-text {
  font-family: var(--font-body); font-size: 15px; line-height: 1.85;
  color: var(--text); white-space: pre-wrap;
}

.output-actions { display: flex; gap: 10px; margin-top: 24px; padding-top: 20px; border-top: 1px solid var(--border); }
.out-action-btn {
  padding: 8px 16px; background: var(--bg2); color: var(--text2);
  border: 1px solid var(--border); border-radius: 8px; font-family: var(--font-body);
  font-size: 12px; cursor: pointer; transition: all .15s; display: flex; align-items: center; gap: 6px;
}
.out-action-btn:hover { border-color: var(--green); color: var(--green); }
.out-action-btn.speaking { background: var(--green-dim); color: var(--green); border-color: var(--green); }

.placeholder {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  height: 100%; gap: 16px; color: var(--text3); text-align: center;
}
.placeholder-icon { font-size: 48px; opacity: .3; }
.placeholder-title { font-family: var(--font-display); font-size: 16px; font-weight: 700; color: var(--text2); }
.placeholder-sub { font-size: 13px; line-height: 1.6; max-width: 320px; }

.error-box {
  background: rgba(255,60,60,.06); border: 1px solid rgba(255,60,60,.25);
  border-radius: var(--radius); padding: 16px 20px;
  font-size: 13px; color: #ff6b6b; line-height: 1.6;
}

/* Processing */
.processing-card {
  background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 24px; max-width: 400px; margin: 0 auto;
}
.proc-title { font-family: var(--font-display); font-size: 14px; font-weight: 700; margin-bottom: 20px; color: var(--text2); letter-spacing: .05em; text-transform: uppercase; }
.proc-step { display: flex; align-items: center; gap: 12px; padding: 10px 0; font-size: 13px; }
.proc-step + .proc-step { border-top: 1px solid var(--border); }
.step-icon { font-size: 16px; width: 28px; text-align: center; }
.proc-step.done { color: var(--green); }
.proc-step.active { color: var(--text); }
.proc-step.waiting { color: var(--text3); }

@keyframes spin { to { transform: rotate(360deg); } }
.spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid var(--green-dim); border-top-color: var(--green); border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: 6px; }

@media (max-width: 800px) {
  .app-layout { grid-template-columns: 1fr; }
  .left-panel { border-right: none; border-bottom: 1px solid var(--border); }
  nav { padding: 0 20px; }
}
</style>
</head>
<body>

<nav>
  <span class="nav-logo">Simpli<em>Scan</em></span>
  <div class="nav-status">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusLabel">Connecting</span>
  </div>
  <a href="/landing" class="nav-app-link">About →</a>
</nav>

<div class="app-layout">

  <!-- LEFT PANEL -->
  <div class="left-panel">

    <!-- Input mode -->
    <div class="panel-section">
      <div class="section-label">Input mode</div>
      <div class="mode-tabs">
        <button class="mode-tab active" onclick="switchMode('upload')">Upload</button>
        <button class="mode-tab" onclick="switchMode('batch')">Batch</button>
        <button class="mode-tab" onclick="switchMode('url')">URL</button>
      </div>
    </div>

    <!-- File drop -->
    <div class="panel-section" id="uploadSection" style="flex:1;display:flex;flex-direction:column;">
      <div class="section-label">Document</div>
      <div class="drop-zone" id="dropZone"
           ondragover="onDragOver(event)" ondragleave="onDragLeave()" ondrop="onDrop(event)">
        <input type="file" id="fileInput" accept="image/*" onchange="onFileSelect(event)"/>
        <div class="drop-icon">⌗</div>
        <div class="drop-text">Drop image here</div>
        <div class="drop-hint">JPEG · PNG · WEBP · TIFF · BMP &nbsp;·&nbsp; max 20MB</div>
      </div>
      <div class="thumb-row" id="thumbRow"></div>
    </div>

    <!-- URL input -->
    <div class="panel-section" id="urlSection" style="display:none;">
      <div class="section-label">Image URL</div>
      <input class="url-input" id="urlInput" type="url" placeholder="https://example.com/document.jpg"/>
    </div>

    <!-- AI features -->
    <div class="panel-section">
      <div class="section-label">AI features</div>
      <div class="feature-toggles">
        <div class="toggle-row active" id="toggleOCR">
          <div>
            <div class="toggle-label"><span class="toggle-icon">📖</span> OCR extraction</div>
            <div class="toggle-desc">Always enabled</div>
          </div>
          <div class="toggle-switch"></div>
        </div>
        <div class="toggle-row" id="toggleELI12" onclick="toggleFeature('eli12')">
          <div>
            <div class="toggle-label"><span class="toggle-icon">🧒</span> Plain language</div>
            <div class="toggle-desc">Simplify complex text</div>
          </div>
          <div class="toggle-switch"></div>
        </div>
        <div class="toggle-row" id="toggleTranslate" onclick="toggleFeature('translate')">
          <div>
            <div class="toggle-label"><span class="toggle-icon">🌐</span> Translate</div>
            <div class="toggle-desc">Convert to another language</div>
          </div>
          <div class="toggle-switch"></div>
        </div>
      </div>
      <div class="lang-row" id="langRow" style="display:none;">
        <span class="lang-label">Translate to</span>
        <select class="lang-select" id="targetLang">
          <option value="Hebrew">עברית Hebrew</option>
          <option value="English">English</option>
          <option value="Arabic">عربي Arabic</option>
          <option value="Russian">Русский Russian</option>
          <option value="French">Français French</option>
          <option value="Spanish">Español Spanish</option>
          <option value="Amharic">አማርኛ Amharic</option>
          <option value="Tigrinya">ትግርኛ Tigrinya</option>
        </select>
      </div>
    </div>

    <!-- Run -->
    <div class="panel-section">
      <button class="run-btn" id="runBtn" onclick="runOCR()">
        <span>▶</span>&nbsp; Process document
      </button>
      <div class="action-btns">
        <button class="action-btn" onclick="resetAll()">Reset</button>
        <button class="action-btn" onclick="location.reload()">Refresh</button>
      </div>
    </div>

  </div>

  <!-- RIGHT PANEL -->
  <div class="right-panel">
    <div class="output-header">
      <button class="output-tab active" onclick="switchTab('ocr')" id="tabOCR">Extracted text</button>
      <button class="output-tab" onclick="switchTab('eli12')" id="tabELI12" style="display:none">Plain language</button>
      <button class="output-tab" onclick="switchTab('translation')" id="tabTranslation" style="display:none">Translation</button>
      <span class="output-meta" id="outputMeta"></span>
    </div>
    <div class="output-body" id="outputBody">
      <div class="placeholder">
        <div class="placeholder-icon">▤</div>
        <div class="placeholder-title">No document loaded</div>
        <div class="placeholder-sub">Upload an image or paste a URL, then click Process document to extract text.</div>
      </div>
    </div>
  </div>

</div>

<script>
let inputMode = "upload", selectedFiles = [], features = {eli12:false, translate:false};
let results = {ocr:"", eli12:"", translation:""};
let speaking = false, activeTab = "ocr";

async function checkHealth() {
  try {
    const r = await fetch("/health");
    if (r.ok) {
      document.getElementById("statusDot").className = "status-dot online";
      document.getElementById("statusLabel").textContent = "Online";
    }
  } catch {}
}
checkHealth();

function switchMode(mode) {
  inputMode = mode;
  document.querySelectorAll(".mode-tab").forEach((t,i) => t.classList.toggle("active", ["upload","batch","url"][i]===mode));
  document.getElementById("uploadSection").style.display = mode!=="url" ? "flex" : "none";
  document.getElementById("urlSection").style.display = mode==="url" ? "" : "none";
  document.getElementById("fileInput").multiple = mode==="batch";
  selectedFiles=[]; renderThumbs();
}

function toggleFeature(key) {
  features[key] = !features[key];
  document.getElementById("toggle"+key.charAt(0).toUpperCase()+key.slice(1))
    .classList.toggle("active", features[key]);
  if (key==="translate") document.getElementById("langRow").style.display = features.translate ? "flex" : "none";
}

function onFileSelect(e) { selectedFiles=Array.from(e.target.files); renderThumbs(); }
function onDragOver(e) { e.preventDefault(); document.getElementById("dropZone").classList.add("drag-over"); }
function onDragLeave() { document.getElementById("dropZone").classList.remove("drag-over"); }
function onDrop(e) {
  e.preventDefault(); document.getElementById("dropZone").classList.remove("drag-over");
  selectedFiles=Array.from(e.dataTransfer.files).filter(f=>f.type.startsWith("image/")); renderThumbs();
}
function renderThumbs() {
  const row=document.getElementById("thumbRow"); row.innerHTML="";
  selectedFiles.slice(0,8).forEach(f=>{
    const d=document.createElement("div"); d.className="thumb";
    const i=document.createElement("img"); i.src=URL.createObjectURL(f);
    d.appendChild(i); row.appendChild(d);
  });
}

async function runOCR() {
  const btn=document.getElementById("runBtn");
  btn.disabled=true; btn.className="run-btn loading";
  btn.innerHTML='<span class="spinner"></span> Processing...';
  showProcessing();

  try {
    // Step 1: OCR
    setStep(1,"active");
    let ocrText="";
    if (inputMode==="url") {
      const url=document.getElementById("urlInput").value.trim();
      if (!url) throw new Error("Please enter an image URL.");
      const r=await fetch("/ocr/url?mode=text",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})});
      if (!r.ok){const e=await r.json();throw new Error(Array.isArray(e.detail)?e.detail.map(x=>x.msg).join(", "):e.detail);}
      ocrText=(await r.json()).text;
    } else {
      if (!selectedFiles.length) throw new Error("Please select an image file.");
      const fd=new FormData();
      if (inputMode==="batch") selectedFiles.forEach(f=>fd.append("files",f));
      else fd.append("file",selectedFiles[0]);
      const ep=inputMode==="batch"?"/ocr/batch":"/ocr/upload";
      const r=await fetch(ep+"?mode=text",{method:"POST",body:fd});
      if (!r.ok){const e=await r.json();throw new Error(Array.isArray(e.detail)?e.detail.map(x=>x.msg).join(", "):e.detail);}
      const d=await r.json();
      ocrText=inputMode==="batch"?d.results.map(r=>r.text).join("\\n---\\n"):d.text;
    }
    setStep(1,"done"); results.ocr=ocrText;
    await delay(200); setStep(2,"active");
    await delay(300); setStep(2,"done");

    if (features.eli12) {
      setStep(3,"active");
      const r=await fetch("/ai/eli12",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:ocrText})});
      if (!r.ok) throw new Error("AI simplification failed.");
      results.eli12=(await r.json()).result;
      setStep(3,"done");
    }

    if (features.translate) {
      setStep(4,"active");
      const lang=document.getElementById("targetLang").value;
      const src=features.eli12?results.eli12:ocrText;
      const r=await fetch("/ai/translate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:src,language:lang})});
      if (!r.ok) throw new Error("Translation failed.");
      results.translation=(await r.json()).result;
      setStep(4,"done");
    }

    renderResults();
  } catch(err) {
    document.getElementById("outputBody").innerHTML=`<div class="error-box">⚠ ${escHtml(err.message)}</div>`;
  } finally {
    btn.disabled=false; btn.className="run-btn";
    btn.innerHTML="<span>▶</span>&nbsp; Process document";
  }
}

function delay(ms) { return new Promise(r=>setTimeout(r,ms)); }

function showProcessing() {
  const eli12Visible=features.eli12, transVisible=features.translate;
  document.getElementById("outputBody").innerHTML=`
    <div class="processing-card">
      <div class="proc-title">Processing</div>
      <div class="proc-step waiting" id="s1"><span class="step-icon">📖</span> OCR extraction</div>
      <div class="proc-step waiting" id="s2"><span class="step-icon">🤖</span> AI correction</div>
      ${eli12Visible?'<div class="proc-step waiting" id="s3"><span class="step-icon">🧒</span> Plain language</div>':''}
      ${transVisible?'<div class="proc-step waiting" id="s4"><span class="step-icon">🌐</span> Translation</div>':''}
    </div>`;
  document.getElementById("outputMeta").textContent="";
  ["tabELI12","tabTranslation"].forEach(id=>{
    const t=document.getElementById(id);
    if(t) t.style.display="none";
  });
}

function setStep(n, state) {
  const el=document.getElementById("s"+n);
  if(el) el.className="proc-step "+state;
}

function renderResults() {
  const tabIds = {ocr:"tabOCR", eli12:"tabELI12", translation:"tabTranslation"};
  Object.keys(tabIds).forEach(k => {
    const t=document.getElementById(tabIds[k]);
    if(t) t.style.display = (k==="ocr"||(k==="eli12"&&features.eli12)||(k==="translation"&&features.translate)) ? "" : "none";
  });

  if (features.translate) switchTab("translation");
  else if (features.eli12) switchTab("eli12");
  else switchTab("ocr");

  const words = results.ocr.split(/\\s+/).filter(Boolean).length;
  document.getElementById("outputMeta").textContent = `${words} words`;
}

function switchTab(tab) {
  activeTab=tab;
  document.querySelectorAll(".output-tab").forEach(t => t.classList.remove("active"));
  const tabMap={ocr:"tabOCR", eli12:"tabELI12", translation:"tabTranslation"};
  const activeEl=document.getElementById(tabMap[tab]);
  if(activeEl) activeEl.classList.add("active");

  const text=results[tab]||"";
  if (!text) { document.getElementById("outputBody").innerHTML='<div class="placeholder"><div class="placeholder-icon">▤</div><div class="placeholder-title">No output yet</div></div>'; return; }

  const isRtl=/[\\u05d0-\\u05ea\\u0600-\\u06ff]/.test(text);
  document.getElementById("outputBody").innerHTML=`
    <pre class="output-text" style="direction:${isRtl?"rtl":"ltr"};text-align:${isRtl?"right":"left"}">${escHtml(text)}</pre>
    <div class="output-actions">
      <button class="out-action-btn" id="speakBtn" onclick="speakText(this, ${JSON.stringify(text)})">🔊 Listen</button>
      <button class="out-action-btn" onclick="copyText(this, ${JSON.stringify(text)})">📋 Copy</button>
    </div>`;
}

function speakText(btn, text) {
  if(speaking){window.speechSynthesis.cancel();speaking=false;btn.className="out-action-btn";btn.innerHTML="🔊 Listen";return;}
  const isHe=/[\\u05d0-\\u05ea]/.test(text), isAr=/[\\u0600-\\u06ff]/.test(text);
  const utter=new SpeechSynthesisUtterance(text);
  utter.lang=isHe?"he-IL":isAr?"ar-SA":"en-US"; utter.rate=0.88;
  utter.onend=()=>{speaking=false;btn.className="out-action-btn";btn.innerHTML="🔊 Listen";};
  speaking=true; btn.className="out-action-btn speaking"; btn.innerHTML="⏹ Stop";
  window.speechSynthesis.speak(utter);
}

function copyText(btn, text) {
  navigator.clipboard.writeText(text).then(()=>{btn.innerHTML="✅ Copied";setTimeout(()=>btn.innerHTML="📋 Copy",1500);});
}

function resetAll() {
  selectedFiles=[]; results={ocr:"",eli12:"",translation:""};
  document.getElementById("fileInput").value="";
  document.getElementById("urlInput").value="";
  document.getElementById("thumbRow").innerHTML="";
  document.getElementById("outputBody").innerHTML='<div class="placeholder"><div class="placeholder-icon">▤</div><div class="placeholder-title">No document loaded</div><div class="placeholder-sub">Upload an image or paste a URL, then click Process document to extract text.</div></div>';
  document.getElementById("outputMeta").textContent="";
  if(speaking){window.speechSynthesis.cancel();speaking=false;}
}

function escHtml(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}
</script>
</body>
</html>"""

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_UI_HTML)
