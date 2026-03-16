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
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>גישה פשוטה — הבנת מסמכים</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f5f0e8;
    --card: #ffffff;
    --border: #d4c9b0;
    --primary: #2c5f8a;
    --primary-light: #e8f0f8;
    --accent: #e07b2a;
    --text: #2a2a2a;
    --text-dim: #666;
    --green: #2d7a4a;
    --radius: 12px;
    --shadow: 0 2px 12px rgba(0,0,0,0.08);
    --font: "Heebo", sans-serif;
    --font-size-base: 18px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: var(--font-size-base);
    min-height: 100vh;
    direction: rtl;
  }
  header {
    background: var(--primary);
    color: white;
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: var(--shadow);
  }
  .logo-icon { font-size: 36px; }
  .logo-text h1 { font-size: 24px; font-weight: 700; }
  .logo-text p { font-size: 14px; opacity: 0.85; margin-top: 2px; }
  .status-pill {
    margin-right: auto;
    background: rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #aaa; }
  .status-dot.online { background: #4ade80; box-shadow: 0 0 6px #4ade80; }

  main {
    max-width: 960px;
    margin: 32px auto;
    padding: 0 20px;
    display: flex;
    flex-direction: column;
    gap: 24px;
  }

  .card {
    background: var(--card);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    overflow: hidden;
  }
  .card-header {
    background: var(--primary-light);
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .card-header h2 {
    font-size: 18px;
    font-weight: 600;
    color: var(--primary);
  }
  .card-icon { font-size: 22px; }
  .card-body { padding: 24px; }

  /* Upload area */
  .drop-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 40px 24px;
    text-align: center;
    cursor: pointer;
    transition: all .2s;
    position: relative;
    background: #faf8f4;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--primary);
    background: var(--primary-light);
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }
  .drop-icon { font-size: 48px; margin-bottom: 12px; }
  .drop-text { font-size: 18px; font-weight: 500; color: var(--primary); }
  .drop-hint { font-size: 14px; color: var(--text-dim); margin-top: 6px; }
  .thumb-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; justify-content: center; }
  .thumb { width: 72px; height: 72px; border-radius: 8px; overflow: hidden; border: 2px solid var(--border); }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }

  /* Mode tabs */
  .mode-tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .mode-tab {
    flex: 1; padding: 12px; border: 2px solid var(--border);
    border-radius: 10px; font-family: var(--font); font-size: 16px;
    cursor: pointer; background: white; color: var(--text-dim);
    transition: all .15s; text-align: center; font-weight: 500;
  }
  .mode-tab:hover { border-color: var(--primary); color: var(--primary); }
  .mode-tab.active { border-color: var(--primary); background: var(--primary); color: white; }

  /* URL input */
  .url-input {
    width: 100%; padding: 14px 16px; border: 2px solid var(--border);
    border-radius: 10px; font-family: var(--font); font-size: 16px;
    outline: none; transition: border-color .15s;
  }
  .url-input:focus { border-color: var(--primary); }

  /* Options */
  .options-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px; }
  .option-btn {
    padding: 10px 18px; border: 2px solid var(--border);
    border-radius: 10px; font-family: var(--font); font-size: 15px;
    cursor: pointer; background: white; color: var(--text-dim);
    transition: all .15s; font-weight: 500;
  }
  .option-btn:hover { border-color: var(--primary); color: var(--primary); }
  .option-btn.active { border-color: var(--primary); background: var(--primary-light); color: var(--primary); }

  /* Language selector */
  .lang-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .lang-label { font-size: 16px; font-weight: 500; white-space: nowrap; }
  .lang-select {
    padding: 10px 14px; border: 2px solid var(--border); border-radius: 10px;
    font-family: var(--font); font-size: 15px; background: white;
    outline: none; cursor: pointer; min-width: 160px;
  }
  .lang-select:focus { border-color: var(--primary); }

  /* Run button */
  .run-btn {
    width: 100%; padding: 18px; background: var(--accent); color: white;
    border: none; border-radius: var(--radius); font-family: var(--font);
    font-size: 20px; font-weight: 700; cursor: pointer; transition: all .15s;
    display: flex; align-items: center; justify-content: center; gap: 10px;
  }
  .run-btn:hover { background: #c96d20; transform: translateY(-1px); box-shadow: 0 4px 16px rgba(224,123,42,0.3); }
  .run-btn:disabled { background: #ccc; cursor: not-allowed; transform: none; box-shadow: none; }

  .secondary-btns { display: flex; gap: 10px; margin-top: 10px; }
  .secondary-btn {
    flex: 1; padding: 12px; background: white; color: var(--text-dim);
    border: 2px solid var(--border); border-radius: 10px; font-family: var(--font);
    font-size: 15px; cursor: pointer; transition: all .15s; font-weight: 500;
  }
  .secondary-btn:hover { border-color: var(--primary); color: var(--primary); }

  /* Output */
  .output-tabs { display: flex; border-bottom: 2px solid var(--border); margin-bottom: 0; }
  .output-tab {
    padding: 14px 20px; font-family: var(--font); font-size: 16px;
    font-weight: 600; color: var(--text-dim); cursor: pointer;
    border-bottom: 3px solid transparent; margin-bottom: -2px;
    transition: all .15s; background: none; border-top: none;
    border-left: none; border-right: none;
  }
  .output-tab:hover { color: var(--primary); }
  .output-tab.active { color: var(--primary); border-bottom-color: var(--primary); }

  .output-panel { display: none; padding: 24px; }
  .output-panel.active { display: block; }

  .output-text {
    font-size: 18px; line-height: 1.9; white-space: pre-wrap;
    color: var(--text); direction: rtl;
  }
  .output-actions {
    display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap;
    padding-top: 16px; border-top: 1px solid var(--border);
  }
  .action-btn {
    padding: 10px 18px; background: var(--primary-light); color: var(--primary);
    border: 2px solid var(--primary); border-radius: 10px; font-family: var(--font);
    font-size: 15px; cursor: pointer; font-weight: 600; transition: all .15s;
    display: flex; align-items: center; gap: 6px;
  }
  .action-btn:hover { background: var(--primary); color: white; }
  .action-btn.speaking { background: var(--accent); color: white; border-color: var(--accent); }

  .placeholder {
    text-align: center; padding: 48px 24px; color: var(--text-dim);
  }
  .placeholder-icon { font-size: 56px; margin-bottom: 16px; }
  .placeholder-text { font-size: 18px; }

  .error-box {
    background: #fff0f0; border: 2px solid #ffaaaa; border-radius: 10px;
    padding: 16px 20px; color: #c00; font-size: 16px; line-height: 1.6;
  }

  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block; width: 20px; height: 20px;
    border: 3px solid rgba(255,255,255,0.4);
    border-top-color: white; border-radius: 50%;
    animation: spin .7s linear infinite;
  }

  .processing-steps {
    background: var(--primary-light); border-radius: 10px;
    padding: 20px 24px; margin-bottom: 16px;
  }
  .step { display: flex; align-items: center; gap: 12px; padding: 8px 0; font-size: 16px; }
  .step-icon { font-size: 20px; width: 28px; text-align: center; }
  .step.done .step-icon::after { content: " ✅"; }
  .step.active { color: var(--primary); font-weight: 600; }
  .step.waiting { color: var(--text-dim); }

  footer {
    text-align: center; padding: 24px; color: var(--text-dim);
    font-size: 14px; border-top: 1px solid var(--border);
    margin-top: 16px;
  }

  @media (max-width: 600px) {
    header { padding: 16px; }
    .logo-text h1 { font-size: 20px; }
    main { margin: 16px auto; padding: 0 12px; }
    .card-body { padding: 16px; }
    .run-btn { font-size: 18px; padding: 16px; }
  }
</style>
</head>
<body>

<header>
  <div class="logo-icon">📄</div>
  <div class="logo-text">
    <h1>גישה פשוטה</h1>
    <p>הבנת מסמכים לכולם</p>
  </div>
  <div class="status-pill">
    <span class="status-dot" id="statusDot"></span>
    <span id="statusLabel">מתחבר...</span>
  </div>
</header>

<main>

  <!-- Upload Card -->
  <div class="card">
    <div class="card-header">
      <span class="card-icon">📸</span>
      <h2>העלאת מסמך</h2>
    </div>
    <div class="card-body">
      <!-- Mode tabs -->
      <div class="mode-tabs">
        <button class="mode-tab active" onclick="switchMode(\\'upload\\')">📎 העלאת קובץ</button>
        <button class="mode-tab" onclick="switchMode(\\'batch\\')">📚 מספר קבצים</button>
        <button class="mode-tab" onclick="switchMode(\\'url\\')">🔗 קישור</button>
      </div>

      <!-- Upload zone -->
      <div id="uploadSection">
        <div class="drop-zone" id="dropZone"
             ondragover="onDragOver(event)" ondragleave="onDragLeave()" ondrop="onDrop(event)">
          <input type="file" id="fileInput" accept="image/*" onchange="onFileSelect(event)"/>
          <div class="drop-icon">🖼️</div>
          <div class="drop-text">גררו תמונה לכאן או לחצו לבחירה</div>
          <div class="drop-hint">תומך ב: JPEG, PNG, WEBP, TIFF, BMP</div>
        </div>
        <div class="thumb-row" id="thumbRow"></div>
      </div>

      <!-- URL input -->
      <div id="urlSection" style="display:none">
        <input class="url-input" id="urlInput" type="url" placeholder="הדביקו כאן קישור לתמונה..."/>
      </div>
    </div>
  </div>

  <!-- Options Card -->
  <div class="card">
    <div class="card-header">
      <span class="card-icon">⚙️</span>
      <h2>אפשרויות עיבוד</h2>
    </div>
    <div class="card-body">

      <!-- AI Features -->
      <div style="margin-bottom: 16px;">
        <div style="font-size: 16px; font-weight: 600; margin-bottom: 10px; color: var(--primary);">🤖 תכונות בינה מלאכותית</div>
        <div class="options-grid">
          <button class="option-btn active" id="btnOCR" onclick="toggleOpt(\\'ocr\\')">📖 זיהוי טקסט (OCR)</button>
          <button class="option-btn" id="btnELI12" onclick="toggleOpt(\\'eli12\\')">🧒 הסבר פשוט</button>
          <button class="option-btn" id="btnTranslate" onclick="toggleOpt(\\'translate\\')">🌐 תרגום</button>
        </div>
      </div>

      <!-- Translation language -->
      <div class="lang-row" id="langRow" style="display:none">
        <span class="lang-label">🌍 תרגם ל:</span>
        <select class="lang-select" id="targetLang">
          <option value="עברית">עברית</option>
          <option value="English">English</option>
          <option value="Arabic">عربي</option>
          <option value="Russian">Русский</option>
          <option value="French">Français</option>
          <option value="Spanish">Español</option>
          <option value="Amharic">አማርኛ</option>
          <option value="Tigrinya">ትግርኛ</option>
        </select>
      </div>

      <!-- Run button -->
      <button class="run-btn" id="runBtn" onclick="runOCR()">
        <span>🚀</span> <span>עבד מסמך</span>
      </button>
      <div class="secondary-btns">
        <button class="secondary-btn" onclick="resetAll()">🔄 איפוס</button>
        <button class="secondary-btn" onclick="location.reload()">↺ רענן</button>
      </div>
    </div>
  </div>

  <!-- Output Card -->
  <div class="card" id="outputCard" style="display:none">
    <div class="card-header">
      <span class="card-icon">📋</span>
      <h2>תוצאות</h2>
    </div>

    <!-- Processing steps -->
    <div id="processingSteps" style="display:none; padding: 20px 24px;">
      <div class="processing-steps">
        <div class="step waiting" id="step1"><span class="step-icon">📖</span> זיהוי טקסט (OCR)</div>
        <div class="step waiting" id="step2"><span class="step-icon">🤖</span> תיקון שגיאות AI</div>
        <div class="step waiting" id="step3"><span class="step-icon">🧒</span> הסבר פשוט</div>
        <div class="step waiting" id="step4"><span class="step-icon">🌐</span> תרגום</div>
      </div>
    </div>

    <!-- Output tabs -->
    <div class="output-tabs" id="outputTabs" style="display:none">
      <button class="output-tab active" onclick="switchOutputTab(\\'ocr\\')" id="tabOCR">📖 טקסט מקורי</button>
      <button class="output-tab" onclick="switchOutputTab(\\'eli12\\')" id="tabELI12" style="display:none">🧒 הסבר פשוט</button>
      <button class="output-tab" onclick="switchOutputTab(\\'translation\\')" id="tabTranslation" style="display:none">🌐 תרגום</button>
    </div>

    <!-- OCR panel -->
    <div class="output-panel active" id="panelOCR">
      <div class="placeholder" id="placeholderMsg">
        <div class="placeholder-icon">⬆️</div>
        <div class="placeholder-text">העלו מסמך ולחצו "עבד מסמך"</div>
      </div>
    </div>

    <!-- ELI12 panel -->
    <div class="output-panel" id="panelELI12"></div>

    <!-- Translation panel -->
    <div class="output-panel" id="panelTranslation"></div>
  </div>

</main>

<footer>
  גישה פשוטה — OCR + AI לכולם &nbsp;|&nbsp; EasyOCR · Tesseract · Claude AI
</footer>

<script>
let inputMode = "upload";
let selectedFiles = [];
let opts = { ocr: true, eli12: false, translate: false };
let currentOutput = { ocr: "", eli12: "", translation: "" };
let speaking = false;
let speechUtterance = null;

// Status check
async function checkHealth() {
  try {
    const r = await fetch("/health");
    if (r.ok) {
      document.getElementById("statusDot").className = "status-dot online";
      document.getElementById("statusLabel").textContent = "מוכן";
    }
  } catch {}
}
checkHealth();

function switchMode(mode) {
  inputMode = mode;
  document.querySelectorAll(".mode-tab").forEach((t, i) => {
    t.classList.toggle("active", ["upload","batch","url"][i] === mode);
  });
  document.getElementById("uploadSection").style.display = mode !== "url" ? "" : "none";
  document.getElementById("urlSection").style.display = mode === "url" ? "" : "none";
  document.getElementById("fileInput").multiple = mode === "batch";
  selectedFiles = [];
  renderThumbs();
}

function toggleOpt(key) {
  if (key === "ocr") return; // OCR always on
  opts[key] = !opts[key];
  document.getElementById("btn" + key.charAt(0).toUpperCase() + key.slice(1))
    .classList.toggle("active", opts[key]);
  if (key === "translate") {
    document.getElementById("langRow").style.display = opts.translate ? "flex" : "none";
  }
  // Show/hide ELI12 step
  document.getElementById("step3").style.display = opts.eli12 ? "" : "none";
  document.getElementById("step4").style.display = opts.translate ? "" : "none";
}

function onFileSelect(e) { selectedFiles = Array.from(e.target.files); renderThumbs(); }
function onDragOver(e) { e.preventDefault(); document.getElementById("dropZone").classList.add("drag-over"); }
function onDragLeave() { document.getElementById("dropZone").classList.remove("drag-over"); }
function onDrop(e) {
  e.preventDefault();
  document.getElementById("dropZone").classList.remove("drag-over");
  selectedFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith("image/"));
  renderThumbs();
}
function renderThumbs() {
  const row = document.getElementById("thumbRow");
  row.innerHTML = "";
  selectedFiles.slice(0, 6).forEach(f => {
    const d = document.createElement("div"); d.className = "thumb";
    const i = document.createElement("img"); i.src = URL.createObjectURL(f);
    d.appendChild(i); row.appendChild(d);
  });
}

function setStep(n, state) {
  const el = document.getElementById("step" + n);
  if (!el) return;
  el.className = "step " + state;
}

async function runOCR() {
  const btn = document.getElementById("runBtn");
  btn.disabled = true;
  btn.innerHTML = "<span class=\\"spinner\\"></span> מעבד...";

  // Show output card and processing steps
  document.getElementById("outputCard").style.display = "";
  document.getElementById("processingSteps").style.display = "";
  document.getElementById("outputTabs").style.display = "none";
  document.getElementById("panelOCR").innerHTML = "";
  document.getElementById("panelELI12").innerHTML = "";
  document.getElementById("panelTranslation").innerHTML = "";
  document.getElementById("panelOCR").className = "output-panel active";
  document.getElementById("panelELI12").className = "output-panel";
  document.getElementById("panelTranslation").className = "output-panel";

  setStep(1, "active"); setStep(2, "waiting"); setStep(3, "waiting"); setStep(4, "waiting");

  try {
    // Step 1: OCR
    let ocrText = "";
    if (inputMode === "url") {
      const url = document.getElementById("urlInput").value.trim();
      if (!url) throw new Error("נא להזין קישור לתמונה");
      const r = await fetch("/ocr/url?mode=text", {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({url})
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      ocrText = (await r.json()).text;
    } else {
      if (!selectedFiles.length) throw new Error("נא לבחור קובץ");
      const fd = new FormData();
      if (inputMode === "batch") selectedFiles.forEach(f => fd.append("files", f));
      else fd.append("file", selectedFiles[0]);
      const endpoint = inputMode === "batch" ? "/ocr/batch" : "/ocr/upload";
      const r = await fetch(endpoint + "?mode=text", {method:"POST", body:fd});
      if (!r.ok) { const e = await r.json(); throw new Error(Array.isArray(e.detail) ? e.detail.map(x=>x.msg).join(", ") : e.detail); }
      const data = await r.json();
      ocrText = inputMode === "batch" ? data.results.map(r => r.text).join("\\n---\\n") : data.text;
    }

    setStep(1, "done"); setStep(2, "active");
    currentOutput.ocr = ocrText;

    // Step 2: OCR text is already AI-corrected by the backend
    await new Promise(r => setTimeout(r, 300));
    setStep(2, "done");

    // Show OCR result
    showOutput("ocr", ocrText);

    // Step 3: ELI12
    if (opts.eli12) {
      setStep(3, "active");
      const eli12 = await callAI("eli12", ocrText);
      currentOutput.eli12 = eli12;
      showOutput("eli12", eli12);
      setStep(3, "done");
    }

    // Step 4: Translation
    if (opts.translate) {
      setStep(4, "active");
      const lang = document.getElementById("targetLang").value;
      const sourceText = opts.eli12 ? currentOutput.eli12 : ocrText;
      const translation = await callAI("translate", sourceText, lang);
      currentOutput.translation = translation;
      showOutput("translation", translation);
      setStep(4, "done");
    }

    document.getElementById("processingSteps").style.display = "none";
    document.getElementById("outputTabs").style.display = "flex";

    // Show first available tab
    if (opts.translate) switchOutputTab("translation");
    else if (opts.eli12) switchOutputTab("eli12");
    else switchOutputTab("ocr");

  } catch(err) {
    document.getElementById("panelOCR").innerHTML =
      `<div class="error-box">⚠️ ${escHtml(err.message)}</div>`;
    document.getElementById("processingSteps").style.display = "none";
    document.getElementById("outputTabs").style.display = "flex";
  } finally {
    btn.disabled = false;
    btn.innerHTML = "<span>🚀</span> <span>עבד מסמך</span>";
  }
}

async function callAI(type, text, lang) {
  const r = await fetch("/ai/" + type, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text, language: lang || ""})
  });
  if (!r.ok) {
    const e = await r.json();
    throw new Error(e.detail || "שגיאת AI");
  }
  return (await r.json()).result;
}

function showOutput(type, text) {
  const panel = document.getElementById("panel" + type.charAt(0).toUpperCase() + type.slice(1).replace("12","12"));
  const tabId = "tab" + type.charAt(0).toUpperCase() + type.slice(1).replace("12","12");

  // Fix panel id lookup
  const panelMap = {ocr: "panelOCR", eli12: "panelELI12", translation: "panelTranslation"};
  const tabMap = {ocr: "tabOCR", eli12: "tabELI12", translation: "tabTranslation"};
  const p = document.getElementById(panelMap[type]);
  const t = document.getElementById(tabMap[type]);
  if (t) t.style.display = "";

  const isHe = /[\\u05d0-\\u05ea]/.test(text);
  p.innerHTML = `
    <pre class="output-text" style="direction:${isHe?"rtl":"ltr"}">${escHtml(text)}</pre>
    <div class="output-actions">
      <button class="action-btn" onclick="speakText(\\`${escHtml(text).replace(/\\`/g,"\\\\`")}\\`, this)">
        🔊 האזן
      </button>
      <button class="action-btn" onclick="copyText(\\`${escHtml(text).replace(/\\`/g,"\\\\`")}\\`, this)">
        📋 העתק
      </button>
    </div>`;
}

function switchOutputTab(type) {
  const panelMap = {ocr: "panelOCR", eli12: "panelELI12", translation: "panelTranslation"};
  const tabMap = {ocr: "tabOCR", eli12: "tabELI12", translation: "tabTranslation"};
  Object.keys(panelMap).forEach(k => {
    document.getElementById(panelMap[k]).className = "output-panel" + (k === type ? " active" : "");
    const tab = document.getElementById(tabMap[k]);
    if (tab) tab.classList.toggle("active", k === type);
  });
}

function speakText(text, btn) {
  if (speaking) {
    window.speechSynthesis.cancel();
    speaking = false;
    document.querySelectorAll(".action-btn.speaking").forEach(b => {
      b.className = "action-btn"; b.textContent = "🔊 האזן";
    });
    return;
  }
  const isHe = /[\\u05d0-\\u05ea]/.test(text);
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = isHe ? "he-IL" : "en-US";
  utter.rate = 0.85;
  utter.onend = () => {
    speaking = false;
    btn.className = "action-btn"; btn.innerHTML = "🔊 האזן";
  };
  speaking = true;
  btn.className = "action-btn speaking"; btn.innerHTML = "⏹ עצור";
  window.speechSynthesis.speak(utter);
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    btn.innerHTML = "✅ הועתק!";
    setTimeout(() => btn.innerHTML = "📋 העתק", 1500);
  });
}

function resetAll() {
  selectedFiles = [];
  document.getElementById("fileInput").value = "";
  document.getElementById("urlInput").value = "";
  document.getElementById("thumbRow").innerHTML = "";
  document.getElementById("outputCard").style.display = "none";
  currentOutput = {ocr:"", eli12:"", translation:""};
  if (speaking) { window.speechSynthesis.cancel(); speaking = false; }
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
</script>
</body>
</html>"""

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_UI_HTML)


_LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SimpliScan — OCR API for Everyone</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --ink: #0d0d0d;
  --ink2: #3a3a3a;
  --muted: #888;
  --bg: #fafaf8;
  --surface: #ffffff;
  --border: #e8e4dc;
  --accent: #1a6b4a;
  --accent2: #e85d2f;
  --accent-light: #e8f5ee;
  --radius: 16px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--ink);
  font-family: 'DM Sans', sans-serif;
  font-size: 16px;
  line-height: 1.6;
  overflow-x: hidden;
}

/* NAV */
nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  display: flex; align-items: center; padding: 18px 60px;
  background: rgba(250,250,248,0.92); backdrop-filter: blur(12px);
  border-bottom: 1px solid var(--border);
}
.nav-logo { font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: var(--ink); text-decoration: none; }
.nav-logo span { color: var(--accent); }
.nav-links { display: flex; gap: 32px; margin: 0 auto; }
.nav-links a { font-size: 14px; font-weight: 500; color: var(--ink2); text-decoration: none; transition: color .15s; }
.nav-links a:hover { color: var(--accent); }
.nav-cta {
  background: var(--ink); color: #fff; padding: 10px 22px;
  border-radius: 8px; font-size: 14px; font-weight: 500; text-decoration: none;
  transition: background .15s;
}
.nav-cta:hover { background: var(--accent); }

/* HERO */
.hero {
  padding: 160px 60px 100px;
  display: grid; grid-template-columns: 1fr 1fr; gap: 80px; align-items: center;
  max-width: 1200px; margin: 0 auto;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--accent-light); color: var(--accent);
  padding: 6px 14px; border-radius: 100px; font-size: 13px; font-weight: 500;
  margin-bottom: 24px; border: 1px solid #b8dfc9;
}
.hero-badge-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }
.hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 58px; font-weight: 800; line-height: 1.08;
  letter-spacing: -1.5px; margin-bottom: 24px;
}
.hero h1 em { font-style: normal; color: var(--accent); }
.hero p {
  font-size: 18px; color: var(--ink2); line-height: 1.7;
  margin-bottom: 40px; max-width: 480px;
}
.hero-btns { display: flex; gap: 14px; flex-wrap: wrap; }
.btn-primary {
  background: var(--accent); color: #fff;
  padding: 14px 28px; border-radius: 10px;
  font-size: 16px; font-weight: 500; text-decoration: none;
  transition: all .15s; display: inline-flex; align-items: center; gap: 8px;
}
.btn-primary:hover { background: #14543a; transform: translateY(-1px); box-shadow: 0 8px 24px rgba(26,107,74,0.25); }
.btn-secondary {
  background: var(--surface); color: var(--ink);
  padding: 14px 28px; border-radius: 10px; border: 1.5px solid var(--border);
  font-size: 16px; font-weight: 500; text-decoration: none;
  transition: all .15s; display: inline-flex; align-items: center; gap: 8px;
}
.btn-secondary:hover { border-color: var(--ink); transform: translateY(-1px); }

/* Demo window */
.demo-window {
  background: var(--surface); border-radius: var(--radius);
  border: 1.5px solid var(--border); overflow: hidden;
  box-shadow: 0 20px 60px rgba(0,0,0,0.08);
}
.demo-bar {
  background: #f0ede6; padding: 12px 16px;
  display: flex; align-items: center; gap: 8px;
  border-bottom: 1px solid var(--border);
}
.demo-dot { width: 10px; height: 10px; border-radius: 50%; }
.demo-body { padding: 24px; font-family: 'DM Sans', monospace; font-size: 13px; }
.demo-label { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 8px; }
.demo-img-placeholder {
  background: #f5f2eb; border-radius: 8px;
  height: 80px; margin-bottom: 16px;
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 13px; border: 1.5px dashed var(--border);
}
.demo-arrow { text-align: center; color: var(--accent); font-size: 18px; margin: 8px 0; }
.demo-output {
  background: var(--accent-light); border-radius: 8px;
  padding: 14px 16px; border: 1px solid #b8dfc9;
}
.demo-output-text { font-size: 13px; line-height: 1.8; color: var(--ink); }
.demo-tags { display: flex; gap: 6px; margin-top: 10px; flex-wrap: wrap; }
.demo-tag {
  background: #fff; border: 1px solid #b8dfc9;
  border-radius: 6px; padding: 3px 10px; font-size: 11px;
  color: var(--accent); font-weight: 500;
}

/* STATS */
.stats-bar {
  background: var(--ink); color: #fff;
  display: grid; grid-template-columns: repeat(4, 1fr);
  max-width: 1200px; margin: 0 auto 80px; border-radius: var(--radius);
}
.stat { padding: 40px 32px; border-right: 1px solid rgba(255,255,255,0.1); }
.stat:last-child { border-right: none; }
.stat-num { font-family: 'Syne', sans-serif; font-size: 42px; font-weight: 800; line-height: 1; margin-bottom: 8px; }
.stat-num span { color: var(--accent2); }
.stat-label { font-size: 14px; color: rgba(255,255,255,0.6); }

/* FEATURES */
.section { max-width: 1200px; margin: 0 auto 100px; padding: 0 60px; }
.section-header { margin-bottom: 56px; }
.section-eyebrow {
  font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .12em; color: var(--accent); margin-bottom: 16px;
}
.section-title {
  font-family: 'Syne', sans-serif; font-size: 42px;
  font-weight: 800; line-height: 1.15; letter-spacing: -1px;
}
.section-sub { font-size: 18px; color: var(--ink2); margin-top: 12px; max-width: 560px; }

.features-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
.feature-card {
  background: var(--surface); border-radius: var(--radius);
  border: 1.5px solid var(--border); padding: 32px 28px;
  transition: all .2s;
}
.feature-card:hover { border-color: var(--accent); transform: translateY(-3px); box-shadow: 0 12px 32px rgba(0,0,0,0.06); }
.feature-icon {
  width: 48px; height: 48px; border-radius: 12px;
  background: var(--accent-light); display: flex; align-items: center;
  justify-content: center; font-size: 22px; margin-bottom: 20px;
}
.feature-card h3 { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; margin-bottom: 10px; }
.feature-card p { font-size: 14px; color: var(--ink2); line-height: 1.7; }

/* HOW IT WORKS */
.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; position: relative; }
.steps::before {
  content: ''; position: absolute;
  top: 28px; left: 10%; right: 10%;
  height: 1.5px; background: var(--border);
}
.step { text-align: center; padding: 0 16px; }
.step-num {
  width: 56px; height: 56px; border-radius: 50%;
  background: var(--surface); border: 2px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 800;
  margin: 0 auto 20px; position: relative; z-index: 1; color: var(--accent);
}
.step h3 { font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 700; margin-bottom: 8px; }
.step p { font-size: 13px; color: var(--ink2); line-height: 1.6; }

/* LANGUAGES */
.lang-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.lang-pill {
  background: var(--surface); border: 1.5px solid var(--border);
  border-radius: 100px; padding: 8px 20px; font-size: 14px; font-weight: 500;
  display: flex; align-items: center; gap: 8px; transition: all .15s;
}
.lang-pill:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-light); }
.lang-flag { font-size: 18px; }

/* USE CASES */
.use-cases { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.use-case {
  background: var(--surface); border-radius: var(--radius);
  border: 1.5px solid var(--border); padding: 32px;
  display: flex; gap: 20px; align-items: flex-start;
  transition: all .2s;
}
.use-case:hover { border-color: var(--accent); box-shadow: 0 8px 24px rgba(0,0,0,0.05); }
.use-case-icon {
  font-size: 32px; flex-shrink: 0;
  width: 56px; height: 56px; background: var(--accent-light);
  border-radius: 12px; display: flex; align-items: center; justify-content: center;
}
.use-case h3 { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 700; margin-bottom: 8px; }
.use-case p { font-size: 14px; color: var(--ink2); line-height: 1.7; }

/* CTA */
.cta-section {
  background: var(--ink); border-radius: 24px;
  max-width: 1200px; margin: 0 auto 100px; padding: 80px 60px;
  display: grid; grid-template-columns: 1fr auto; gap: 60px; align-items: center;
  position: relative; overflow: hidden;
}
.cta-section::before {
  content: ''; position: absolute;
  width: 400px; height: 400px; border-radius: 50%;
  background: rgba(26,107,74,0.2); top: -100px; right: -100px;
}
.cta-section::after {
  content: ''; position: absolute;
  width: 200px; height: 200px; border-radius: 50%;
  background: rgba(232,93,47,0.15); bottom: -60px; left: 200px;
}
.cta-section h2 {
  font-family: 'Syne', sans-serif; font-size: 46px;
  font-weight: 800; color: #fff; line-height: 1.1;
  letter-spacing: -1px; position: relative; z-index: 1;
}
.cta-section h2 em { font-style: normal; color: #6ee7b7; }
.cta-section p { color: rgba(255,255,255,0.65); font-size: 16px; margin-top: 14px; position: relative; z-index: 1; }
.cta-btns { display: flex; flex-direction: column; gap: 12px; position: relative; z-index: 1; }
.cta-btn-main {
  background: var(--accent2); color: #fff;
  padding: 16px 32px; border-radius: 10px;
  font-size: 16px; font-weight: 500; text-decoration: none;
  text-align: center; white-space: nowrap; transition: all .15s;
}
.cta-btn-main:hover { background: #c94d24; }
.cta-btn-docs {
  background: rgba(255,255,255,0.1); color: #fff;
  padding: 14px 32px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2);
  font-size: 15px; font-weight: 500; text-decoration: none;
  text-align: center; transition: all .15s;
}
.cta-btn-docs:hover { background: rgba(255,255,255,0.2); }

/* FOOTER */
footer {
  border-top: 1px solid var(--border);
  padding: 40px 60px; max-width: 1200px; margin: 0 auto;
  display: flex; align-items: center; justify-content: space-between;
}
.footer-logo { font-family: 'Syne', sans-serif; font-size: 18px; font-weight: 800; }
.footer-logo span { color: var(--accent); }
.footer-links { display: flex; gap: 24px; }
.footer-links a { font-size: 13px; color: var(--muted); text-decoration: none; }
.footer-links a:hover { color: var(--ink); }

/* ANIMATIONS */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp .6s ease forwards; }
.fade-up-2 { animation: fadeUp .6s .1s ease forwards; opacity: 0; }
.fade-up-3 { animation: fadeUp .6s .2s ease forwards; opacity: 0; }
.fade-up-4 { animation: fadeUp .6s .3s ease forwards; opacity: 0; }

@media (max-width: 900px) {
  nav { padding: 16px 24px; }
  .nav-links { display: none; }
  .hero { grid-template-columns: 1fr; padding: 120px 24px 60px; gap: 48px; }
  .hero h1 { font-size: 40px; }
  .stats-bar { grid-template-columns: repeat(2,1fr); margin: 0 24px 60px; }
  .section { padding: 0 24px; }
  .features-grid { grid-template-columns: 1fr; }
  .steps { grid-template-columns: repeat(2,1fr); }
  .steps::before { display: none; }
  .use-cases { grid-template-columns: 1fr; }
  .cta-section { grid-template-columns: 1fr; padding: 48px 32px; margin: 0 24px 60px; }
  footer { padding: 32px 24px; flex-direction: column; gap: 16px; }
}
</style>
</head>
<body>

<nav>
  <a href="#" class="nav-logo">Simpli<span>Scan</span></a>
  <div class="nav-links">
    <a href="#features">Features</a>
    <a href="#how">How it works</a>
    <a href="#languages">Languages</a>
    <a href="#usecases">Use cases</a>
  </div>
  <a href="https://ocr-service-4e7i.onrender.com" class="nav-cta" target="_blank">Try it free →</a>
</nav>

<!-- HERO -->
<section class="hero">
  <div>
    <div class="hero-badge fade-up">
      <span class="hero-badge-dot"></span>
      Now with Claude AI correction
    </div>
    <h1 class="fade-up-2">OCR that actually <em>understands</em> your documents</h1>
    <p class="fade-up-3">Extract, simplify, and translate text from any image — in Hebrew, English, Arabic, and more. Built for seniors, immigrants, and anyone facing a complex document.</p>
    <div class="hero-btns fade-up-4">
      <a href="https://ocr-service-4e7i.onrender.com" class="btn-primary" target="_blank">🚀 Try it live</a>
      <a href="https://ocr-service-4e7i.onrender.com/docs" class="btn-secondary" target="_blank">📖 API docs</a>
    </div>
  </div>
  <div class="demo-window fade-up-3">
    <div class="demo-bar">
      <div class="demo-dot" style="background:#ff5f56"></div>
      <div class="demo-dot" style="background:#ffbd2e"></div>
      <div class="demo-dot" style="background:#27c93f"></div>
      <span style="margin-right:auto;margin-left:12px;font-size:12px;color:#888">simpliscan.app</span>
    </div>
    <div class="demo-body">
      <div class="demo-label">Input — government letter (image)</div>
      <div class="demo-img-placeholder">📄 מכתב מביטוח לאומי — תמונה</div>
      <div class="demo-arrow">↓ OCR + AI simplification</div>
      <div class="demo-output">
        <div class="demo-label">Output — simplified text</div>
        <div class="demo-output-text">
          קיבלת הודעה מביטוח לאומי.<br>
          עליך לשלם <strong>₪340</strong> עד לתאריך <strong>15.04.2026</strong>.<br>
          לתשלום — לחץ כאן או התקשר: <strong>*6050</strong>
        </div>
        <div class="demo-tags">
          <span class="demo-tag">✓ Hebrew OCR</span>
          <span class="demo-tag">✓ AI simplified</span>
          <span class="demo-tag">✓ Key dates extracted</span>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- STATS -->
<div style="max-width:1200px;margin:0 auto 80px;padding:0 60px;">
  <div class="stats-bar">
    <div class="stat">
      <div class="stat-num">99<span>%</span></div>
      <div class="stat-label">Hebrew + English accuracy</div>
    </div>
    <div class="stat">
      <div class="stat-num">8<span>+</span></div>
      <div class="stat-label">Languages supported</div>
    </div>
    <div class="stat">
      <div class="stat-num">3<span>s</span></div>
      <div class="stat-label">Average processing time</div>
    </div>
    <div class="stat">
      <div class="stat-num">$0</div>
      <div class="stat-label">Free to use — no signup</div>
    </div>
  </div>
</div>

<!-- FEATURES -->
<section class="section" id="features">
  <div class="section-header">
    <div class="section-eyebrow">Features</div>
    <h2 class="section-title">Everything you need to read any document</h2>
    <p class="section-sub">From raw scanning to AI explanation — one platform handles it all.</p>
  </div>
  <div class="features-grid">
    <div class="feature-card">
      <div class="feature-icon">📖</div>
      <h3>Smart OCR</h3>
      <p>Advanced text recognition using Tesseract with custom Hebrew preprocessing. Handles skewed, low-res, and noisy scans.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🤖</div>
      <h3>AI correction</h3>
      <p>Claude AI automatically fixes garbled characters, fills in missing words, and reconstructs damaged text from context.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🧒</div>
      <h3>Explain like I'm 12</h3>
      <p>Complex legal, medical, or bureaucratic language translated into plain, simple sentences anyone can understand.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🌐</div>
      <h3>Instant translation</h3>
      <p>Translate extracted text into Hebrew, English, Arabic, Russian, French, Spanish, Amharic, and Tigrinya.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">🔊</div>
      <h3>Text-to-speech</h3>
      <p>Listen to your document read aloud. Auto-detects language and uses the correct voice for Hebrew or English.</p>
    </div>
    <div class="feature-card">
      <div class="feature-icon">📦</div>
      <h3>Batch processing</h3>
      <p>Upload up to 20 images at once. Process entire document folders in a single API call or drag-and-drop session.</p>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section class="section" id="how">
  <div class="section-header">
    <div class="section-eyebrow">How it works</div>
    <h2 class="section-title">From image to understanding in seconds</h2>
  </div>
  <div class="steps">
    <div class="step">
      <div class="step-num">1</div>
      <h3>Upload</h3>
      <p>Drop an image, paste a URL, or send via API. Supports JPEG, PNG, WEBP, TIFF, BMP.</p>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <h3>Extract</h3>
      <p>Tesseract OCR reads the text. Hebrew and English are processed simultaneously.</p>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <h3>Enhance</h3>
      <p>Claude AI fixes errors, simplifies language, and optionally translates — all automatically.</p>
    </div>
    <div class="step">
      <div class="step-num">4</div>
      <h3>Use</h3>
      <p>Copy the text, listen via TTS, or receive clean JSON via the REST API.</p>
    </div>
  </div>
</section>

<!-- LANGUAGES -->
<section class="section" id="languages">
  <div class="section-header">
    <div class="section-eyebrow">Languages</div>
    <h2 class="section-title">Built for a multilingual world</h2>
    <p class="section-sub">Native support for Hebrew and English OCR, with AI translation into 8 languages.</p>
  </div>
  <div class="lang-grid">
    <div class="lang-pill"><span class="lang-flag">🇮🇱</span> עברית — OCR + Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇺🇸</span> English — OCR + Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇸🇦</span> عربي — Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇷🇺</span> Русский — Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇫🇷</span> Français — Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇪🇸</span> Español — Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇪🇹</span> አማርኛ — Translation</div>
    <div class="lang-pill"><span class="lang-flag">🇪🇷</span> ትግርኛ — Translation</div>
  </div>
</section>

<!-- USE CASES -->
<section class="section" id="usecases">
  <div class="section-header">
    <div class="section-eyebrow">Use cases</div>
    <h2 class="section-title">Who it's built for</h2>
  </div>
  <div class="use-cases">
    <div class="use-case">
      <div class="use-case-icon">👴</div>
      <div>
        <h3>Senior citizens</h3>
        <p>Scan a letter from Bituach Leumi, get a plain-language explanation read aloud. No tech skills required.</p>
      </div>
    </div>
    <div class="use-case">
      <div class="use-case-icon">✈️</div>
      <div>
        <h3>New immigrants</h3>
        <p>Understand official Israeli documents in your native language — Arabic, Russian, Amharic, or English.</p>
      </div>
    </div>
    <div class="use-case">
      <div class="use-case-icon">⚖️</div>
      <div>
        <h3>Legal & bureaucratic documents</h3>
        <p>Dense government letters, rental contracts, or medical forms simplified into plain, actionable language.</p>
      </div>
    </div>
    <div class="use-case">
      <div class="use-case-icon">🏥</div>
      <div>
        <h3>Medical documents</h3>
        <p>Extract and explain diagnoses, prescriptions, and discharge letters in simple terms patients can understand.</p>
      </div>
    </div>
    <div class="use-case">
      <div class="use-case-icon">🎓</div>
      <div>
        <h3>Academic articles</h3>
        <p>Scan research papers and get a plain-language summary — perfect for students or curious readers.</p>
      </div>
    </div>
    <div class="use-case">
      <div class="use-case-icon">🏢</div>
      <div>
        <h3>Developers & businesses</h3>
        <p>REST API with JSON output. Integrate document understanding into any app in minutes.</p>
      </div>
    </div>
  </div>
</section>

<!-- CTA -->
<div style="padding:0 60px;">
  <div class="cta-section">
    <div>
      <h2>Ready to make documents <em>accessible</em> to everyone?</h2>
      <p>Free to use. No account required. Deployed and ready now.</p>
    </div>
    <div class="cta-btns">
      <a href="https://ocr-service-4e7i.onrender.com" class="cta-btn-main" target="_blank">Open the app →</a>
      <a href="https://ocr-service-4e7i.onrender.com/docs" class="cta-btn-docs" target="_blank">View API docs</a>
    </div>
  </div>
</div>

<footer>
  <div class="footer-logo">Simpli<span>Scan</span></div>
  <div class="footer-links">
    <a href="https://ocr-service-4e7i.onrender.com">App</a>
    <a href="https://ocr-service-4e7i.onrender.com/docs">API</a>
    <a href="https://github.com/gilbartor-glitch/ocr-service">GitHub</a>
  </div>
</footer>

</body>
</html>
"""

@app.get("/landing", include_in_schema=False, response_class=HTMLResponse)
async def serve_landing():
    return HTMLResponse(_LANDING)
