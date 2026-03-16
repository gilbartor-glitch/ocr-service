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
