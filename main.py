from __future__ import annotations
import asyncio, logging, re, cv2, httpx, numpy as np
from contextlib import asynccontextmanager
from typing import Annotated, Literal
import pytesseract, os
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, HttpUrl

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("ocr_service")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_reader = None

@asynccontextmanager
async def lifespan(app):
    log.info("Tesseract OCR ready (en+he).")
    yield

app = FastAPI(title="OCR Service", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALLOWED = {"image/jpeg","image/png","image/webp","image/tiff","image/bmp"}
MAX_BYTES = 20 * 1024 * 1024


async def _ai_enhance(text):
    log.info(f"AI enhance called. Key set: {bool(ANTHROPIC_API_KEY)}, Text length: {len(text)}")
    if not ANTHROPIC_API_KEY or not text.strip():
        log.warning("AI skipped: key empty or text empty")
        return text
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY,
                         "anthropic-version": "2023-06-01",
                         "content-type": "application/json"},
                json={"model": "claude-sonnet-4-6", "max_tokens": 2048,
                      "messages": [{"role": "user", "content":
                        f"You are an OCR correction assistant. Fix OCR errors, wrong letters, garbled words, and fill missing words based on context. Preserve original language (Hebrew/English). Preserve line breaks. Output ONLY the corrected text.\n\n{text}"}]}
            )
        if resp.status_code == 200:
            enhanced = resp.json()["content"][0]["text"].strip()
            log.info("AI enhancement applied.")
            return enhanced
        log.warning(f"AI skipped: {resp.status_code} {resp.text[:100]}")
        return text
    except Exception as e:
        log.warning(f"AI failed: {e}")
        return text

def _clean_text(raw):
    lines = raw.splitlines()
    cleaned = [re.sub(r"[^\S\n]+"," ",l).strip() for l in lines]
    return re.sub(r"\n{3,}","\n\n","\n".join(cleaned)).strip()

def _to_structured(text):
    lines = [l for l in text.splitlines() if l.strip()]
    kv = {}
    for line in lines:
        m = re.match(r"^([^:\n]{1,60}):\s*(.+)$", line)
        if m: kv[m.group(1).strip()] = m.group(2).strip()
    words = text.split()
    return {"lines": lines, "key_value_pairs": kv,
            "stats": {"line_count": len(lines), "word_count": len(words), "char_count": len(text)}}

def _preprocess(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: raise ValueError("Could not decode image.")
    # Upscale small images
    h, w = img.shape[:2]
    if max(h, w) < 1500:
        scale = 2.0 if max(h, w) < 800 else 1.5
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # Convert to grayscale — let each engine do its own binarization
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Light denoise only
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    return gray

def _is_hebrew(text):
    he_chars = sum(1 for c in text if "\u05d0" <= c <= "\u05ea")
    return he_chars > max(3, len(text) * 0.2)

def _run_ocr(image):
    # English via EasyOCR
    en_lines = get_reader().readtext(image, detail=0, paragraph=True,
                                     min_size=10, text_threshold=0.6, low_text=0.3)
    en_text = "\n".join(en_lines)

    # Hebrew via Tesseract
    he_text = ""
    try:
        best = ""
        for psm in [6, 3, 4]:
            r = pytesseract.image_to_string(
                image, lang="heb+eng",
                config=f"--psm {psm} --oem 1"
            )
            if len(r.strip()) > len(best.strip()):
                best = r
        he_text = best.strip()
    except Exception as e:
        log.warning(f"Tesseract failed: {e}")

    # Count Hebrew chars in each result
    he_in_tess = sum(1 for c in he_text if "\u05d0" <= c <= "\u05ea")
    he_in_easy = sum(1 for c in en_text if "\u05d0" <= c <= "\u05ea")

    if he_in_tess > 10 and he_in_tess >= he_in_easy:
        # Primarily Hebrew document — use Tesseract
        return he_text
    elif he_in_tess > 5 and len(en_text) > 20:
        # Mixed — combine: Hebrew lines first, then English
        he_lines = [l.strip() for l in he_text.splitlines() if l.strip()]
        en_only = [l for l in en_lines if not _is_hebrew(l)]
        all_lines = he_lines + en_only
        return "\n".join(all_lines)
    else:
        return en_text

async def _ocr_from_bytes(data):
    loop = asyncio.get_event_loop()
    gray = await loop.run_in_executor(None, _preprocess, data)
    raw = await loop.run_in_executor(None, _run_ocr, gray)
    cleaned = _clean_text(raw)
    enhanced = await _ai_enhance(cleaned)
    return enhanced

class TextResult(BaseModel):
    filename: str
    text: str
    structured: dict | None = None

class BatchResponse(BaseModel):
    results: list[TextResult]
    total: int

class UrlRequest(BaseModel):
    url: HttpUrl

OutputMode = Literal["text", "structured"]

def _build(filename, text, mode):
    return TextResult(filename=filename, text=text,
                      structured=_to_structured(text) if mode=="structured" else None)

@app.get("/health", tags=["meta"])
async def health():
    return {"status":"ok","model":"tesseract","languages":["en","he"],"version":"2.0.0"}

@app.post("/ocr/upload", response_model=TextResult, tags=["ocr"])
async def ocr_single(file: Annotated[UploadFile, File()], mode: OutputMode = Query("text")):
    if file.content_type not in ALLOWED: raise HTTPException(415, f"Unsupported: {file.content_type}")
    data = await file.read()
    if len(data) > MAX_BYTES: raise HTTPException(413, "File too large.")
    try: text = await _ocr_from_bytes(data)
    except Exception as e: raise HTTPException(500, str(e))
    return _build(file.filename or "upload", text, mode)

@app.post("/ocr/batch", response_model=BatchResponse, tags=["ocr"])
async def ocr_batch(files: Annotated[list[UploadFile], File()], mode: OutputMode = Query("text")):
    if not files: raise HTTPException(400, "No files.")
    if len(files) > 20: raise HTTPException(400, "Max 20 files.")
    async def process(f):
        if f.content_type not in ALLOWED:
            return TextResult(filename=f.filename or "?", text=f"[SKIPPED] {f.content_type}")
        data = await f.read()
        if len(data) > MAX_BYTES:
            return TextResult(filename=f.filename or "?", text="[SKIPPED] Too large.")
        try: text = await _ocr_from_bytes(data)
        except Exception as e: return TextResult(filename=f.filename or "?", text=f"[ERROR] {e}")
        return _build(f.filename or "?", text, mode)
    results = []
    for f in files: results.append(await process(f))
    return BatchResponse(results=results, total=len(results))

@app.post("/ocr/url", response_model=TextResult, tags=["ocr"])
async def ocr_url(body: UrlRequest, mode: OutputMode = Query("text")):
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
            r = await c.get(str(body.url)); r.raise_for_status()
    except httpx.HTTPStatusError as e: raise HTTPException(502, f"Remote error {e.response.status_code}")
    except httpx.RequestError as e: raise HTTPException(502, str(e))
    if len(r.content) > MAX_BYTES: raise HTTPException(413, "Too large.")
    first_bytes = r.content[:20].lower()
    if b"<html" in first_bytes or b"<!doc" in first_bytes:
        raise HTTPException(422, "URL returned an HTML page, not an image. Right-click the image and choose Open image in new tab, then copy that URL.")
    try: text = await _ocr_from_bytes(r.content)
    except Exception as e: raise HTTPException(422, f"Could not decode image. Make sure the URL points directly to an image file. Detail: {str(e)}")
    return _build(str(body.url).split("/")[-1] or "remote", text, mode)

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OCR Service</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:        #0a0a0a;
    --bg2:       #111111;
    --bg3:       #1a1a1a;
    --border:    #2a2a2a;
    --amber:     #f5a623;
    --amber-dim: #7a5312;
    --green:     #39d353;
    --red:       #ff4d4d;
    --text:      #e8e8e8;
    --text-dim:  #666;
    --radius:    4px;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr auto;
  }

  /* ── header ── */
  header {
    border-bottom: 1px solid var(--border);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .logo {
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--amber);
  }
  .logo-sep { color: var(--border); margin: 0 4px; }
  .logo-version {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    font-weight: 300;
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--text-dim);
    margin-left: auto;
    transition: background .3s;
  }
  .status-dot.online { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: .1em;
  }

  /* ── main layout ── */
  main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    height: calc(100vh - 57px - 41px);
  }

  /* ── left panel ── */
  .panel-left {
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .panel-section {
    border-bottom: 1px solid var(--border);
    padding: 20px 24px;
  }
  .panel-section:last-child { border-bottom: none; flex: 1; display: flex; flex-direction: column; }

  .section-label {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 14px;
  }

  /* mode tabs */
  .tabs {
    display: flex;
    gap: 2px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 3px;
  }
  .tab {
    flex: 1;
    padding: 7px 12px;
    font-family: var(--mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .08em;
    text-align: center;
    cursor: pointer;
    border-radius: 2px;
    color: var(--text-dim);
    transition: all .15s;
    border: none;
    background: transparent;
    user-select: none;
  }
  .tab:hover { color: var(--text); }
  .tab.active { background: var(--bg3); color: var(--amber); }

  /* options row */
  .options-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }
  .option-chip {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all .15s;
    background: transparent;
    color: var(--text-dim);
    user-select: none;
  }
  .option-chip:hover { border-color: var(--amber-dim); color: var(--text); }
  .option-chip.selected { border-color: var(--amber); color: var(--amber); background: rgba(245,166,35,.06); }
  .option-chip .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    border: 1px solid currentColor;
    transition: background .15s;
  }
  .option-chip.selected .dot { background: var(--amber); }

  /* url input */
  .url-input-wrap { display: flex; gap: 8px; }
  .url-input {
    flex: 1;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 9px 12px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    outline: none;
    transition: border-color .15s;
  }
  .url-input:focus { border-color: var(--amber); }
  .url-input::placeholder { color: var(--text-dim); }

  /* drop zone */
  .drop-zone {
    flex: 1;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    cursor: pointer;
    transition: all .2s;
    position: relative;
    overflow: hidden;
    min-height: 160px;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: var(--amber);
    background: rgba(245,166,35,.03);
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0;
    opacity: 0; cursor: pointer;
    width: 100%; height: 100%;
  }
  .drop-icon {
    font-size: 28px;
    opacity: .35;
    pointer-events: none;
  }
  .drop-text {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text-dim);
    text-align: center;
    pointer-events: none;
    line-height: 1.8;
  }
  .drop-text strong { color: var(--amber); font-weight: 500; }

  /* preview thumbnails */
  .thumb-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
  }
  .thumb {
    width: 56px; height: 56px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    position: relative;
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb-count {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    padding: 4px 0;
  }

  /* run button */
  .run-btn {
    width: 100%;
    padding: 12px;
    background: var(--amber);
    color: #000;
    border: none;
    border-radius: var(--radius);
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .12em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all .15s;
    margin-top: 14px;
  }
  .run-btn:hover { background: #ffb93e; }
  .run-btn:disabled {
    background: var(--bg3);
    color: var(--text-dim);
    cursor: not-allowed;
  }
  .run-btn.loading {
    background: var(--amber-dim);
    color: var(--amber);
    cursor: wait;
  }

  /* ── right panel ── */
  .panel-right {
    display: flex;
    flex-direction: column;
  }

  .output-header {
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .output-title {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--text-dim);
  }
  .output-meta {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    margin-left: auto;
  }
  .copy-btn {
    background: transparent;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 5px 12px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    cursor: pointer;
    letter-spacing: .08em;
    transition: all .15s;
  }
  .copy-btn:hover { border-color: var(--amber); color: var(--amber); }

  /* output view tabs */
  .view-tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
  }
  .view-tab {
    padding: 10px 20px;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all .15s;
    border: none;
    background: transparent;
    margin-bottom: -1px;
  }
  .view-tab:hover { color: var(--text); }
  .view-tab.active { color: var(--amber); border-bottom: 2px solid var(--amber); }

  /* output content */
  .output-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
  }
  .output-body::-webkit-scrollbar { width: 4px; }
  .output-body::-webkit-scrollbar-track { background: transparent; }
  .output-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .output-text {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 300;
    line-height: 1.8;
    white-space: pre-wrap;
    color: var(--text);
  }

  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 10px;
    color: var(--text-dim);
  }
  .placeholder-icon { font-size: 36px; opacity: .2; }
  .placeholder-text {
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: .1em;
    text-transform: uppercase;
  }

  /* structured view */
  .struct-section { margin-bottom: 24px; }
  .struct-section-title {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }
  .struct-line {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--text);
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,.03);
    line-height: 1.6;
  }
  .kv-row {
    display: grid;
    grid-template-columns: 200px 1fr;
    gap: 12px;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,.03);
  }
  .kv-key { font-family: var(--mono); font-size: 11px; color: var(--amber); font-weight: 500; }
  .kv-val { font-family: var(--mono); font-size: 12px; color: var(--text); }

  .stats-row {
    display: flex;
    gap: 20px;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 12px 16px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    flex: 1;
  }
  .stat-val { font-family: var(--mono); font-size: 22px; font-weight: 500; color: var(--amber); }
  .stat-lbl { font-family: var(--mono); font-size: 10px; color: var(--text-dim); letter-spacing: .1em; text-transform: uppercase; }

  /* multi-result tabs */
  .result-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    padding: 0 24px 14px;
    border-bottom: 1px solid var(--border);
  }
  .result-tab {
    font-family: var(--mono);
    font-size: 10px;
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: 2px;
    cursor: pointer;
    color: var(--text-dim);
    background: transparent;
    transition: all .15s;
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .result-tab:hover { border-color: var(--amber-dim); color: var(--text); }
  .result-tab.active { border-color: var(--amber); color: var(--amber); background: rgba(245,166,35,.06); }

  /* error */
  .error-box {
    background: rgba(255,77,77,.07);
    border: 1px solid rgba(255,77,77,.3);
    border-radius: var(--radius);
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--red);
    line-height: 1.6;
  }

  /* spinner */
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid var(--amber-dim);
    border-top-color: var(--amber);
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle;
    margin-right: 7px;
  }

  /* footer */
  footer {
    border-top: 1px solid var(--border);
    padding: 10px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .footer-text {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    letter-spacing: .08em;
  }
  .footer-link {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-dim);
    text-decoration: none;
    letter-spacing: .08em;
    margin-left: auto;
    transition: color .15s;
  }
  .footer-link:hover { color: var(--amber); }

  /* RTL text detection */
  .rtl { direction: rtl; text-align: right; }

  @media (max-width: 768px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto 1fr; height: auto; }
    .panel-left { border-right: none; border-bottom: 1px solid var(--border); }
    header { padding: 14px 16px; }
    .panel-section { padding: 16px; }
  }
</style>
</head>
<body>

<header>
  <span class="logo">OCR<span class="logo-sep">/</span>Service</span>
  <span class="logo-version">v2.0.0 · en + he</span>
  <span class="status-label" id="statusLabel">CONNECTING</span>
  <span class="status-dot" id="statusDot"></span>
</header>

<main>
  <!-- ── LEFT PANEL ── -->
  <div class="panel-left">

    <!-- Input mode -->
    <div class="panel-section">
      <div class="section-label">Input mode</div>
      <div class="tabs">
        <button class="tab active" data-mode="upload" onclick="switchInputMode('upload')">Upload</button>
        <button class="tab" data-mode="batch" onclick="switchInputMode('batch')">Batch</button>
        <button class="tab" data-mode="url" onclick="switchInputMode('url')">URL</button>
      </div>
    </div>

    <!-- Output options -->
    <div class="panel-section">
      <div class="section-label">Output format</div>
      <div class="options-row">
        <button class="option-chip selected" data-out="text" onclick="selectOutput('text')">
          <span class="dot"></span>Raw text
        </button>
        <button class="option-chip" data-out="structured" onclick="selectOutput('structured')">
          <span class="dot"></span>Structured
        </button>
      </div>
    </div>

    <!-- Drop zone / URL input -->
    <div class="panel-section" id="dropSection">
      <div class="section-label">Image</div>
      <div class="drop-zone" id="dropZone"
           ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
        <input type="file" id="fileInput" accept="image/*"
               onchange="onFileSelect(event)"/>
        <div class="drop-icon">⌗</div>
        <div class="drop-text">
          <strong>Drop image here</strong><br/>
          or click to browse<br/>
          JPEG · PNG · WEBP · TIFF · BMP
        </div>
      </div>
      <div class="thumb-row" id="thumbRow"></div>
    </div>

    <div class="panel-section" id="urlSection" style="display:none">
      <div class="section-label">Image URL</div>
      <div class="url-input-wrap">
        <input class="url-input" id="urlInput" type="url"
               placeholder="https://example.com/image.jpg"/>
      </div>
    </div>

    <!-- Run -->
    <div style="padding: 0 24px 24px;">
      <button class="run-btn" id="runBtn" onclick="runOCR()">Run OCR</button><div style=\"display:flex;gap:8px;margin-top:8px\"><button class=\"run-btn\" style=\"flex:1;background:transparent;color:var(--text-dim);border:1px solid var(--border)\" onclick=\"resetAll()\">Reset</button><button class=\"run-btn\" style=\"flex:1;background:transparent;color:var(--text-dim);border:1px solid var(--border)\" onclick=\"location.reload()\">Refresh</button></div>
    </div>

  </div>

  <!-- ── RIGHT PANEL ── -->
  <div class="panel-right">
    <div class="output-header">
      <span class="output-title">Output</span>
      <span class="output-meta" id="outputMeta"></span>
      <button class="copy-btn" id="copyBtn" onclick="copyOutput()" style="display:none">Copy</button>
    </div>

    <div class="view-tabs" id="viewTabs" style="display:none">
      <button class="view-tab active" data-view="text" onclick="switchView('text')">Text</button>
      <button class="view-tab" data-view="structured" onclick="switchView('structured')">Structured</button>
    </div>

    <div class="result-tabs" id="resultTabs" style="display:none"></div>

    <div class="output-body" id="outputBody">
      <div class="placeholder">
        <div class="placeholder-icon">▤</div>
        <div class="placeholder-text">Awaiting input</div>
      </div>
    </div>
  </div>
</main>

<footer>
  <span class="footer-text">EasyOCR · FastAPI · Docker</span>
  <a class="footer-link" href="/docs" target="_blank">API docs →</a>
</footer>

<script>
// ── State ──────────────────────────────────────────────────────────────────
let inputMode  = 'upload';   // 'upload' | 'batch' | 'url'
let outputMode = 'text';     // 'text'   | 'structured'
let viewMode   = 'text';     // 'text'   | 'structured'
let selectedFiles = [];
let results = [];
let activeResult = 0;

// ── Status check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch('/health');
    if (r.ok) {
      document.getElementById('statusDot').className = 'status-dot online';
      document.getElementById('statusLabel').textContent = 'ONLINE';
    }
  } catch { /* stays CONNECTING */ }
}
checkHealth();

// ── Input mode ────────────────────────────────────────────────────────────
function switchInputMode(mode) {
  inputMode = mode;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.mode === mode));

  const isUrl = mode === 'url';
  document.getElementById('dropSection').style.display  = isUrl ? 'none' : '';
  document.getElementById('urlSection').style.display   = isUrl ? ''     : 'none';

  const fileInput = document.getElementById('fileInput');
  fileInput.multiple = mode === 'batch';

  selectedFiles = [];
  renderThumbs();
}

// ── Output mode ───────────────────────────────────────────────────────────
function selectOutput(mode) {
  outputMode = mode;
  document.querySelectorAll('.option-chip').forEach(c =>
    c.classList.toggle('selected', c.dataset.out === mode));
}

// ── File handling ─────────────────────────────────────────────────────────
function onFileSelect(e) {
  selectedFiles = Array.from(e.target.files);
  renderThumbs();
}
function onDragOver(e)  { e.preventDefault(); document.getElementById('dropZone').classList.add('drag-over'); }
function onDragLeave()  { document.getElementById('dropZone').classList.remove('drag-over'); }
function onDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  selectedFiles = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
  renderThumbs();
}

function renderThumbs() {
  const row = document.getElementById('thumbRow');
  row.innerHTML = '';
  if (!selectedFiles.length) return;
  const show = selectedFiles.slice(0, 8);
  show.forEach(f => {
    const div = document.createElement('div');
    div.className = 'thumb';
    const img = document.createElement('img');
    img.src = URL.createObjectURL(f);
    div.appendChild(img);
    row.appendChild(div);
  });
  if (selectedFiles.length > 8) {
    const extra = document.createElement('div');
    extra.className = 'thumb-count';
    extra.textContent = `+${selectedFiles.length - 8} more`;
    row.appendChild(extra);
  }
}

// ── Run OCR ───────────────────────────────────────────────────────────────
async function runOCR() {
  const btn = document.getElementById('runBtn');
  btn.disabled = true;
  btn.className = 'run-btn loading';
  btn.innerHTML = '<span class="spinner"></span>Processing…';
  setOutputLoading();

  try {
    const mode = outputMode;

    if (inputMode === 'url') {
      const url = document.getElementById('urlInput').value.trim();
      if (!url) throw new Error('Please enter an image URL.');
      const r = await fetch(`/ocr/url?mode=${mode}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url })
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      results = [await r.json()];

    } else if (inputMode === 'batch' && selectedFiles.length) {
      const fd = new FormData();
      selectedFiles.forEach(f => fd.append('files', f));
      const r = await fetch(`/ocr/batch?mode=${mode}`, { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      const data = await r.json();
      results = data.results;

    } else {
      if (!selectedFiles.length) throw new Error('Please select an image file.');
      const fd = new FormData();
      fd.append('file', selectedFiles[0]);
      const r = await fetch(`/ocr/upload?mode=${mode}`, { method: 'POST', body: fd });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || r.statusText); }
      results = [await r.json()];
    }

    activeResult = 0;
    renderResults();

  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.className = 'run-btn';
    btn.textContent = 'Run OCR';
  }
}

// ── Rendering ─────────────────────────────────────────────────────────────
function setOutputLoading() {
  document.getElementById('outputBody').innerHTML =
    `<div class="placeholder"><span class="spinner" style="width:18px;height:18px"></span></div>`;
  document.getElementById('copyBtn').style.display = 'none';
  document.getElementById('outputMeta').textContent = '';
  document.getElementById('viewTabs').style.display  = 'none';
  document.getElementById('resultTabs').style.display = 'none';
}

function showError(msg) {
  document.getElementById('outputBody').innerHTML =
    `<div class="error-box">⚠ ${escHtml(msg)}</div>`;
}

function renderResults() {
  if (!results.length) return;

  // multi-result tabs (batch)
  const tabsEl = document.getElementById('resultTabs');
  if (results.length > 1) {
    tabsEl.style.display = 'flex';
    tabsEl.innerHTML = results.map((r, i) =>
      `<button class="result-tab ${i === activeResult ? 'active' : ''}"
               onclick="selectResult(${i})"
               title="${escHtml(r.filename)}">${escHtml(r.filename)}</button>`
    ).join('');
  } else {
    tabsEl.style.display = 'none';
  }

  const cur = results[activeResult];
  const hasStructured = cur.structured !== null;

  // view tabs
  const vtEl = document.getElementById('viewTabs');
  vtEl.style.display = hasStructured ? 'flex' : 'none';
  if (!hasStructured) viewMode = 'text';

  vtEl.querySelectorAll('.view-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.view === viewMode));

  // stats
  const meta = document.getElementById('outputMeta');
  if (cur.structured) {
    const s = cur.structured.stats;
    meta.textContent = `${s.line_count} lines · ${s.word_count} words · ${s.char_count} chars`;
  } else {
    const words = cur.text.split(/\\s+/).filter(Boolean).length;
    meta.textContent = `${words} words · ${cur.text.length} chars`;
  }

  document.getElementById('copyBtn').style.display = '';

  // render body
  const body = document.getElementById('outputBody');
  if (viewMode === 'structured' && cur.structured) {
    renderStructured(body, cur.structured);
  } else {
    const isRtl = /[\\u0590-\\u05FF]/.test(cur.text);
    body.innerHTML =
      `<pre class="output-text ${isRtl ? 'rtl' : ''}">${escHtml(cur.text)}</pre>`;
  }
}

function renderStructured(body, s) {
  let html = '';

  // stats
  html += `<div class="struct-section">
    <div class="struct-section-title">Stats</div>
    <div class="stats-row">
      <div class="stat"><div class="stat-val">${s.stats.line_count}</div><div class="stat-lbl">Lines</div></div>
      <div class="stat"><div class="stat-val">${s.stats.word_count}</div><div class="stat-lbl">Words</div></div>
      <div class="stat"><div class="stat-val">${s.stats.char_count}</div><div class="stat-lbl">Chars</div></div>
    </div>
  </div>`;

  // key-value
  if (Object.keys(s.key_value_pairs).length) {
    html += `<div class="struct-section">
      <div class="struct-section-title">Key — Value pairs</div>`;
    for (const [k, v] of Object.entries(s.key_value_pairs)) {
      html += `<div class="kv-row">
        <div class="kv-key">${escHtml(k)}</div>
        <div class="kv-val">${escHtml(v)}</div>
      </div>`;
    }
    html += `</div>`;
  }

  // lines
  html += `<div class="struct-section">
    <div class="struct-section-title">Lines (${s.lines.length})</div>`;
  s.lines.forEach((ln, i) => {
    const isRtl = /[\\u0590-\\u05FF]/.test(ln);
    html += `<div class="struct-line ${isRtl ? 'rtl' : ''}">
      <span style="color:var(--text-dim);margin-right:12px;font-size:10px">${String(i+1).padStart(3,'0')}</span>${escHtml(ln)}
    </div>`;
  });
  html += `</div>`;

  body.innerHTML = html;
}

function selectResult(i) {
  activeResult = i;
  document.querySelectorAll('.result-tab').forEach((t, idx) =>
    t.classList.toggle('active', idx === i));
  renderResults();
}

function switchView(v) {
  viewMode = v;
  renderResults();
}

// ── Copy ──────────────────────────────────────────────────────────────────
function copyOutput() {
  if (!results.length) return;
  const cur = results[activeResult];
  navigator.clipboard.writeText(cur.text).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy', 1500);
  });
}

// ── Utils ─────────────────────────────────────────────────────────────────
function resetAll(){selectedFiles=[];results=[];activeResult=0;document.getElementById('fileInput').value='';document.getElementById('urlInput').value='';document.getElementById('thumbRow').innerHTML='';document.getElementById('outputBody').innerHTML='<div class=\"placeholder\"><div class=\"placeholder-icon\">&#9636;</div><div class=\"placeholder-text\">Awaiting input</div></div>';document.getElementById('copyBtn').style.display='none';document.getElementById('outputMeta').textContent='';document.getElementById('viewTabs').style.display='none';document.getElementById('resultTabs').style.display='none';}
function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_UI_HTML)
