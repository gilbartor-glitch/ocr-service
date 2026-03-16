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

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SimpliScan</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0c0c0c;--bg2:#141414;--bg3:#1c1c1c;--border:#2a2a2a;--border2:#383838;--text:#f0f0f0;--text2:#999;--text3:#555;--green:#00c875;--gdim:#003d21;--r:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;font-size:15px;min-height:100vh}
nav{display:flex;align-items:center;padding:0 40px;height:60px;border-bottom:1px solid var(--border);background:var(--bg);position:sticky;top:0;z-index:100}
.logo{font-family:'Syne',sans-serif;font-size:18px;font-weight:800}
.logo em{font-style:normal;color:var(--green)}
.nst{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:12px;color:var(--text2)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--text3)}
.dot.on{background:var(--green);box-shadow:0 0 8px var(--green)}
.nlink{margin-left:20px;font-size:12px;color:var(--text2);text-decoration:none;padding:5px 12px;border:1px solid var(--border2);border-radius:6px}
.nlink:hover{border-color:var(--green);color:var(--green)}
.wrap{display:grid;grid-template-columns:400px 1fr;min-height:calc(100vh - 60px)}
.left{border-right:1px solid var(--border);display:flex;flex-direction:column;background:var(--bg2)}
.sec{border-bottom:1px solid var(--border);padding:20px 24px}
.sec.grow{border-bottom:none;flex:1;display:flex;flex-direction:column}
.lbl{font-family:'Syne',sans-serif;font-size:10px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--text3);margin-bottom:14px}
.tabs{display:flex;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:3px;gap:2px}
.tab{flex:1;padding:8px;text-align:center;font-size:12px;font-weight:500;color:var(--text2);cursor:pointer;border-radius:6px;border:none;background:transparent;transition:all .15s}
.tab.active{background:var(--bg3);color:var(--green);border:1px solid var(--border2)}
.dz{flex:1;border:1px dashed var(--border2);border-radius:var(--r);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;cursor:pointer;transition:all .2s;position:relative;min-height:160px;background:var(--bg)}
.dz:hover,.dz.over{border-color:var(--green);background:rgba(0,200,117,.03)}
.dz input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dzi{font-size:28px;opacity:.4;pointer-events:none}
.dzt{font-size:13px;font-weight:500;color:var(--text2);pointer-events:none}
.dzh{font-size:11px;color:var(--text3);pointer-events:none}
.thumbs{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.thumb{width:56px;height:56px;border-radius:8px;overflow:hidden;border:1px solid var(--border)}
.thumb img{width:100%;height:100%;object-fit:cover}
.urlin{width:100%;padding:10px 14px;background:var(--bg);border:1px solid var(--border);border-radius:8px;font-size:13px;color:var(--text);outline:none}
.urlin:focus{border-color:var(--green)}
.urlin::placeholder{color:var(--text3)}
.togs{display:flex;flex-direction:column;gap:8px}
.tog{display:flex;align-items:center;justify-content:space-between;padding:11px 14px;background:var(--bg);border:1px solid var(--border);border-radius:8px;cursor:pointer}
.tog.on{border-color:var(--green);background:rgba(0,200,117,.04)}
.tlbl{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:500}
.tdesc{font-size:11px;color:var(--text3);margin-top:2px}
.sw{width:34px;height:19px;border-radius:10px;background:var(--bg3);border:1px solid var(--border2);position:relative;transition:all .2s;flex-shrink:0}
.sw::after{content:'';position:absolute;width:13px;height:13px;background:var(--text3);border-radius:50%;top:2px;left:2px;transition:all .2s}
.tog.on .sw{background:var(--gdim);border-color:var(--green)}
.tog.on .sw::after{background:var(--green);left:17px}
.lrow{display:flex;align-items:center;gap:10px;margin-top:10px;padding:10px 14px;background:var(--bg);border:1px solid var(--border);border-radius:8px}
.llbl{font-size:12px;color:var(--text2);white-space:nowrap}
.lsel{flex:1;background:transparent;border:none;color:var(--text);font-size:13px;outline:none;cursor:pointer}
.lsel option{background:var(--bg2)}
.runbtn{width:100%;padding:13px;background:var(--green);color:#000;border:none;border-radius:var(--r);font-family:'Syne',sans-serif;font-size:13px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px}
.runbtn:hover{background:#00e688}
.runbtn:disabled{background:var(--bg3);color:var(--text3);cursor:not-allowed}
.abts{display:flex;gap:8px;margin-top:8px}
.abt{flex:1;padding:9px;background:transparent;color:var(--text2);border:1px solid var(--border);border-radius:8px;font-size:12px;cursor:pointer}
.abt:hover{border-color:var(--border2);color:var(--text)}
.right{display:flex;flex-direction:column;background:var(--bg)}
.ohdr{border-bottom:1px solid var(--border);padding:0 32px;display:flex;align-items:center;height:48px}
.otab{height:100%;padding:0 16px;font-size:11px;font-weight:500;color:var(--text3);border:none;background:transparent;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;letter-spacing:.06em;text-transform:uppercase}
.otab.active{color:var(--green);border-bottom-color:var(--green)}
.ometa{margin-left:auto;font-size:11px;color:var(--text3)}
.obody{flex:1;overflow-y:auto;padding:32px}
.obody::-webkit-scrollbar{width:4px}
.obody::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.otext{font-size:15px;line-height:1.9;color:var(--text)}
.oacts{display:flex;gap:10px;margin-top:20px;padding-top:18px;border-top:1px solid var(--border)}
.oabtn{padding:8px 16px;background:var(--bg2);color:var(--text2);border:1px solid var(--border);border-radius:8px;font-size:12px;cursor:pointer;display:flex;align-items:center;gap:6px}
.oabtn:hover{border-color:var(--green);color:var(--green)}
.oabtn.spk{background:var(--gdim);color:var(--green);border-color:var(--green)}
.ph{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:14px;color:var(--text3);text-align:center}
.ph-icon{font-size:44px;opacity:.25}
.ph-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:700;color:var(--text2)}
.ph-sub{font-size:13px;line-height:1.6;max-width:300px}
.err{background:rgba(255,60,60,.06);border:1px solid rgba(255,60,60,.2);border-radius:var(--r);padding:16px;font-size:13px;color:#ff6b6b;line-height:1.6}
.pcard{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:24px;max-width:380px;margin:0 auto}
.ptitle{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--text3);margin-bottom:18px}
.pstep{display:flex;align-items:center;gap:12px;padding:10px 0;font-size:13px}
.pstep+.pstep{border-top:1px solid var(--border)}
.pstep.done{color:var(--green)}.pstep.active{color:var(--text)}.pstep.waiting{color:var(--text3)}
.si{font-size:15px;width:24px;text-align:center}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;width:13px;height:13px;border:2px solid var(--gdim);border-top-color:var(--green);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
@media(max-width:800px){.wrap{grid-template-columns:1fr}.left{border-right:none;border-bottom:1px solid var(--border)}nav{padding:0 16px}}
</style>
</head>
<body>
<nav>
  <span class="logo">Simpli<em>Scan</em></span>
  <div class="nst"><span class="dot" id="dot"></span><span id="sl">Connecting</span></div>
  <a href="/landing" class="nlink">About</a>
</nav>
<div class="wrap">
  <div class="left">
    <div class="sec">
      <div class="lbl">Input mode</div>
      <div class="tabs">
        <button class="tab active" id="tabUp" onclick="setMode('upload')">Upload</button>
        <button class="tab" id="tabBa" onclick="setMode('batch')">Batch</button>
        <button class="tab" id="tabUr" onclick="setMode('url')">URL</button>
      </div>
    </div>
    <div class="sec grow" id="upSec">
      <div class="lbl">Document</div>
      <div class="dz" id="dz" ondragover="dzOver(event)" ondragleave="dzLeave()" ondrop="dzDrop(event)">
        <input type="file" id="fileIn" accept="image/*" onchange="fileChange(event)"/>
        <div class="dzi">&#x2317;</div>
        <div class="dzt">Drop image here or click to browse</div>
        <div class="dzh">JPEG &middot; PNG &middot; WEBP &middot; TIFF &middot; BMP</div>
      </div>
      <div class="thumbs" id="thumbs"></div>
    </div>
    <div class="sec" id="urlSec" style="display:none">
      <div class="lbl">Image URL</div>
      <input class="urlin" id="urlIn" type="url" placeholder="https://example.com/image.jpg"/>
    </div>
    <div class="sec">
      <div class="lbl">AI features</div>
      <div class="togs">
        <div class="tog on" style="cursor:default">
          <div><div class="tlbl">&#x1F4D6; OCR extraction</div><div class="tdesc">Always on</div></div>
          <div class="sw"></div>
        </div>
        <div class="tog" id="togELI" onclick="togFeat('eli12')">
          <div><div class="tlbl">&#x1F9D2; Plain language</div><div class="tdesc">Simplify complex text</div></div>
          <div class="sw"></div>
        </div>
        <div class="tog" id="togTR" onclick="togFeat('translate')">
          <div><div class="tlbl">&#x1F310; Translate</div><div class="tdesc">Convert to another language</div></div>
          <div class="sw"></div>
        </div>
      </div>
      <div class="lrow" id="lrow" style="display:none">
        <span class="llbl">Translate to</span>
        <select class="lsel" id="langSel">
          <option value="Hebrew">Hebrew</option>
          <option value="English">English</option>
          <option value="Arabic">Arabic</option>
          <option value="Russian">Russian</option>
          <option value="French">French</option>
          <option value="Spanish">Spanish</option>
          <option value="Amharic">Amharic</option>
        </select>
      </div>
    </div>
    <div class="sec">
      <button class="runbtn" id="runBtn" onclick="doRun()">&#x25B6; Process document</button>
      <div class="abts">
        <button class="abt" onclick="doReset()">Reset</button>
        <button class="abt" onclick="location.reload()">Refresh</button>
      </div>
    </div>
  </div>
  <div class="right">
    <div class="ohdr">
      <button class="otab active" id="tab0" onclick="showTab('ocr')">Extracted text</button>
      <button class="otab" id="tab1" onclick="showTab('eli12')" style="display:none">Plain language</button>
      <button class="otab" id="tab2" onclick="showTab('translation')" style="display:none">Translation</button>
      <span class="ometa" id="meta"></span>
    </div>
    <div class="obody" id="outBody">
      <div class="ph"><div class="ph-icon">&#x25A4;</div><div class="ph-title">No document loaded</div><div class="ph-sub">Upload an image then click Process document.</div></div>
    </div>
  </div>
</div>
<script>
var mode='upload',selFiles=[],feats={eli12:false,translate:false},out={ocr:'',eli12:'',translation:''},isSpeaking=false,curTab='ocr';
fetch('/health').then(function(r){if(r.ok){document.getElementById('dot').className='dot on';document.getElementById('sl').textContent='Online';}}).catch(function(){});
function setMode(m){
  mode=m;
  document.getElementById('tabUp').classList.toggle('active',m==='upload');
  document.getElementById('tabBa').classList.toggle('active',m==='batch');
  document.getElementById('tabUr').classList.toggle('active',m==='url');
  document.getElementById('upSec').style.display=m!=='url'?'flex':'none';
  document.getElementById('urlSec').style.display=m==='url'?'':'none';
  document.getElementById('fileIn').multiple=(m==='batch');
  selFiles=[];renderThumbs();
}
function togFeat(k){
  feats[k]=!feats[k];
  document.getElementById(k==='eli12'?'togELI':'togTR').classList.toggle('on',feats[k]);
  if(k==='translate')document.getElementById('lrow').style.display=feats.translate?'flex':'none';
}
function fileChange(e){selFiles=Array.from(e.target.files);renderThumbs();}
function dzOver(e){e.preventDefault();document.getElementById('dz').classList.add('over');}
function dzLeave(){document.getElementById('dz').classList.remove('over');}
function dzDrop(e){
  e.preventDefault();document.getElementById('dz').classList.remove('over');
  selFiles=Array.from(e.dataTransfer.files).filter(function(f){return f.type.indexOf('image/')===0;});
  renderThumbs();
}
function renderThumbs(){
  var row=document.getElementById('thumbs');row.innerHTML='';
  selFiles.slice(0,8).forEach(function(f){
    var d=document.createElement('div');d.className='thumb';
    var i=document.createElement('img');i.src=URL.createObjectURL(f);
    d.appendChild(i);row.appendChild(d);
  });
}
function doRun(){
  var btn=document.getElementById('runBtn');
  btn.disabled=true;btn.innerHTML='<span class="spin"></span> Processing...';
  showProcessing();
  runAsync().catch(function(e){
    document.getElementById('outBody').innerHTML='<div class="err">&#x26A0; '+esc(e.message)+'</div>';
  }).finally(function(){
    btn.disabled=false;btn.innerHTML='&#x25B6; Process document';
  });
}
async function runAsync(){
  setStep(1,'active');
  var txt='';
  if(mode==='url'){
    var u=document.getElementById('urlIn').value.trim();
    if(!u)throw new Error('Please enter an image URL.');
    var r=await fetch('/ocr/url?mode=text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u})});
    if(!r.ok){var e=await r.json();throw new Error(e.detail||r.statusText);}
    txt=(await r.json()).text;
  } else {
    if(!selFiles.length)throw new Error('Please select an image file.');
    var fd=new FormData();
    if(mode==='batch')selFiles.forEach(function(f){fd.append('files',f);});
    else fd.append('file',selFiles[0]);
    var ep=mode==='batch'?'/ocr/batch':'/ocr/upload';
    var r=await fetch(ep+'?mode=text',{method:'POST',body:fd});
    if(!r.ok){var e=await r.json();throw new Error(Array.isArray(e.detail)?e.detail.map(function(x){return x.msg;}).join(', '):e.detail);}
    var d=await r.json();
    txt=mode==='batch'?d.results.map(function(x){return x.text;}).join('
---
'):d.text;
  }
  setStep(1,'done');out.ocr=txt;
  await sleep(200);setStep(2,'active');await sleep(300);setStep(2,'done');
  if(feats.eli12){
    setStep(3,'active');
    var r=await fetch('/ai/eli12',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:txt})});
    if(!r.ok)throw new Error('AI simplification failed.');
    out.eli12=(await r.json()).result;setStep(3,'done');
  }
  if(feats.translate){
    setStep(4,'active');
    var lang=document.getElementById('langSel').value;
    var src=feats.eli12?out.eli12:txt;
    var r=await fetch('/ai/translate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:src,language:lang})});
    if(!r.ok)throw new Error('Translation failed.');
    out.translation=(await r.json()).result;setStep(4,'done');
  }
  renderResults();
}
function sleep(ms){return new Promise(function(r){setTimeout(r,ms);});}
function showProcessing(){
  var h='<div class="pcard"><div class="ptitle">Processing</div>';
  h+='<div class="pstep waiting" id="s1"><span class="si">&#x1F4D6;</span> OCR extraction</div>';
  h+='<div class="pstep waiting" id="s2"><span class="si">&#x1F916;</span> AI correction</div>';
  if(feats.eli12)h+='<div class="pstep waiting" id="s3"><span class="si">&#x1F9D2;</span> Plain language</div>';
  if(feats.translate)h+='<div class="pstep waiting" id="s4"><span class="si">&#x1F310;</span> Translation</div>';
  h+='</div>';
  document.getElementById('outBody').innerHTML=h;
  document.getElementById('meta').textContent='';
  document.getElementById('tab1').style.display='none';
  document.getElementById('tab2').style.display='none';
}
function setStep(n,s){var el=document.getElementById('s'+n);if(el)el.className='pstep '+s;}
function renderResults(){
  document.getElementById('tab1').style.display=feats.eli12?'':'none';
  document.getElementById('tab2').style.display=feats.translate?'':'none';
  if(feats.translate)showTab('translation');
  else if(feats.eli12)showTab('eli12');
  else showTab('ocr');
  var wc=out.ocr.split(/\\s+/).filter(function(w){return w.length>0;}).length;
  document.getElementById('meta').textContent=wc+' words';
}
function showTab(tab){
  curTab=tab;
  document.querySelectorAll('.otab').forEach(function(t){t.classList.remove('active');});
  var ids={ocr:'tab0',eli12:'tab1',translation:'tab2'};
  var el=document.getElementById(ids[tab]);if(el)el.classList.add('active');
  var text=out[tab]||'';
  if(!text){
    document.getElementById('outBody').innerHTML='<div class="ph"><div class="ph-icon">&#x25A4;</div><div class="ph-title">No output</div></div>';
    return;
  }
  var heCount=0;
  for(var i=0;i<text.length;i++){var code=text.charCodeAt(i);if(code>=1488&&code<=1514)heCount++;}
  var isRtl=heCount>2;
  var lines=text.split('
').map(function(l){return esc(l);}).join('<br>');
  window._ct=text;
  document.getElementById('outBody').innerHTML=
    '<div class="otext" style="direction:'+(isRtl?'rtl':'ltr')+';text-align:'+(isRtl?'right':'left')+'">'+lines+'</div>'+
    '<div class="oacts">'+
    '<button class="oabtn" id="spkBtn" onclick="doSpeak()">&#x1F50A; Listen</button>'+
    '<button class="oabtn" onclick="doCopy(this)">&#x1F4CB; Copy</button>'+
    '</div>';
}
function doSpeak(){
  var btn=document.getElementById('spkBtn');
  if(isSpeaking){window.speechSynthesis.cancel();isSpeaking=false;btn.className='oabtn';btn.innerHTML='&#x1F50A; Listen';return;}
  var text=window._ct||'';
  var heC=0;for(var i=0;i<text.length;i++){var c=text.charCodeAt(i);if(c>=1488&&c<=1514)heC++;}
  var u=new SpeechSynthesisUtterance(text);
  u.lang=heC>2?'he-IL':'en-US';u.rate=0.88;
  u.onend=function(){isSpeaking=false;var b=document.getElementById('spkBtn');if(b){b.className='oabtn';b.innerHTML='&#x1F50A; Listen';}};
  isSpeaking=true;btn.className='oabtn spk';btn.innerHTML='&#x23F9; Stop';
  window.speechSynthesis.speak(u);
}
function doCopy(btn){
  navigator.clipboard.writeText(window._ct||'').then(function(){
    btn.innerHTML='&#x2705; Copied';setTimeout(function(){btn.innerHTML='&#x1F4CB; Copy';},1500);
  });
}
function doReset(){
  selFiles=[];out={ocr:'',eli12:'',translation:''};
  document.getElementById('fileIn').value='';
  document.getElementById('urlIn').value='';
  document.getElementById('thumbs').innerHTML='';
  document.getElementById('outBody').innerHTML='<div class="ph"><div class="ph-icon">&#x25A4;</div><div class="ph-title">No document loaded</div><div class="ph-sub">Upload an image then click Process document.</div></div>';
  document.getElementById('meta').textContent='';
  if(isSpeaking){window.speechSynthesis.cancel();isSpeaking=false;}
}
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
</script>
</body>
</html>"""

@app.get("/", include_in_schema=False, response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_UI_HTML)

_LANDING_HTML = """<!DOCTYPE html>
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
    return HTMLResponse(_LANDING_HTML)
