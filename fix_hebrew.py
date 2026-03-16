import re

path = "/Users/gilbartor/ocr_service/main.py"
content = open(path).read()

# ── Fix 1: Use two separate readers ──────────────────────────────────────────
content = content.replace(
    "_reader: easyocr.Reader | None = None",
    "_reader_en: easyocr.Reader | None = None\n_reader_he: easyocr.Reader | None = None"
)

content = content.replace(
    """    global _reader
    log.info("Loading EasyOCR model (English + Hebrew)…")
    loop = asyncio.get_event_loop()
    _reader = await loop.run_in_executor(
        None, lambda: easyocr.Reader(["en", "he"])
    )
    log.info("EasyOCR model ready.")""",
    """    global _reader_en, _reader_he
    log.info("Loading EasyOCR English model…")
    loop = asyncio.get_event_loop()
    _reader_en = await loop.run_in_executor(None, lambda: easyocr.Reader(["en"]))
    log.info("Loading EasyOCR Hebrew model…")
    _reader_he = await loop.run_in_executor(None, lambda: easyocr.Reader(["he"]))
    log.info("EasyOCR models ready (en + he).")"""
)

content = content.replace(
    """def get_reader() -> easyocr.Reader:
    if _reader is None:
        raise RuntimeError("Reader not initialised.")
    return _reader""",
    """def get_reader_en() -> easyocr.Reader:
    if _reader_en is None:
        raise RuntimeError("English reader not initialised.")
    return _reader_en

def get_reader_he() -> easyocr.Reader:
    if _reader_he is None:
        raise RuntimeError("Hebrew reader not initialised.")
    return _reader_he"""
)

# ── Fix 2: Dual-pass OCR with smart merge ────────────────────────────────────
content = content.replace(
    """def _run_ocr(image: np.ndarray) -> str:
    results = get_reader().readtext(image, detail=0, paragraph=True)
    return "\\n".join(results)


async def _ocr_from_bytes(data: bytes) -> str:
    loop = asyncio.get_event_loop()
    gray = await loop.run_in_executor(None, _bytes_to_gray, data)
    raw = await loop.run_in_executor(None, _run_ocr, gray)
    return _clean_text(raw)""",
    """def _run_ocr_en(image: np.ndarray) -> list:
    return get_reader_en().readtext(image, detail=1, paragraph=False)

def _run_ocr_he(image: np.ndarray) -> list:
    return get_reader_he().readtext(image, detail=1, paragraph=False)

def _is_hebrew(text: str) -> bool:
    he_chars = sum(1 for c in text if '\\u05d0' <= c <= '\\u05ea')
    return he_chars > len(text) * 0.3

def _merge_results(en_results: list, he_results: list) -> str:
    \"\"\"
    Merge English and Hebrew OCR results.
    For each detected region, pick the reading with higher confidence.
    Then sort all results by vertical position (top to bottom).
    \"\"\"
    combined = []

    # Add English results
    for (bbox, text, conf) in en_results:
        if text.strip():
            y = bbox[0][1]  # top-left y coordinate
            combined.append((y, bbox, text.strip(), conf, "en"))

    # Add Hebrew results — only if they don't heavily overlap an English region
    he_boxes = [(bbox, text, conf) for (bbox, text, conf) in he_results if text.strip()]
    en_boxes = [(bbox, text, conf) for (bbox, text, conf) in en_results if text.strip()]

    for (he_bbox, he_text, he_conf) in he_boxes:
        he_y = he_bbox[0][1]
        he_y2 = he_bbox[2][1]
        he_x = he_bbox[0][0]
        he_x2 = he_bbox[2][0]

        # Check overlap with any English result
        overlaps = False
        for (en_bbox, en_text, en_conf) in en_boxes:
            en_y = en_bbox[0][1]
            en_y2 = en_bbox[2][1]
            en_x = en_bbox[0][0]
            en_x2 = en_bbox[2][0]

            y_overlap = min(he_y2, en_y2) - max(he_y, en_y)
            x_overlap = min(he_x2, en_x2) - max(he_x, en_x)

            if y_overlap > 5 and x_overlap > 5:
                # Overlap found — keep whichever has higher confidence
                if he_conf > en_conf and _is_hebrew(he_text):
                    combined = [(y, b, t, c, lang) for (y, b, t, c, lang) in combined
                                if not (b == en_bbox)]
                    combined.append((he_y, he_bbox, he_text, he_conf, "he"))
                overlaps = True
                break

        if not overlaps and _is_hebrew(he_text):
            combined.append((he_y, he_bbox, he_text, he_conf, "he"))

    # Sort top-to-bottom
    combined.sort(key=lambda x: x[0])

    lines = [text for (_, _, text, _, _) in combined]
    return "\\n".join(lines)


async def _ocr_from_bytes(data: bytes) -> str:
    loop = asyncio.get_event_loop()
    gray = await loop.run_in_executor(None, _bytes_to_gray, data)

    # Run both passes concurrently
    en_task = loop.run_in_executor(None, _run_ocr_en, gray)
    he_task = loop.run_in_executor(None, _run_ocr_he, gray)
    en_results, he_results = await asyncio.gather(en_task, he_task)

    raw = _merge_results(en_results, he_results)
    return _clean_text(raw)"""
)

# Fix health endpoint
content = content.replace(
    'return {"status": "ok", "model": "easyocr", "languages": ["en", "he"], "version": "2.0.0"}',
    'return {"status": "ok", "model": "easyocr", "languages": ["en", "he"], "mode": "dual-pass", "version": "2.1.0"}'
)

open(path, "w").write(content)
print("✅ Done — dual-pass Hebrew+English OCR applied.")
print("Verify readers:", "_reader_en" in content, "_reader_he" in content)
print("Verify merge:", "_merge_results" in content)
