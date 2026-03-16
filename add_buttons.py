path = "/Users/gilbartor/ocr_service/main.py"
c = open(path).read()

# Find Run OCR button
idx = c.find("Run OCR</button>")
if idx == -1:
    print("Not found! Searching...")
    idx = c.find("Run OCR")
    print("Context:", repr(c[idx-100:idx+100]))
else:
    end = idx + len("Run OCR</button>")
    buttons = '<div style=\\"display:flex;gap:8px;margin-top:8px\\"><button class=\\"run-btn\\" style=\\"flex:1;background:transparent;color:var(--text-dim);border:1px solid var(--border)\\" onclick=\\"resetAll()\\">Reset</button><button class=\\"run-btn\\" style=\\"flex:1;background:transparent;color:var(--text-dim);border:1px solid var(--border)\\" onclick=\\"location.reload()\\">Refresh</button></div>'
    c = c[:end] + buttons + c[end:]
    print("Buttons added.")

if "function resetAll" not in c:
    c = c.replace("function escHtml(s)", """function resetAll(){selectedFiles=[];results=[];activeResult=0;document.getElementById('fileInput').value='';document.getElementById('urlInput').value='';document.getElementById('thumbRow').innerHTML='';document.getElementById('outputBody').innerHTML='<div class=\\"placeholder\\"><div class=\\"placeholder-icon\\">&#9636;</div><div class=\\"placeholder-text\\">Awaiting input</div></div>';document.getElementById('copyBtn').style.display='none';document.getElementById('outputMeta').textContent='';document.getElementById('viewTabs').style.display='none';document.getElementById('resultTabs').style.display='none';}
function escHtml(s)""")
    print("resetAll added.")

open(path, "w").write(c)
