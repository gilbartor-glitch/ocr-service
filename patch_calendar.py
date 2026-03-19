import re

# ── Patch main.py ──────────────────────────────────────────────
content = open('main.py').read()
old = '- reply_address: full postal address to reply to if found in document or null'
new = '''- reply_address: full postal address to reply to if found in document or null
- deadline_date: the most important deadline or due date in ISO format YYYY-MM-DD or null
- deadline_title: short label for the deadline e.g. "Payment due" or "Appeal deadline" or null'''
if old in content:
    open('main.py','w').write(content.replace(old, new))
    print('main.py patched')
else:
    print('main.py: text not found, searching...')
    idx = content.find('reply_address')
    print(repr(content[idx:idx+100]))

# ── Patch ui.html ──────────────────────────────────────────────
content = open('ui.html').read()
old = "function showUrgent(){aiModal('Time-sensitive actions','List all deadlines, due dates, and time-sensitive actions from this document with dates if available.\\n\\n'+out.ocr);}"
new = """function showUrgent(){
  var a=analysis||{};
  var calBtn='';
  if(a.deadline_date){
    var gcal='https://calendar.google.com/calendar/render?action=TEMPLATE&text='+encodeURIComponent(a.deadline_title||'Document deadline')+'&dates='+a.deadline_date.replace(/-/g,'')+'%2F'+a.deadline_date.replace(/-/g,'')+'&details='+encodeURIComponent('From SimpliScan: '+(a.document_type||''));
    calBtn='<div style="display:flex;gap:10px;margin-bottom:16px">'
      +'<a href="'+gcal+'" target="_blank" style="display:inline-flex;align-items:center;gap:8px;padding:10px 18px;background:linear-gradient(135deg,#4285f4,#34a853);color:#fff;border-radius:10px;text-decoration:none;font-weight:700;font-size:13px;font-family:Space Grotesk,sans-serif">&#x1F4C5; Add to Google Calendar</a>'
      +'<button onclick="downloadICS()" style="display:inline-flex;align-items:center;gap:8px;padding:10px 18px;background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:10px;font-weight:600;font-size:13px;font-family:Space Grotesk,sans-serif;cursor:pointer">&#x2B07;&#xFE0F; Download .ics</button>'
      +'</div>';
  }
  openModal('Time-sensitive actions','<div class="loading"><span class="spin"></span> Analyzing...</div>');
  fetch('/ai/eli12',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:'List all deadlines, due dates, and time-sensitive actions from this document with dates if available.\\n\\n'+out.ocr})})
    .then(function(r){return r.json();})
    .then(function(d){document.getElementById('modalBody').innerHTML=calBtn+'<div style="line-height:1.8;color:var(--text)">'+d.result.split('\\n').map(function(l){return esc(l);}).join('<br>')+'</div>';})
    .catch(function(){document.getElementById('modalBody').innerHTML='<div style="color:var(--red)">Failed.</div>';});
}
function downloadICS(){
  var a=analysis||{};
  if(!a.deadline_date)return;
  var d=a.deadline_date.replace(/-/g,'');
  var title=a.deadline_title||'Document deadline';
  var ics='BEGIN:VCALENDAR\\nVERSION:2.0\\nBEGIN:VEVENT\\nDTSTART;VALUE=DATE:'+d+'\\nDTEND;VALUE=DATE:'+d+'\\nSUMMARY:'+title+'\\nDESCRIPTION:From SimpliScan\\nEND:VEVENT\\nEND:VCALENDAR';
  var blob=new Blob([ics],{type:'text/calendar'});
  var url=URL.createObjectURL(blob);
  var a2=document.createElement('a');a2.href=url;a2.download='deadline.ics';a2.click();
}"""
if old in content:
    open('ui.html','w').write(content.replace(old, new))
    print('ui.html patched')
else:
    print('ui.html: text not found, searching...')
    idx = content.find('showUrgent')
    print(repr(content[idx:idx+120]))
