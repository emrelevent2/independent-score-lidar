"""Render all submission PDFs with the bundled document runtime."""
from pathlib import Path
import json,re,hashlib
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript';out=M/'qa';out.mkdir(exist_ok=True)
info={}
for name in ['main','supplement','cover_letter']:
 path=M/(name+'.pdf');doc=pdfium.PdfDocument(path);reader=PdfReader(path);pages=[]
 for i in range(len(doc)):
  png=out/f'{name}_{i+1:02d}.png';doc[i].render(scale=1.6).to_pil().save(png)
  text=reader.pages[i].extract_text()
  assert text and '\ufffd' not in text,(name,i,'Missing glyph/text')
  pages.append(dict(page=i+1,characters=len(text),image=str(png),sha256=hashlib.sha256(png.read_bytes()).hexdigest()))
 # Contact sheets are only navigation aids; inspect full pages for final QA.
 for start in range(0,len(pages),3):
  thumbs=[]
  for item in pages[start:start+3]:
   im=Image.open(item['image']).convert('RGB');im.thumbnail((650,900));thumbs.append(im)
  canvas=Image.new('RGB',(sum(im.width for im in thumbs),max(im.height for im in thumbs)),(220,220,220));x=0
  for im in thumbs:canvas.paste(im,(x,0));x+=im.width
  canvas.save(out/f'{name}_sheet_{start//3+1}.png')
 info[name]=dict(pages=pages,pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
source=(M/'main.tex').read_text();abstract=source.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
info['abstract_word_count']=len(abstract.split())
assert 150<=info['abstract_word_count']<=250
assert len(info['main']['pages'])<=13
(out/'render_manifest.json').write_text(json.dumps(info,indent=2))
print({k:len(v['pages']) for k,v in info.items() if isinstance(v,dict)})
print('Abstract words',info['abstract_word_count'])
