"""Copy installed Times New Roman fonts into the local TeX build directory."""
from pathlib import Path
import shutil
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];dest=ROOT/'manuscript/ded_style_article_v2/fonts';dest.mkdir(exist_ok=True)
for style,weight,name in [('normal','normal','times.ttf'),('normal','bold','timesbd.ttf'),('italic','normal','timesi.ttf'),('italic','bold','timesbi.ttf')]:
    prop=font_manager.FontProperties(family='Times New Roman',style=style,weight=weight)
    source=Path(font_manager.findfont(prop,fallback_to_default=False));shutil.copyfile(source,dest/name)
print('Local Times New Roman fonts prepared.')
