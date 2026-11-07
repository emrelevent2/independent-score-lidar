"""One-time assembly of a coherent results revision; preserve v2 sources."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manuscript'


def main():
    archive=M/'archive_v2';archive.mkdir(exist_ok=True)
    for name in ['main.tex','supplement.tex','cover_letter.tex']:
        backup=archive/name
        if not backup.exists():backup.write_bytes((M/name).read_bytes())
    old=(archive/'main.tex').read_text();prefix=old.split(r'\section{Experimental design}')[0]
    abstract='''Spatial information can improve weak-return lidar localization while obscuring whether the queried pixel supplies local photon support. We construct a donor-only alternative mixture and a certified upper bound on the continuous-delay, bounded-width single-return likelihood. Gaussian response envelopes, concave profile tangents and ball arithmetic make a universal-inference decision computable; its guarantee remains conditional on the stated detector model. Separately, we evaluate empirical reporting policies that distinguish existence decisions from returned locations. A fresh coherent-surface study localizes 702 of 900 returns versus 677 for guided query refinement, correcting 25 wrong-depth reports at identical null decisions, but exposes a severe isolated-target failure. Support-based routing recovers many isolated targets with a false-alarm tradeoff. A subsequent guarded policy instead preserves ordinary local decisions. In 6099 new acquisitions it localizes 480 of 900 coherent returns versus 468, reducing wrong-depth reports from 16 to four, with the same six declarations among 2000 nulls. The paired gain is 1.33 percentage points, with a two-stage 95\\% interval of 0.67--2.22 points. Isolated-target counts remain unchanged. Two held-out recorded frames, evaluated against released intervals after numerical count attenuation, do not reproduce general superiority of the preceding spatial policies. These outcomes establish a model-conditional computational construction and a limited controlled localization gain, with physical transfer still unresolved.'''
    prefix=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',prefix,flags=re.S)
    prefix=prefix.replace('a new independence principle, or a universal improvement over denoising.','a new independence principle, or a universal improvement over denoising.\nA fresh guarded-localization comparison and a negative held-out real-data\ntransfer experiment further distinguish controlled gains from physical\ngeneralization.')
    new=prefix+'\n'+r'\input{current_empirical_methods.tex}'+'\n'+r'\input{current_experiments.tex}'+'\n'+r'\input{current_results.tex}'+'\n'+r'\input{current_discussion.tex}'+'\n'
    (M/'main.tex').write_text(new)
    old_results=old.split(r'\section{Initial frozen evaluation}')[1].split(r'\section{Discussion}')[0]
    (M/'initial_additional_results.tex').write_text(r'\section{Additional controlled-study results}'+'\n'+old_results)
    refs=M/'references.tex';text=refs.read_text()
    if r'\bibitem{stb}' not in text:
        text=text.replace(r'\end{thebibliography}',r'''\bibitem{stb} H. Dong \emph{et al.}, ``SP-TransientBench: A real-captured
single photon perception benchmark,'' arXiv:2606.18952, 2026.
Dataset: \url{https://huggingface.co/datasets/shuinb/SP-TransientBench},
commit cadc5d7a0fa0cf5e6f49973ba3633198c4c3c02d.
\end{thebibliography}''')
        refs.write_text(text)
    print('Assembled main revision; abstract words',len(abstract.split()),flush=True)


if __name__=='__main__':main()
