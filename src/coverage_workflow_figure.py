"""Vector schematic of the two current coverage evaluation modes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.path import Path as MplPath
ROOT=Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.1,3.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,label,size=7.6,color='white'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.01',facecolor=color,edgecolor='#53636c',lw=.65))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,linespacing=1.35)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.7,color='#53636c',shrinkA=2,shrinkB=2))
    for x,title,color in [(.015,'(a) Certified coverage evidence','#edf4f8'),(.515,'(b) Calibrated numerical score','#f7f2e9')]:
        box(x,.02,.47,.95,'',color=color);ax.text(x+.235,.93,title,ha='center',fontsize=9,weight='bold')
        box(x+.025,.75,.17,.10,'Donors $Z$\nquery excluded')
        box(x+.275,.75,.17,.10,'Query $H$\nall local photons')
        box(x+.025,.54,.17,.13,'Spatial + global\nproper mixture',size=7.2)
        arrow((x+.11,.75),(x+.11,.67));arrow((x+.36,.75),(x+.36,.67))
        if x<.5:
            box(x+.275,.54,.17,.13,'Complete null\nupper bound $U$')
            box(x+.065,.29,.34,.16,'Sparse log numerator $Q_-$\noriginal weights retained\n'+r'Lower log evidence $Q_- - U$',size=7.8)
            box(x+.065,.075,.34,.13,r'Declare if $Q_- - U\geq\log(1/\alpha)$'+'\nmodel-conditional null control',color='#d6e8f1')
        else:
            box(x+.275,.54,.17,.13,'Feasible null\nlog likelihood $L$')
            box(x+.065,.29,.34,.16,'Full numerical mixture score\n'+r'$\log q_Z^{\mathrm{cov}}(H)-L(H)$',size=8)
            box(x+.065,.075,.34,.13,'Declare above calibrated cutoff\nno analytic evidence certificate',color='#efe1c5')
        arrow((x+.11,.54),(x+.17,.45));arrow((x+.36,.54),(x+.30,.45));arrow((x+.235,.29),(x+.235,.205))
        # Query scores the mixture; it does not define its parameters/weights.
        points=[(x+.445,.80),(x+.455,.80),(x+.455,.37),(x+.405,.37)]
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*3)
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=8,lw=.7,color='#53636c'))
    for ext in ['pdf','png']:fig.savefig(ROOT/f'manuscript/figures/coverage_workflow_v5.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
