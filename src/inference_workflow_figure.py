"""Vector scientific schematic separating certified and empirical pipelines."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,PathPatch
from matplotlib.path import Path as MplPath
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'manuscript/figures'


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.1,4.0));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,color in [(.015,'#edf4f8'),(.51,'#f7f2e9')]:
        ax.add_patch(FancyBboxPatch((x,.015),.475,.96,boxstyle='round,pad=0.003,rounding_size=0.012',facecolor=color,edgecolor='#9ba6ad',linewidth=.6))
    ax.text(.25,.95,'(a) Certified existence',ha='center',va='center',weight='bold',fontsize=10)
    ax.text(.75,.95,'(b) Empirical guarded location',ha='center',va='center',weight='bold',fontsize=10)
    def box(x,y,w,h,text,face='white',size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.008',facecolor=face,edgecolor='#495962',linewidth=.65))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.35)
    def arrow(a,b,color='#53636c'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,linewidth=.75,color=color,shrinkA=2,shrinkB=2))
    def elbow(points,color='#53636c'):
        codes=[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)
        ax.add_patch(FancyArrowPatch(path=MplPath(points,codes),arrowstyle='-|>',mutation_scale=8,linewidth=.75,color=color))
    box(.04,.80,.17,.10,'Donors $Z$\nquery excluded')
    box(.275,.80,.17,.10,'Query $H$\nall local photons')
    box(.04,.58,.17,.15,'Proper mixture\nspatial + global',size=7.2)
    box(.275,.58,.17,.15,'Continuous null\nupper bound $U(H)$')
    box(.09,.35,.32,.15,'Certified lower log evidence\n$Q_- - U(H)$')
    box(.065,.105,.37,.15,r'Declare if $Q_- - U\,\geq\,\log(1/\alpha)$'+'\nModel-conditional null control',face='#d8e9f2',size=7.6)
    arrow((.125,.80),(.125,.73));arrow((.36,.80),(.36,.73))
    arrow((.125,.58),(.20,.50));arrow((.36,.58),(.30,.50));arrow((.25,.35),(.25,.255))
    elbow([(.445,.85),(.47,.85),(.47,.425),(.415,.425)])
    ax.text(.475,.77,'$H$',ha='center',fontsize=7)
    ax.text(.25,.045,'Proper alternative + complete null cover',ha='center',fontsize=7,color='#40515b')
    box(.535,.80,.17,.10,'Donors $Z$\nquery excluded')
    box(.77,.80,.17,.10,'Query $H$\nall local photons')
    box(.535,.58,.17,.15,'Donor location $d_Z$\nsupport feature $R_Z$',size=7.2)
    box(.77,.58,.17,.15,'Local / guided fits\n$S_L,\ d_L,\ S_G$',size=7.2)
    box(.58,.35,.34,.15,'Support, position and score agree?\nYes: $\hat d=(d_L+d_Z)/2$\nOtherwise: $\hat d=d_L$',size=7.6)
    box(.535,.105,.195,.15,'Report $\hat d$\nonly if declared',face='#f1e2c9')
    box(.77,.105,.195,.15,'Declare if $S_L>c$\nsame local decision',face='#f1e2c9',size=7.6)
    arrow((.62,.80),(.62,.73));arrow((.855,.80),(.855,.73))
    arrow((.705,.655),(.77,.655));arrow((.62,.58),(.67,.50));arrow((.855,.58),(.83,.50))
    arrow((.65,.35),(.635,.255));arrow((.77,.18),(.73,.18))
    elbow([(.94,.655),(.975,.655),(.975,.285),(.865,.285),(.865,.26)])
    ax.text(.75,.045,'Empirical cut-off; no inherited e-value guarantee',ha='center',fontsize=6.8,color='#40515b')
    OUT.mkdir(exist_ok=True)
    for extension in ['pdf','png']:fig.savefig(OUT/f'inference_workflow.{extension}',bbox_inches='tight',pad_inches=.04,dpi=240)
    plt.close(fig);print('Created vector inference workflow',flush=True)


if __name__=='__main__':main()
