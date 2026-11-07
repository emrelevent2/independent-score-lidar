import numpy as np
from paired_rank_fusion import paired_rank_pvalue


def test_exhaustive_exchangeable_label_calibration_with_ties():
    # Uniformly designate any one row observed. Symmetry must control size
    # for this arbitrary finite multiset,without a distributional assumption.
    values=np.array([[3,1],[2,4],[2,4],[0,8],[5,0],[1,1],[7,2],[3,1],[1,9],[2,3]])
    p=[]
    for i in range(len(values)):
        r=paired_rank_pvalue(values[i],np.delete(values,i,axis=0));p.append(r['pvalue'])
        assert r['pvalue']>=min(r['component_pvalues']) and r['pvalue']>0
    for alpha in [.1,.2,.5,1.]:assert np.mean(np.array(p)<=alpha)<=alpha


def test_one_component_recovers_original_upper_tail_monte_carlo_rank():
    b=np.array([[1.],[2.],[2.],[5.]])
    r=paired_rank_pvalue([2.],b)
    assert r['pvalue']==4/5 and r['selected']==0
