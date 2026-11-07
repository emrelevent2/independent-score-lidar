"""Classical paired-rank Monte Carlo fusion;not a new testing principle.

Rows contain one observed score pair plus jointly resampled pairs. Under
exchangeability the symmetric rank test is conservative. Fitted-null path
bootstrap rows are only approximately exchangeable under an unknown null.
"""
import numpy as np
from scipy.stats import rankdata


def paired_rank_pvalue(observed,bootstrap):
    observed=np.asarray(observed,dtype=float).reshape(1,-1);boot=np.asarray(bootstrap,dtype=float)
    if boot.ndim!=2 or boot.shape[1]!=observed.shape[1] or np.any(np.isnan(observed)) or np.any(np.isnan(boot)):raise ValueError('Paired score matrix required')
    values=np.vstack([observed,boot]);n=len(values)
    ranks=np.column_stack([rankdata(-values[:,j],method='max')/n for j in range(values.shape[1])]);minimum=np.min(ranks,axis=1);combined=float(np.count_nonzero(minimum<=minimum[0])/n)
    return dict(pvalue=combined,component_pvalues=ranks[0].tolist(),selected=int(np.argmin(ranks[0])))
