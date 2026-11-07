import numpy as np
from flint import ctx,arb
from certified_safe_components import exact_safe_statistics,periodic_point_rates,log_likelihood_interval,mixture_log_lower
from certified_safe_null import bound_null,declare_from_fixed_mixture


def example():
    ctx.prec=128
    return exact_safe_statistics([.4,3.1,6.2,9.4,14.2,17.1,22.7,28.6],2,16,1.2)


def test_complete_null_cover_bounds_feasible_continuous_rate_points():
    s=example();b=bound_null(s,(1.4,1.6),initial_cells=4,max_boxes=24,gap=.001)
    assert b['complete_cover'] and b['upper']>=b['feasible_lower']
    for a in [0.,.3,2.,1000.]:
        for bg in [.01,1.,1000.]:
            for d in [0.,3.14159,9.87,16.]:
                for sig in [1.4,1.6]:
                    point=log_likelihood_interval(s,periodic_point_rates(16,a,d,bg,sig))
                    assert arb(b['upper'])>=point


def test_partial_cover_budget_never_certifies_from_only_a_local_witness():
    s=example();rates=[np.full(16,.1)];num=mixture_log_lower(s,rates,[1]);r=declare_from_fixed_mixture(s,num,initial_cells=4,max_boxes=4)
    assert not r['declared']
    if r['upper'] is not None:assert r['complete_cover']


def test_zero_events_have_exact_composite_null_supremum_zero():
    ctx.prec=128;s=exact_safe_statistics([],5,16,3.2);r=bound_null(s)
    assert r['upper']==0 and r['complete_cover'] and not r['declared']
