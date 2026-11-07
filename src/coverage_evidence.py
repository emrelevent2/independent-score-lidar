"""Certified original-weight sparse evaluation of a proper coverage mixture."""
import hashlib
import numpy as np
from flint import arb
from coverage_alternative import coverage_components,numeric_scores,selected_indices,sparse_numerator_lower,location_report
from adaptive_evidence import mixture_interval,feasible_local_null
from envelope_evidence import continuous_null_bound,outward_float


def evaluate_coverage(h,proposal,settings):
    models=coverage_components(proposal,cycles=settings['cycles'],amplitudes=settings['secondary_amplitudes'])
    _,weighted,full_numeric=numeric_scores(h,models,settings['cycles']);location=location_report(models,weighted)
    old=mixture_interval(h,models['rates'][:4],settings['cycles']);witness=feasible_local_null(h,settings['cycles'],tuple(settings['fwhm_range']))
    threshold=(arb(1)/arb(str(settings['alpha']))).log();upper_threshold=outward_float(threshold,True)
    old_upper_e=outward_float(arb(old['upper'])-arb(witness['lower']),True)
    candidate=(old_upper_e>=outward_float(threshold,False) or full_numeric-witness['lower']>=upper_threshold-settings['numerical_screen_margin'])
    sparse=None;bound=None;old_log=new_log=None;old_declared=new_declared=False
    if candidate:
        sparse=sparse_numerator_lower(h,models,selected_indices(weighted,settings['global_top_components']),settings['cycles'])
        bound=continuous_null_bound(h,settings['cycles'],tuple(settings['fwhm_range']),initial_cells=settings['initial_cells'],
            max_boxes=settings['max_boxes'],normalized_envelope=False)
        old_log=outward_float(arb(old['lower'])-arb(bound['upper']),False)
        new_log=outward_float(arb(sparse['lower'])-arb(bound['upper']),False)
        old_declared=bool(old_log>=upper_threshold);new_declared=bool(new_log>=upper_threshold)
    return dict(full_components=len(models['weights']),spatial_mass=str(models['spatial_mass']),
        rate_matrix_sha256=hashlib.sha256(models['rates'].tobytes()).hexdigest(),old_mixture_interval=old,
        local_null_witness=witness,numeric_full_log_mixture=full_numeric,
        old_fast_score=(old['lower']+old['upper'])/2-witness['lower'],coverage_fast_score=full_numeric-witness['lower'],
        screened=not candidate,screen_is_not_acceptance_certificate=True,sparse_numerator=sparse,bound=bound,
        threshold_upper=upper_threshold,old_log_e_lower=old_log,coverage_log_e_lower=new_log,
        old_report=dict(declared=old_declared,delay=proposal['secondary'],primary=proposal['primary'],head='spatial'),
        coverage_report=dict(declared=new_declared,primary=proposal['primary'],**location))
