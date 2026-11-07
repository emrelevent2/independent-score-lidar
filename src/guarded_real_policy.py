"""Prespecified count-model transfer of the simulator's guarded location.

No first-event cycle count is inferred. The existing working-Poisson donor
support feature is used. Exact local score identity preserves declarations,
not the accuracy of returned coordinates or a sensor-calibration guarantee.
"""
from real_poisson_analysis_v2 import route


def guarded_real_report(record,normalizers):
    report=route(record,normalizers,'local');q=record['proposal'];r=record['reports']
    displacement=abs(r['local']['delay']-q['secondary'])
    use=q['support_proxy']>=3.5 and displacement<=4. and r['guided']['score']>=.5*r['local']['score']
    if use:report.update(delay=float(.5*(r['local']['delay']+q['secondary'])),head='guarded')
    report.update(location_adjusted=bool(use),displacement=float(displacement),local_score_identical=True,
        transfer='Working-Poisson donor support; no invented cycle count or conditional e-value claim.')
    return report
