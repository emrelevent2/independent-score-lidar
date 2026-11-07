"""Empirical fallback when a separated local candidate has stronger evidence."""
from support_routing import route_report


def disagreement_report(proposal,spatial,local,normalizers,ratio=1.5,gate=3.5):
    r=route_report(proposal,spatial,local,normalizers,'route',gate)
    ss=spatial['score']/normalizers['spatial'];ls=local['score']/normalizers['local']
    separated=abs(local['delay']-proposal['secondary'])>4.
    override=bool(separated and ls>ratio*ss)
    if override:
        r.update(score=float(ls),delay=float(local['delay']),primary=float(local['primary']),
            head='local',location_source='queried local likelihood fit')
    r.update(policy='disagreement_route',ratio=float(ratio),disagreement_override=override,
        disagreement_bins=float(abs(local['delay']-proposal['secondary'])))
    return r
