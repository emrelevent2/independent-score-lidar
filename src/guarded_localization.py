"""Keep local existence decisions; cautiously adjust the reported position.

Score identity means this postprocessor cannot change a local null alarm.
It can change localization errors, which require separate held-out testing.
"""
from support_routing import donor_support


def guarded_report(proposal,spatial,local,setting):
    displacement=abs(local['delay']-proposal['secondary'])
    use=donor_support(proposal)>=3.5 and displacement<=setting['distance'] and spatial['score']>=setting['fraction']*local['score']
    delay=(1-setting['blend'])*local['delay']+setting['blend']*proposal['secondary'] if use else local['delay']
    return dict(score=float(local['score']),primary=float(local['primary']),delay=float(delay),head='guarded' if use else 'local',
        displacement=float(displacement),location_adjusted=bool(use),local_decision_unchanged=True)
