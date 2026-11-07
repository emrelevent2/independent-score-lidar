"""Empirical pair-agreement policy. All scores require fresh calibration.

The raw query never affects the donor proposal. Query agreement can affect
the final score/report, so this policy is explicitly NOT an e-value.
"""
import numpy as np


def agreement_report(record,normalizers,setting):
    q=record['proposal'];local=record['reports']['local'];guided=record['reports']['guided']
    locations=np.array([local['primary'],local['delay']],float)
    donor=np.array([q['primary'],q['secondary']],float)
    distances=[np.max(np.abs(locations-donor)),np.max(np.abs(locations-donor[::-1]))]
    if distances[1]<distances[0]:donor=donor[::-1]
    distance=float(min(distances))
    compatible=distance<=setting['distance'] and q['support_proxy']>=3.5
    ls=local['score']/normalizers['local'];ss=guided['score']/normalizers['guided']
    score=ls+setting['bonus']*min(ls,ss) if compatible else ls
    if compatible:locations=(1-setting['blend'])*locations+setting['blend']*donor
    return dict(score=float(score),primary=float(locations[0]),delay=float(locations[1]),
        head='agreement' if compatible else 'local',compatible=bool(compatible),pair_displacement=distance)
