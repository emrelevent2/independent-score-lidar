import numpy as np
from guarded_real_analysis import joint_block_difference


def test_same_sequence_frames_share_block_weights_and_preserve_cancellation():
    q=[dict(row=0,column=0),dict(row=16,column=0)]
    a=[dict(pair=True),dict(pair=False)];b=[dict(pair=False),dict(pair=True)]
    f1=dict(sequence=3,queries=q,left=a,right=b);f2=dict(sequence=3,queries=q,left=b,right=a)
    result=joint_block_difference([f1,f2],'pair',replicates=100)
    assert result['sequences']==1 and result['frames']==2
    assert result['equal_frame_mean_difference']==0. and np.allclose(result['interval_95'],[0.,0.])


def test_zero_observed_discordance_is_explicit_not_a_zero_risk_claim():
    q=[dict(row=0,column=0),dict(row=16,column=0)];e=[dict(pair=True),dict(pair=False)]
    result=joint_block_difference([dict(sequence=3,queries=q,left=e,right=e)],'pair',replicates=100)
    assert result['observed_zero_discordance'] and result['gain']==result['loss']==0
    assert 'not zero population risk' in result['interpretation']
