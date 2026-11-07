import unittest
import numpy as np
from deadtime_exposure import sufficient_statistics
from safe_deadtime_exposure import safe_lag,safe_statistics,simulate_two_dead_times


class SafeExposureTests(unittest.TestCase):
    def test_censoring_lag_matches_two_known_cases(self):
        self.assertEqual(safe_lag(50,80),130.)
        self.assertEqual(safe_lag(198,80),198.)

    def test_short_gap_hits_restart_mask_even_when_unscored(self):
        s,e,info=safe_statistics([1.,3.,7.5],3,4,3.)
        np.testing.assert_array_equal(info['eligible'],[True,False,True])
        np.testing.assert_array_equal(s,[0,1,0,1])
        # Active intervals [0,1),[6,7.5),[10.5,12).
        np.testing.assert_allclose(e,[1.,0.,1.5,1.5])
        self.assertEqual(info['safe_duration'],4.)

    def test_safe_mask_is_really_active_despite_hidden_avalanches(self):
        k=16;td=5.3;te=9.2;lag=safe_lag(td,te)
        obs,latent=simulate_two_dead_times(np.full(k,.15),200,td,te,8721)
        s,e,info=safe_statistics(obs,200,k,lag)
        self.assertGreater(len(latent),len(obs));self.assertGreater(len(obs),int(s.sum()))
        for a,b in zip(info['open_starts'],info['open_ends']):
            j=np.searchsorted(latent,a,side='right')-1
            if j>=0:self.assertGreaterEqual(a-latent[j],td-1e-9)
            # Any latent hit strictly inside a safe interval must be observed.
            inside=latent[(latent>a+1e-9)&(latent<b-1e-9)]
            self.assertEqual(len(inside),0)

    def test_when_detector_dead_exceeds_electronics_all_observed_hits_are_scored(self):
        obs,latent=simulate_two_dead_times(np.full(16,.03),200,19.3,8.,981)
        np.testing.assert_array_equal(obs,latent)
        s,e,info=safe_statistics(obs,200,16,19.3);a,b=sufficient_statistics(obs,200,16,19.3)
        np.testing.assert_array_equal(s,a);np.testing.assert_allclose(e,b,atol=2e-10)

    def test_safe_homogeneous_rate_estimation(self):
        rates=np.full(16,.04);obs,latent=simulate_two_dead_times(rates,20000,23.7,61.2,65327)
        s,e,_=safe_statistics(obs,20000,16,safe_lag(23.7,61.2))
        self.assertLess(abs(s.sum()/e.sum()-.04),.0015)


if __name__=='__main__':unittest.main()
