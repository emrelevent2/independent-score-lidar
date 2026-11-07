import unittest
import numpy as np
from deadtime_exposure import simulate_events,sufficient_statistics
from deadtime_scan import pulse,scan,fit_null,likelihood_parts
from deadtime_path_bootstrap import simulate,fixed_fit_scores,fine_scan,one_step,refit_from_start,fine_kernels,fine_correlations


class BootstrapTests(unittest.TestCase):
    def test_inverse_hazard_matches_reference_and_keeps_recovery(self):
        for seed in [18,119,1483]:
            f,_=pulse(7.3,32,1.5);rates=.8*f+.5/32
            a=simulate(rates,100,39.3,seed);b=simulate_events(rates,100,39.3,seed)
            np.testing.assert_allclose(a,b,rtol=0,atol=2e-11)
            self.assertTrue(np.all(np.diff(a)>=39.3-1e-10))

    def test_fine_fft_bank_matches_direct_templates(self):
        rng=np.random.default_rng(32);v=rng.normal(size=(3,32));ff,_=fine_kernels(32,1.5,4)
        actual=fine_correlations(v,ff)
        expected=np.array([[row@pulse(t/4,32,1.5)[0] for t in range(128)] for row in v])
        np.testing.assert_allclose(actual,expected,atol=2e-14)

    def test_batch_scores_equal_scalar_and_integer_source(self):
        f,_=pulse(13.3,64,1.5);rates=.8*f+.3/64;ss=[];ee=[]
        for seed in [719,720]:
            events=simulate(rates,400,81.3,seed);s,e=sufficient_statistics(events,400,64,81.3);ss.append(s);ee.append(e)
        fit=fit_null(ss[0],ee[0],1.5);theta=fit['theta'];v,loc=fixed_fit_scores(ss,ee,theta,1.5,subdivision=1)
        for i in range(2):
            old=scan(ss[i],ee[i],1.5,fit=fit)
            np.testing.assert_allclose(v[i],old['score'],rtol=1e-10,atol=1e-10);self.assertEqual(loc[i],old['delay'])
            scalar=fine_scan(ss[i],ee[i],1.5,fit=fit,subdivision=1)
            np.testing.assert_allclose(v[i],scalar['score'],rtol=1e-10,atol=1e-10)

    def test_one_step_increases_likelihood_and_refit_optimizes_all_parameters(self):
        theta=np.array([.8,13.3,.3]);f,_=pulse(theta[1],64,1.5);events=simulate(.8*f+.3/64,400,81.3,732)
        s,e=sufficient_statistics(events,400,64,81.3)
        ll=likelihood_parts(s,e,.8*f+.3/64,'duration')[0]
        a=one_step(s,e,theta,1.5,'duration');b=refit_from_start(s,e,theta,1.5,'duration')
        self.assertGreaterEqual(a['loglik'],ll-1e-10);self.assertGreaterEqual(b['loglik'],a['loglik']-1e-6)
        self.assertGreater(np.linalg.norm(b['theta']-theta),1e-3)


if __name__=='__main__':unittest.main()
