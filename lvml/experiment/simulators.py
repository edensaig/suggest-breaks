import functools

import numpy as np

import surprise

from ..core.simulation import StationaryRecommender, IVP

__all__=[
    'EquilibriumEmpiricalRateSimulator',
    'DiscreteEmpiricalRateSimulator',
    'LatentUserParameters',
]

class EmpiricalRateSimulator:
    def __init__(
        self,
        lv,
        kappa,
        random_state,
        simulator_params={},
    ):
        self.lv = lv
        self.simulator_params=simulator_params
        self.kappa = kappa
        self.rng = surprise.utils.get_rng(random_state)
    
    @staticmethod
    def evaluate(lv, ivp, recommender, simulator_params):
        raise NotImplementedError
        
    def evaluate_for_user(self, p_fb, user):
        rec = user.build_rec(p_fb, self.kappa)
        ivp = user.build_ivp(p_fb, self.kappa)
        return self.evaluate(
            lv=self.lv,
            ivp=ivp,
            recommender=rec,
            simulator_params=self.simulator_params,
        )
    
    def empirical_rate_from_users_vec(self, p_fb, users):
        p_fb = float(p_fb)
        return np.array([
            self.evaluate_for_user(p_fb, u)['rate']
            for u in users
        ])
    
    def evaluate_from_users_personalized(self, p_fb_vec, users):
        return [
            self.evaluate_for_user(p_fb, u)
            for u,p_fb in zip(users, p_fb_vec)
        ]

    
class EquilibriumEmpiricalRateSimulator(EmpiricalRateSimulator):
    @staticmethod
    def evaluate(lv, ivp, recommender, simulator_params):
        res = recommender.equilibrium(lv)
        return {
            'rate': res[0],
            'wellbeing': res[0]*res[1],
            'avg_rating': recommender.rating_probabilities()@np.arange(1,6),
            'survival_pct': int(res[0]>0),
        }

    
class DiscreteEmpiricalRateSimulator(EmpiricalRateSimulator):
    @staticmethod
    def evaluate(lv, ivp, recommender, simulator_params):
        res = lv.simulate_discrete(ivp, recommender, **simulator_params)
        return {
            'rate': res.empirical_rate(),
            'wellbeing': res.empirical_wellbeing(),
            'avg_rating': res.average_rating(),
            'survival_pct': res.survival_pct(),
        }


class LatentUserParameters:
    def __init__(self, iuid, predictions, lv, softmax_t, simulation_length, rng):
        self.iuid = iuid
        self.uid = predictions[0].uid
        self.predictions = predictions
        self._predicted_ratings = np.array([pred.est for pred in predictions])
        self._true_ratings = np.array([pred.r_ui for pred in predictions])
        self._lv = lv
        self._softmax_t = softmax_t
        self._simulation_length = simulation_length
        self._rng = rng if rng is not None else surprise.get_rng()
    
    @functools.lru_cache(maxsize=None)
    def true_ratings(self, kappa):
        return (1-kappa)*self._true_ratings + kappa*self._predicted_ratings

    def predicted_ratings(self):
        return self._predicted_ratings

    @functools.lru_cache(maxsize=None)
    def build_rec(self, p_fb, kappa):
        # if p_fb==0:
        #     return self._myopic_rec
        return StationaryRecommender(
            p_fb=p_fb,
            predicted_ratings=self.predicted_ratings(),
            true_ratings=self.true_ratings(kappa),
            softmax_t=self._softmax_t,
        )

    @functools.lru_cache(maxsize=None)
    def build_ivp(self, p_fb, kappa):
        return IVP(
            y_0 = (
                self.build_rec(p_fb, kappa)
                .equilibrium(self._lv)
                *(1+1e-1*self._rng.uniform(low=-1,high=1))
            ),
            T = self._simulation_length,
        )


