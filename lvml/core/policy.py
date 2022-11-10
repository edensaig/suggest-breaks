import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence
import scipy.optimize


class ForcedBreaksPolicy:
    def __init__(self, *args, **kwargs):
        pass

    def p_forced_break(self, X: ArrayLike) -> float:
        " Return the probability of forced break given user features "
        raise NotImplementedError


class MyopicForcedBreaksPolicy(ForcedBreaksPolicy):
    """
    Implementation of the `default` policy
    """
    def p_forced_break(self, X: ArrayLike) -> float:
        # X.shape is assumed to be (n_users, n_features)
        return np.zeros(len(X))


class PredictionBasedForcedBreaksPolicy(ForcedBreaksPolicy):
    """
    Policy optimization based on engagement predictions (abstract class)
    """
    def __init__(self, predictors, *args, **kwargs):
        # predictors is a dictionary mapping forced break probabilities (e.g p=0.1)
        # to consumption rate predictors f_p(u)
        self.predictors = predictors


class LotkaVolterraAlphaBetaForcedBreaksPolicy(PredictionBasedForcedBreaksPolicy):
    """
    Policy optimization based on LV parameter estimation (abstract class)
    """
    @staticmethod
    def lv_params_from_predictions(predictions):
        A = np.zeros((len(predictions),2))
        b = np.zeros(len(predictions))
        for j,(p, fp) in enumerate(predictions.items()):
            A[j,0] = -1/(1-p)**2
            A[j,1] = 1/(1-p)
            b[j]=fp
        (abgd, gd), loss = scipy.optimize.nnls(A,b)
        return abgd/(gd+1e-5), gd


    def alpha_beta_hat(self, X):
        n_users = len(X)
        out = np.zeros(n_users)
        for i in range(n_users):
            predictions = {
                p: f(X[[i]])
                for p,f in self.predictors.items()
            }
            out[i] = self.lv_params_from_predictions(predictions)[0]
        return out


class LotkaVolterraOptimalForcedBreaksPolicy(LotkaVolterraAlphaBetaForcedBreaksPolicy):
    """
    Implementation of the `LV` policy
    """
    def __init__(self, max_p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_p = max_p

    def p_forced_break(self, X: ArrayLike) -> float:
        ab = self.alpha_beta_hat
        return np.clip(
            1-2*ab(X),
            a_min=0,
            a_max=self.max_p,
        )


class ArgmaxForcedBreaksPolicy(PredictionBasedForcedBreaksPolicy):
    """
    Implementation of the `best-of` policy
    """
    def p_forced_break(self, X: ArrayLike) -> float:
        policies = np.array(list(self.predictors))
        predictions = np.vstack([self.predictors[p](X) for p in policies])
        argmax = np.argmax(predictions, axis=0)
        return policies[argmax]
