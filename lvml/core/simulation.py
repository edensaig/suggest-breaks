import numpy as np
import scipy.integrate
from scipy.special import softmax
import pandas as pd
from numba import njit
import random
import functools
import collections
import math

MAX_RATING = 5

class StationaryRecommender:
    def __init__(
        self,
        p_fb: float,
        predicted_ratings: np.array,
        true_ratings: np.array,
        softmax_t: float,
    ):
        self.p_fb = p_fb
        self.predicted_ratings = np.array(predicted_ratings)
        self.true_ratings = np.array(true_ratings).round().astype(int)
        assert min(self.true_ratings)>=1
        assert max(self.true_ratings)<=MAX_RATING
        self.softmax_t = softmax_t

    def item_probabilities(self):
        return softmax(self.predicted_ratings/self.softmax_t)

    def rating_probabilities(self) -> np.array:
        item_probs = self.item_probabilities()
        probs = np.zeros(MAX_RATING)
        for rating, prob in zip(self.true_ratings, item_probs):
            assert rating-1>=0
            probs[rating-1] += prob
        return probs

    def effective_beta(self, lv_system) -> float:
        return lv_system.beta@self.rating_probabilities()

    def equilibrium(self, lv_system) -> np.array:
        alpha = lv_system.alpha
        beta = self.effective_beta(lv_system)
        gamma = lv_system.gamma
        delta = lv_system.delta
        rho = 1-self.p_fb
        if beta*rho-alpha<=0:
            return np.array([0,1])
        else:
            return np.array([
                (gamma/delta)*(1/rho)*(1-(alpha/beta)*(1/rho)),
                (alpha/beta)*(1/rho),
            ])
    

class DiscreteSimulationResult:
    def __init__(self, events, n_events, total_wellbeing, avg_rating, last_interaction_time, T):
        self.raw_events = events
        self.n_events = n_events
        self.total_wellbeing = total_wellbeing
        self.avg_rating = avg_rating
        self.last_interaction_time = last_interaction_time
        self.T = T
        self._df = None
        
    def events_df(self):
        assert self.raw_events is not None
        if self._df is None:
            self._df = (
                pd.DataFrame(
                    data=self.raw_events,
                    columns=[
                        't',
                        'latent_state',
                        'recommendations',
                        'avg_beta',
                        'avg_delta',
                    ]
                )
                .assign(
                    # lambda_i=lambda df: 1/df['t'].diff(),
                    cnt=lambda df: df.reset_index().index,
                    cumulative_rate=lambda df: df['cnt']/df['t'],
                    dt=lambda df: df['t'].diff(),
                )
                .sort_values('t')
                .set_index('t')
            )
        return self._df
        
    def empirical_rate(self):
        return self.n_events/self.T

    def empirical_wellbeing(self):
        return self.total_wellbeing/self.T

    def average_rating(self):
        return self.avg_rating

    def survival_pct(self):
        return self.last_interaction_time/self.T


class IVP:
    def __init__(self, y_0, T):
        self.y_0 = y_0
        self.T = T


class LotkaVolterraDynamicalSystem:
    def __init__(self, alpha, beta, gamma, delta):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.num_channels = len(self.beta)

    def simulate_ode(self, ivp, recommender, num_steps=250):
        f = self._lv_ode_f()
        t_vec = np.linspace(0,ivp.T,num_steps)
        a = 1-recommender.p_fb
        beta = recommender.effective_beta(self)
        ode_solver_result = scipy.integrate.solve_ivp(
            fun=lambda t,y: f(y, a, beta),
            t_span=(0,ivp.T),
            y0=ivp.y_0,
            t_eval=t_vec,
        )
        return ode_solver_result

    def simulate_discrete(self, ivp, recommender, batch_size=1, record_events=True, rate_limit_params=None):
        # make sure events are recorded if rate limits are applied
        assert record_events or rate_limit_params is None
        # run optimized discrete simulation
        events, agg = _simulate_discrete_optimized(
            alpha=self.alpha,
            beta=tuple(self.beta),
            gamma=self.gamma,
            delta=self.delta,
            y_0=ivp.y_0,
            T=ivp.T,
            p_fb=recommender.p_fb,
            item_probabilities=recommender.item_probabilities(),
            true_ratings=recommender.true_ratings,
            batch_size=batch_size,
            rate_limit_params=rate_limit_params,
            record_events=record_events,
        )
        # make sure event counts match
        if record_events:
            assert agg.n_events==len(events)
        return DiscreteSimulationResult(
            events=events if record_events else None,
            n_events=agg.n_events,
            total_wellbeing=agg.total_wellbeing,
            avg_rating=agg.avg_rating,
            last_interaction_time=agg.last_interaction,
            T=ivp.T,
        )
    
    def _lv_ode_f(self):
        alpha = self.alpha
        gamma = self.gamma
        delta = self.delta
        return lambda y, a, beta: np.array([
            -alpha + a*beta*y[1],
            gamma*(1-y[1]) - a*delta*y[0],
        ])*y


AggregatedSimulationResult = collections.namedtuple(
    'AggregatedSimulationResult',
    ['n_events', 'total_wellbeing', 'avg_rating', 'last_interaction'],
)

DiscreteSimulationEvent = collections.namedtuple(
    'DiscreteSimulationEvent',
    [ 't', 'y', 'feedback', 'avg_beta', 'avg_delta'],
)

RateLimitParams = collections.namedtuple(
    'RateLimitParams',
    ['lookback_n','rate_threshold','cooldown_period'],
)

@njit
def random_seed(n):
    np.random.seed(n)

@njit
def _simulate_discrete_optimized(
    alpha: np.array,
    beta: np.array,
    gamma: np.array,
    delta: np.array,
    y_0: np.array,
    T: float,
    p_fb: float,
    item_probabilities: np.array,
    true_ratings: np.array,
    batch_size: int,
    rate_limit_params: RateLimitParams,
    record_events: bool = True,
):
    t = 0.0
    y = np.copy(y_0)
    events = []
    n_events = 0
    total_wellbeing = 0
    total_non_empty = 0
    total_rating = 0
    last_interaction_time = 0
    cooldown_end=0
    if y[0]<=0:
        return events, AggregatedSimulationResult(0,0.0,0.0,0.0)
    dt = 1/y[0]
    item_cdf = np.cumsum(item_probabilities)
    assert abs(item_cdf[-1]-1)<=1e-5
    while t<T and (y>0).all():
        last_interaction_time = t
        explicit_feedback = []
        num_non_empty = 0
        total_beta = 0
        total_delta = 0
        for batch_ind in range(batch_size):
            fb = (np.random.rand() <= p_fb) or (t<cooldown_end) # forced break decision
            if fb:
                if record_events:
                    explicit_feedback.append(None)
            else:
                # stochastic recommendation
                num_non_empty += 1
                selected_item = np.searchsorted(item_cdf, np.random.rand(), side='right')
                true_rating = true_ratings[selected_item]
                if record_events:
                    explicit_feedback.append((selected_item, true_rating))
                total_rating += true_rating
                total_beta += beta[true_rating-1]
                total_delta += delta
        avg_beta = total_beta / batch_size
        avg_delta = total_delta / batch_size
        if record_events:
            events.append(DiscreteSimulationEvent(
                t=t,
                y=y.copy(),
                feedback=explicit_feedback,
                avg_beta=avg_beta,
                avg_delta=avg_delta,
            ))
            if rate_limit_params is not None:
                if n_events>rate_limit_params.lookback_n:
                    lookback_dt = t-events[-(rate_limit_params.lookback_n+1)].t
                    lookback_rate = rate_limit_params.lookback_n/lookback_dt
                    if lookback_rate>rate_limit_params.rate_threshold:
                        cooldown_end = math.ceil(t/rate_limit_params.cooldown_period)*rate_limit_params.cooldown_period
        n_events += 1
        total_wellbeing += y[1]
        total_non_empty += num_non_empty
        # advance system
        y += np.array([
            -alpha + avg_beta*y[1],
            gamma*(1-y[1]) - avg_delta*y[0],
        ])*y*dt
        # next timestamp is 1/rate
        if y[0]<=0:
            break
        # dt = 0.01
        dt = 1/y[0]
        t += dt
    return (
        events,
        AggregatedSimulationResult(
            n_events=n_events,
            total_wellbeing=total_wellbeing,
            avg_rating=total_rating/total_non_empty if total_non_empty>0 else 0.0,
            last_interaction=last_interaction_time,
        ),
    )