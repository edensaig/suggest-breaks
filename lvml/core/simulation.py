import numpy as np
import scipy.integrate
from scipy.special import softmax
import pandas as pd
from numba import njit
import random
import functools
import collections
import math
import enum

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
    def __init__(self, events, aggregated_results, T):
        self.raw_events = events
        self.agg = aggregated_results
        self.T = T
        self._df = None
        
    def events_df(self):
        assert self.raw_events is not None
        if self._df is None:
            self._df = (
                pd.DataFrame(
                    data=self.raw_events,
                    columns=self.raw_events[0]._fields,
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
        return self.agg.n_events/self.T

    def average_rating(self):
        return self.agg.avg_rating

    def survival_pct(self):
        return self.agg.last_interaction_time/self.T


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

    def simulate_discrete(self, ivp, recommender, batch_size=1, record_events=True, rate_limit_params=None, adaptive_policy_params=None):
        # make sure events are recorded if rate limits are applied
        assert record_events or rate_limit_params is None
        simulate = lambda t0, T, p_fb: _simulate_discrete_optimized(
            lv_params=LotkaVolterraSystemParams(
                alpha=self.alpha,
                beta=tuple(self.beta),
                gamma=self.gamma,
                delta=self.delta,
            ),
            y0=ivp.y_0,
            t0=t0,
            T=T,
            p_fb=p_fb,
            item_probabilities=recommender.item_probabilities(),
            true_ratings=recommender.true_ratings,
            batch_size=batch_size,
            rate_limit_params=rate_limit_params,
            adaptive_policy_params=None if adaptive_policy_params is None else _AdaptiveRatingsPolicyInternalParams(
                rating_sampling_rate=adaptive_policy_params.rating_sampling_rate,
            ),
            record_events=record_events,
        )
        # run optimized discrete simulation
        events0, agg0 = simulate(
            t0=0.0,
            T=ivp.T if adaptive_policy_params is None else adaptive_policy_params.callback_time,
            p_fb=recommender.p_fb,
        )
        if adaptive_policy_params is not None:
            events1, agg1 = simulate(
                t0=agg0.next_interaction_time,
                T=ivp.T,
                p_fb=adaptive_policy_params.callback_f(events0, agg0),
            )
            events = events0+events1 if record_events else None
            agg = AggregatedSimulationResult(
                n_events=agg0.n_events + agg1.n_events,
                total_non_empty=agg0.total_non_empty + agg1.total_non_empty,
                avg_rating=(
                    (agg0.avg_rating*agg0.total_non_empty + agg1.avg_rating*agg1.total_non_empty)
                    /(agg0.total_non_empty+agg1.total_non_empty+1e-9)
                ),
                last_interaction_time=agg1.last_interaction_time,
                next_interaction_time=agg1.next_interaction_time,
                ratings_histogram=agg0.ratings_histogram + agg1.ratings_histogram,
                simulation_status=agg1.simulation_status,
            )
        else:
            events=events0
            agg=agg0
        # make sure event counts match
        if record_events:
            assert agg.n_events==len(events)
        return DiscreteSimulationResult(
            events=events if record_events else None,
            aggregated_results=agg,
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


class DiscreteSimulationStatus(enum.Enum):
    REACHED_TIME_HORIZON = 0
    EXTINCT_AT_START = 1
    EXTINCT = 2
    CALLBACK = 3


AggregatedSimulationResult = collections.namedtuple(
    'AggregatedSimulationResult',
    [
        'n_events',
        'total_non_empty',
        'avg_rating',
        'last_interaction_time',
        'next_interaction_time',
        'ratings_histogram',
        'simulation_status',
    ],
)

DiscreteSimulationEvent = collections.namedtuple(
    'DiscreteSimulationEvent',
    [
        't',
        'y',
        'feedback',
        'avg_beta',
        'avg_delta',
        'p_fb',
    ],
)

LotkaVolterraSystemParams = collections.namedtuple(
    'LotkaVolterraSystemParams',
    [
        'alpha',
        'beta',
        'gamma',
        'delta',
    ],
)

RateLimitParams = collections.namedtuple(
    'RateLimitParams',
    [
        'lookback_n',
        'rate_upper_bound',
        'cooldown_period',
    ],
)

AdaptiveRatingsPolicyParams = collections.namedtuple(
    'AdaptiveRatingsPolicyParams',
    [
        'callback_f',
        'callback_time',
        'rating_sampling_rate',
    ],
)

_AdaptiveRatingsPolicyInternalParams = collections.namedtuple(
    '_AdaptiveRatingsPolicyInternalParams',
    [
        'rating_sampling_rate',
    ],
)

@njit
def random_seed(n):
    np.random.seed(n)

@njit
def _simulate_discrete_optimized(
    lv_params: LotkaVolterraSystemParams,
    y0: np.array,
    t0: float,
    T: float,
    p_fb: float,
    item_probabilities: np.array,
    true_ratings: np.array,
    batch_size: int,
    rate_limit_params: RateLimitParams,
    adaptive_policy_params: _AdaptiveRatingsPolicyInternalParams,
    record_events: bool = True,
):
    alpha = lv_params.alpha
    beta = lv_params.beta
    gamma = lv_params.gamma
    delta = lv_params.delta
    t = t0
    y = np.copy(y0)
    events = []
    n_events = 0
    total_non_empty = 0
    total_rating = 0
    last_interaction_time = 0
    cooldown_end=0
    ratings_histogram=np.zeros(5)
    if y[0]<=0:
        return (
            events,
            AggregatedSimulationResult(
                n_events=0,
                total_non_empty=0,
                avg_rating=0.0,
                last_interaction_time=0.0,
                next_interaction_time=0.0,
                ratings_histogram=ratings_histogram,
                simulation_status=DiscreteSimulationStatus.EXTINCT_AT_START,
            )
        )
    dt = 1/y[0]
    next_interaction_time = dt
    item_cdf = np.cumsum(item_probabilities)
    assert abs(item_cdf[-1]-1)<=1e-5
    return_status = DiscreteSimulationStatus.REACHED_TIME_HORIZON
    while t<T and (y>0).all():
        last_interaction_time = t
        explicit_feedback = []
        num_non_empty = 0
        total_beta = 0
        total_delta = 0
        for batch_ind in range(batch_size):
            # stochastic recommendation    
            selected_item = np.searchsorted(item_cdf, np.random.rand(), side='right')
            item_rating = true_ratings[selected_item]
            item_beta = beta[item_rating-1]
            fb = (  # forced break decision
                (np.random.rand() <= p_fb)
                or (t<cooldown_end)
            )
            if fb:
                if record_events:
                    explicit_feedback.append(None)
            else:
                num_non_empty += 1    
                if record_events:
                    explicit_feedback.append((selected_item, item_rating))
                total_rating += item_rating
                total_beta += item_beta
                total_delta += delta
                if (adaptive_policy_params is None) or (np.random.rand() <= adaptive_policy_params.rating_sampling_rate):
                    ratings_histogram[item_rating-1] += 1
        avg_beta = total_beta / batch_size
        avg_delta = total_delta / batch_size
        if record_events:
            events.append(DiscreteSimulationEvent(
                t=t,
                y=y.copy(),
                feedback=explicit_feedback,
                avg_beta=avg_beta,
                avg_delta=avg_delta,
                p_fb=p_fb,
            ))
            if rate_limit_params is not None:
                if n_events>rate_limit_params.lookback_n:
                    lookback_dt = t-events[-(rate_limit_params.lookback_n+1)].t
                    lookback_rate = rate_limit_params.lookback_n/lookback_dt
                    if lookback_rate>rate_limit_params.rate_upper_bound:
                        cooldown_end = math.ceil(t/rate_limit_params.cooldown_period)*rate_limit_params.cooldown_period
        n_events += 1
        total_non_empty += num_non_empty
        # advance system
        y += np.array([
            -alpha + avg_beta*y[1],
            gamma*(1-y[1]) - avg_delta*y[0],
        ])*y
        if y[0]<=0:
            return_status = DiscreteSimulationStatus.EXTINCT
            break
        # next timestamp is 1/rate
        dt = 1/y[0]
        t += dt
        next_interaction_time = t
    return (
        events,
        AggregatedSimulationResult(
            n_events=n_events,
            total_non_empty=total_non_empty,
            avg_rating=total_rating/total_non_empty if total_non_empty>0 else 0.0,
            last_interaction_time=last_interaction_time,
            next_interaction_time=next_interaction_time,
            ratings_histogram=ratings_histogram,
            simulation_status=return_status,
        ),
    )


#
# Stateless behavioral model
#

class StatelessBehavioralModel:
    def __init__(self, tau):
        self.tau = tau

    def simulate_discrete(self, ivp, recommender, batch_size=1, record_events=True):
        events, agg = _simulate_stateless_optimized(
            tau=self.tau,
            T=ivp.T,
            p_fb=recommender.p_fb,
            item_probabilities=recommender.item_probabilities(),
            true_ratings=recommender.true_ratings,
            batch_size=batch_size,
            record_events=record_events,
        )
        # make sure event counts match
        if record_events:
            assert agg.n_events==len(events)
        return DiscreteSimulationResult(
            events=events if record_events else None,
            aggregated_results=agg,
            T=ivp.T,
        )

StatelessSimulationEvent = collections.namedtuple(
    'StatelessSimulationEvent',
    [
        't',
        'feedback',
        'p_fb',
    ],
)

@njit
def _simulate_stateless_optimized(
    tau: float,
    T: float,
    p_fb: float,
    item_probabilities: np.array,
    true_ratings: np.array,
    batch_size: int,
    record_events: bool = True,
):
    t = 0.0
    events = []
    n_events = 0
    total_non_empty = 0
    total_rating_overall = 0.0
    last_interaction_time = 0.0
    ratings_histogram=np.zeros(5)
    next_interaction_time = 0.0
    item_cdf = np.cumsum(item_probabilities)
    assert abs(item_cdf[-1]-1)<=1e-5
    return_status = DiscreteSimulationStatus.REACHED_TIME_HORIZON
    while t<T:
        last_interaction_time = t
        explicit_feedback = []
        num_non_empty = 0
        total_rating_batch = 0
        for batch_ind in range(batch_size):
            # stochastic recommendation
            selected_item = np.searchsorted(item_cdf, np.random.rand(), side='right')
            item_rating = true_ratings[selected_item]
            fb = (np.random.rand() <= p_fb)  # forced break decision
            if fb:
                if record_events:
                    explicit_feedback.append(None)
            else:
                num_non_empty += 1
                if record_events:
                    explicit_feedback.append((selected_item, item_rating))
                total_rating_overall += item_rating
                total_rating_batch += item_rating
                ratings_histogram[item_rating-1] += 1
        if record_events:
            events.append(StatelessSimulationEvent(
                t=t,
                feedback=explicit_feedback,
                p_fb=p_fb,
            ))
        n_events += 1
        total_non_empty += num_non_empty
        # advance system
        avg_rating_batch = total_rating_batch/batch_size
        dt = 1/(tau*avg_rating_batch+1e-6)
        t += dt
        next_interaction_time = t
    return (
        events,
        AggregatedSimulationResult(
            n_events=n_events,
            total_non_empty=total_non_empty,
            avg_rating=total_rating_overall/total_non_empty if total_non_empty>0 else 0.0,
            last_interaction_time=last_interaction_time,
            next_interaction_time=next_interaction_time,
            ratings_histogram=ratings_histogram,
            simulation_status=return_status,
        ),
    )
