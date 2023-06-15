import contextlib
import datetime
import re
import numpy as np

__all__=[
    'ExperimentParamTracker',
    'print_progress',
]

class ExperimentParamTracker:
    def __init__(self):
        self.params_dct = {}

    def declare_param(self, value, name):
        assert not ((name in self.params_dct) and np.any(self.params_dct[name]!=value))
        # assert name not in self.params_dct
        # assert name.isalpha()
        self.params_dct[name] = value
        return value

    def get_param(self, name):
        return self.params_dct[name]


@contextlib.contextmanager
def print_progress(description):
    """
    Convenience function to track stage running times through the simulation.
    """
    print(f'{description}...',end=' ')
    start_time = datetime.datetime.now()
    try:
        yield None
    finally:
        end_time = datetime.datetime.now()
        print(f'Done ({end_time-start_time})')