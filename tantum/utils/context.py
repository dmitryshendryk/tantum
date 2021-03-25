
import contextlib


@contextlib.contextmanager
def disable_tracking_bn_stats(model):

    def switch(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True 
    
    model.apply(switch)
    yield
    model.apply(switch)