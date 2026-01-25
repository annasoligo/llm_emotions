"""
Activation steering experiments for vLLM models.

Core API:
    from experiments.steering import VLLMSteering

    steering = VLLMSteering(llm, layer=30)
    steering.load_vectors('experiments/steering/vectors/')
    steering.set('anger', scale=1.5)

Plotting API (no vLLM required):
    from experiments.steering.plotting import plot_steering_results
"""
try:
    from .core import VLLMSteering
    __all__ = ['VLLMSteering']
except ImportError:
    # vLLM not installed - steering unavailable but plotting still works
    __all__ = []
