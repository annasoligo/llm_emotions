"""
Activation steering experiments for vLLM models.

Core API:
    from experiments.steering import VLLMSteering

    steering = VLLMSteering(llm, layer=30)
    steering.load_vectors('experiments/steering/vectors/')
    steering.set('anger', scale=1.5)
"""
from .core import VLLMSteering

__all__ = ['VLLMSteering']
