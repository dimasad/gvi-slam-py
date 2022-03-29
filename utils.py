"""Common utility functions."""


import collections
import functools

import jax
import jax.numpy as jnp
import numpy as np


class Packer:
    def __init__(self,  **shapes):
        self.shapes = collections.OrderedDict(**shapes)
        """The shapes of the underlying components."""
        
        self.sizes = np.array([np.prod(s, 0, int) for s in shapes.values()])
        """Size of each component."""
        
        self.offsets = np.cumsum(np.r_[0, self.sizes])
        """Offset of each element in the parent vector."""
        
        self.size = self.offsets[-1]
        """Total number of elements."""

    def pack(self, *args, **kwargs):
        components = self.collect(*args, **kwargs)
        return pack(*components.values())
    
    def collect(self, *args, **kwargs):
        components = collections.OrderedDict(zip(self.shapes, args))
        for name in self.shapes:
            arg = kwargs.get(name)
            if arg is not None:
                components[name] = arg
        shapes = [np.shape(val) for val in components.values()]
        assert set(components) == set(self.shapes)
        assert all(s1 == s2 for s1,s2 in zip(shapes, self.shapes.values()))
        return components
    
    def unpack(self, vec):
        assert np.shape(vec) == (self.size,)
        components = collections.OrderedDict()
        for i, (name, shape) in enumerate(self.shapes.items()):
            s = jnp.s_[self.offsets[i]:self.offsets[i+1]]
            components[name] = vec[s].reshape(shape)
        return components


def pack(*args):
    """Pack a sequence of arrays into a single vector."""
    return jnp.concatenate([jnp.ravel(arg) for arg in args])


def jax_jit_method(f=None, **kwargs):
    """Decorator for JAX just-in-time compilation of a instance method."""
    if f is None:
        return functools.partial(jax_jit_method, **kwargs)
    @functools.wraps(f)
    def getter(obj):
        return jax.jit(f.__get__(obj), **kwargs)
    return functools.cached_property(getter)


def jax_vectorize_method(f=None, **kwargs):
    """Decorator for JAX vectorization of a instance method."""
    if f is None:
        return functools.partial(jax_vectorize_method, **kwargs)
    
    # Replace aliases
    if 's' in kwargs:
        kwargs['signature'] = kwargs['s']
        del kwargs['s']
    if 'e' in kwargs:
        kwargs['excluded'] = kwargs['e']
        del kwargs['e']
    
    @functools.wraps(f)
    def getter(obj):
        return jax.numpy.vectorize(f.__get__(obj), **kwargs)
    return functools.cached_property(getter)


def jax_vectorize(f=None, **kwargs):
    """Decorator for JAX vectorization of a function."""
    if f is None:
        return functools.partial(jax_vectorize, **kwargs)
    
    # Replace aliases
    if 's' in kwargs:
        kwargs['signature'] = kwargs['s']
        del kwargs['s']
    if 'e' in kwargs:
        kwargs['excluded'] = kwargs['e']
        del kwargs['e']

    return jax.numpy.vectorize(f, **kwargs)
