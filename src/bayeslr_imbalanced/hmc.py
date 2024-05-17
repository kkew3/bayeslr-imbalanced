"""
Hamiltonian Monte Carlo.
"""

import collections
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax import lax

HMCState = collections.namedtuple('HMCState', [
    'pos',
    'logdensity',
    'aux',
    'pos_grad',
])


def init(pos, logdensity_fun) -> HMCState:
    logdensity_fun_grad = jax.value_and_grad(logdensity_fun, has_aux=True)
    (den, aux), g = logdensity_fun_grad(pos)
    return HMCState(pos, den, aux, g)


def step(
    key,
    state: HMCState,
    logdensity_fun,
    mass,
    n_integration_steps,
    step_size,
) -> HMCState:
    """
    Step and update the state once.

    :param key: the random key, used to generate the initial momentum
    :param state: current state (pos, log density, aux, pos_grad)
    :param logdensity_fun: a callable that takes as input current position and
           returns a tuple (the log density, auxiliary data)
    :param mass: the kinematic mass diagonal, used to generate the initial
           momentum, and to compute the kinematic energy
    :param n_integration_steps: number of integration steps
    :param step_size: the step size
    :return: updated state (pos, log density, aux, pos_grad)
    """
    def body_fun(state, _x):
        pos, mom = state
        pos += step_size * mom
        g, aux = logdensity_grad(pos)
        mom -= step_size * (-g)
        return (pos, mom), aux

    def neg_hamiltonian(den, mom):
        """Compute the negative Hamiltonian given log density and momentum."""
        return den - jnp.dot(mom / mass, mom) / 2

    logdensity_fun_grad = jax.jit(
        jax.value_and_grad(logdensity_fun, has_aux=True))
    logdensity_grad = jax.grad(logdensity_fun, has_aux=True)
    key_accept, key_mom = random.split(key, 2)
    pos, den, aux, pos_grad = state
    mom = random.normal(key_mom, shape=pos.shape) * jnp.sqrt(mass)
    (pos_updated, mom_updated), _ = lax.scan(
        body_fun, (pos, mom - step_size / 2 * (-pos_grad)),
        None,
        length=n_integration_steps - 1)
    pos_updated += step_size * mom_updated
    (den_updated,
     aux_updated), pos_updated_grad = logdensity_fun_grad(pos_updated)
    mom_updated -= step_size / 2 * (-pos_updated_grad)
    a = jnp.minimum(
        1,
        jnp.exp(
            neg_hamiltonian(den_updated, mom_updated)
            - neg_hamiltonian(den, mom)))
    accept = random.uniform(key_accept, minval=0, maxval=1) <= a
    pos_updated = lax.select(accept, pos_updated, pos)
    aux_updated = lax.select(accept, aux_updated, aux)
    return HMCState(pos_updated, den_updated, aux_updated, pos_updated_grad)


@partial(
    jax.jit,
    static_argnames=(
        'logdensity_fun',
        'n_samples',
        'n_integration_steps',
    ),
)
def hmc(
    key,
    pos,
    logdensity_fun,
    mass,
    n_integration_steps,
    step_size,
    n_samples,
) -> HMCState:
    """
    A shortcut function to use HMC.

    :param key: the random key
    :param pos: the initial position
    :param logdensity_fun: the log density function, which takes as input
           current position, and returns a tuple (log density, auxiliary data)
    :param mass: the kinematic mass diagonal
    :param n_integration_steps: number of integration steps
    :param step_size: the step size
    :param n_samples: number of samples to draw
    :return: all historical states in a single tuple
    """
    def body_fun(state, key):
        state = step(key, state, logdensity_fun, mass, n_integration_steps,
                     step_size)
        return state, state

    keys = random.split(key, n_samples)
    state = init(pos, logdensity_fun)
    _, states = lax.scan(body_fun, state, keys)
    return states
