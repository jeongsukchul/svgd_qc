"""Small optimizer factory shared by agent configurations."""

import optax


def make_optimizer(name, learning_rate):
    """Build one of the optimizers exposed by agent config files."""
    name = name.lower()
    if name == "adam":
        return optax.adam(learning_rate=learning_rate)
    if name == "kron":
        # JAX/Optax implementation of the PSGD Kron optimizer used by
        # https://github.com/roger-creus/stable-deep-rl-at-scale.
        from psgd_jax.kron import kron

        return kron(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name!r}; expected 'adam' or 'kron'")
