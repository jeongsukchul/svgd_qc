"""Small optimizer factory shared by agent configurations."""

import optax


def make_optimizer(name, learning_rate, eps=1e-8):
    """Build one of the optimizers exposed by agent config files."""
    name = name.lower()
    if name == "adam":
        return optax.adam(learning_rate=learning_rate, eps=eps)
    if name == "kron":
        # JAX/Optax implementation of the PSGD Kron optimizer used by
        # https://github.com/roger-creus/stable-deep-rl-at-scale.
        from psgd_jax.kron import kron

        return kron(learning_rate=learning_rate)
    raise ValueError(f"Unsupported optimizer: {name!r}; expected 'adam' or 'kron'")


def make_module_optimizer(name, learning_rate, eps=1e-8, module_eps=None):
    """Optimizer with a different Adam eps for selected top-level param modules.

    ``module_eps`` maps a substring of the top-level module key (e.g.
    "actor_drift") to the eps used for that module; all other modules use
    ``eps``.  Motivation: the drift-BC force normalisation pins its gradient
    magnitude, so the decoder benefits from a large eps, but a *global* large
    eps also damps the critic and latent actor (measured slowdown at 1e-2).
    """
    if name.lower() != "adam" or not module_eps:
        return make_optimizer(name, learning_rate, eps=eps)
    import optax as _optax

    transforms = {"rest": _optax.adam(learning_rate=learning_rate, eps=eps)}
    for i, (_, e) in enumerate(module_eps.items()):
        transforms[f"m{i}"] = _optax.adam(learning_rate=learning_rate, eps=e)
    keys = list(module_eps.keys())

    def label(params):
        out = {}
        for k in params:
            lab = "rest"
            for i, sub in enumerate(keys):
                if sub in k:
                    lab = f"m{i}"
                    break
            out[k] = lab
        return out

    return _optax.multi_transform(transforms, param_labels=label)
