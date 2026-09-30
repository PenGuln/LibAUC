import logging
from typing import Dict, Any, Optional
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameter parsing helpers  (AutoMAX / ConfigSpace)
# ---------------------------------------------------------------------------

def parse_hyperparameters_from_dict(items: dict[str, Any]):
    r"""
    Convert a raw hyperparameter specification dict into a mapping of
    ``ConfigSpace`` :class:`~ConfigSpace.hyperparameters.Hyperparameter`
    objects.

    Each value in *items* may be:

    * a plain scalar (``int``, ``float``, ``str``) → :class:`~ConfigSpace.hyperparameters.Constant`
    * a 2-tuple ``(lower, upper)`` → :class:`~ConfigSpace.hyperparameters.UniformFloatHyperparameter`
      or :class:`~ConfigSpace.hyperparameters.UniformIntegerHyperparameter`
    * a ``list`` of choices → :class:`~ConfigSpace.hyperparameters.CategoricalHyperparameter`
    * a ``dict`` with a mandatory ``"val"`` key and optional ``"default"`` /
      ``"log"`` keys (extended AutoMAX format)
    * a ``dict`` with a ``"range"`` key, e.g. ``{"range": [lower, upper]}`` →
      always a continuous/integer bound, regardless of whether ``[lower,
      upper]`` came in as a Python tuple or list. This is the form to use
      from YAML configs, since YAML sequences always load as ``list`` and
      can't be told apart from a 2-choice categorical via ``"val"``.
    * an already-constructed :class:`~ConfigSpace.hyperparameters.Hyperparameter`

    Args:
        items (dict[str, Any]): Mapping of hyperparameter name to its
            specification.

    Returns:
        dict[str, Hyperparameter]: Parsed hyperparameters keyed by name.

    Raises:
        ValueError: If a specification is malformed or an unsupported type is
            encountered.

    Example::

        >>> specs = {
        ...     "lr":      {"range": [1e-4, 0.1], "default": 0.1, "log": True},
        ...     "margin":  {"val": [0.6, 0.8, 1.0], "default": 1.0},
        ...     "dropout": 0.5,
        ... }
        >>> hps = parse_hyperparameters_from_dict(specs)
    """
    ret = {}
    for name, hp in items.items():
        # ── Extended dict format {"range": [lower, upper], ...} ──
        # Explicit bound spec: unambiguous continuous/integer range, usable
        # from YAML (where "val": [lo, hi] would otherwise be read as a
        # 2-item categorical choice, since YAML has no tuple type).
        if isinstance(hp, dict) and "range" in hp:
            bounds = hp["range"]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    f"'{name}' range must be a 2-element [lower, upper] bound, got {bounds}"
                )
            lower, upper = bounds
            logarithmic = hp.get("log", False)
            default = hp.get("default", lower)
            if isinstance(lower, float) or isinstance(upper, float):
                ret[name] = UniformFloatHyperparameter(
                    name=name,
                    lower=lower,
                    upper=upper,
                    default_value=default,
                    log=logarithmic,
                    meta=None,
                )
            else:
                ret[name] = UniformIntegerHyperparameter(
                    name=name,
                    lower=lower,
                    upper=upper,
                    default_value=default,
                    log=logarithmic,
                    meta=None,
                )
            continue

        # ── Extended dict format {"val": ..., "default": ..., "log": ...} ──
        if isinstance(hp, dict):
            val = hp.get("val")
            if val is None:
                raise ValueError(f"Can't find the key 'val' in {hp}")

            logarithmic = hp.get("log", False)
            if isinstance(val, set) and logarithmic:
                logger.warning(
                    "Both categorical and logarithmic are set to True; "
                    "ignoring the logarithmic flag."
                )

            if isinstance(val, (int, str, float)):
                ret[name] = Constant(name, val)
            elif isinstance(val, tuple):
                if len(val) != 2:
                    raise ValueError(
                        f"'{name}' must be (lower, upper) bound, got {val}"
                    )
                default = hp.get("default", val[0])
                if isinstance(val[0], float) or isinstance(val[1], float):
                    ret[name] = UniformFloatHyperparameter(
                        name=name,
                        lower=val[0],
                        upper=val[1],
                        default_value=default,
                        log=logarithmic,
                        meta=None,
                    )
                else:
                    ret[name] = UniformIntegerHyperparameter(
                        name=name,
                        lower=val[0],
                        upper=val[1],
                        log=logarithmic,
                        default_value=default,
                        meta=None,
                    )
            elif isinstance(val, list):
                if len(val) == 0:
                    raise ValueError(
                        f"Can't have empty list for categorical '{name}'"
                    )
                default = hp.get("default", val[0])
                ret[name] = CategoricalHyperparameter(
                    name=name,
                    choices=val,
                    default_value=default,
                    weights=None,
                    meta=None,
                )
            else:
                raise ValueError(f"Unknown value '{val}' for '{name}'")

        # ── Already a Hyperparameter ─────────────────────────────────────────
        elif isinstance(hp, Hyperparameter):
            ret[name] = hp

        # ── Bare tuple → bounds ──────────────────────────────────────────────
        elif isinstance(hp, tuple):
            if len(hp) != 2:
                raise ValueError(
                    f"'{name}' must be (lower, upper) bound, got {hp}"
                )
            lower, upper = hp
            if isinstance(lower, float):
                ret[name] = UniformFloatHyperparameter(name, lower, upper)
            else:
                ret[name] = UniformIntegerHyperparameter(name, lower, upper)

        # ── Bare list → categorical ──────────────────────────────────────────
        elif isinstance(hp, list):
            if len(hp) == 0:
                raise ValueError(
                    f"Can't have empty list for categorical '{name}'"
                )
            ret[name] = CategoricalHyperparameter(name, hp)

        # ── Bare scalar → constant ───────────────────────────────────────────
        elif isinstance(hp, (int, str, float)):
            ret[name] = Constant(name, hp)

        else:
            raise ValueError(f"Unknown value '{hp}' for '{name}'")

    return ret