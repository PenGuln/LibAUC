from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Hyperparameter

class autopartial:
    r"""
    A ``functools.partial``-like wrapper that accepts
    :class:`~ConfigSpace.hyperparameters.Hyperparameter` arguments and builds
    a :class:`~ConfigSpace.ConfigurationSpace` covering all of them.

    When called with a ``Configuration`` object (and an optional prefix
    string), it resolves every hyperparameter to its sampled value and
    invokes the wrapped callable.

    Args:
        autofunc (callable): The function or class to wrap.  May itself be an
            ``autopartial`` instance, in which case the configuration spaces
            are merged.
        *args: Positional hyperparameters — each must be a
            :class:`~ConfigSpace.hyperparameters.Hyperparameter` or an
            ``autopartial``.
        **kwds: Keyword hyperparameters — same constraint as *args*.

    Raises:
        TypeError: If *autofunc* is not callable.
        ValueError: If any argument is not a
            :class:`~ConfigSpace.hyperparameters.Hyperparameter` or
            ``autopartial``.

    Example::

        >>> from ConfigSpace.hyperparameters import UniformFloatHyperparameter
        >>> lr_hp = UniformFloatHyperparameter("lr", 1e-4, 0.1, log=True)
        >>> wrapped = autopartial(MyOptimizer, lr=lr_hp)
        >>> optimizer = wrapped(some_configuration)
    """

    __slots__ = "autofunc", "args", "kwds", "cs", "__dict__", "__weakref__"

    def __new__(cls, autofunc, /, *args, **kwds):
        if not callable(autofunc):
            raise TypeError("the first argument must be callable")

        cs = ConfigurationSpace(seed=0)

        for i, arg in enumerate(args):
            if isinstance(arg, Hyperparameter):
                cs.add_hyperparameter(arg)
            elif hasattr(arg, "autofunc"):
                cs.add_configuration_space(
                    prefix=f"{i}", configuration_space=arg.cs, delimiter=":"
                )
            else:
                raise ValueError(
                    "autopartial arguments must be Hyperparameter or autopartial instances."
                )

        for k, v in kwds.items():
            if isinstance(v, Hyperparameter):
                cs.add_hyperparameter(v)
            elif hasattr(v, "autofunc"):
                cs.add_configuration_space(
                    prefix=f"{k}", configuration_space=v.cs, delimiter=":"
                )
            else:
                raise ValueError(
                    "autopartial keyword arguments must be Hyperparameter or autopartial instances."
                )

        if hasattr(autofunc, "autofunc"):
            args = autofunc.args + args
            kwds = {**autofunc.kwds, **kwds}
            cs.add_configuration_space(
                prefix="", configuration_space=autofunc.cs, delimiter=""
            )
            autofunc = autofunc.autofunc

        self = super(autopartial, cls).__new__(cls)
        self.autofunc = autofunc
        self.args     = args
        self.kwds     = kwds
        self.cs       = cs
        return self

    def __reduce__(self):
        # ``__new__`` requires the wrapped callable, so the default
        # slot-based pickling can't rebuild an instance. Restore the
        # attributes directly; needed to ship the trainer to spawned
        # trial processes (see :mod:`libauc.auto.parallel`).
        return _restore_autopartial, (self.autofunc, self.args, self.kwds, self.cs)

    def resolve(self, space, prefix=""):
        """The ``(args, kwargs)`` this instance would call ``autofunc`` with for *space*.

        Lets a caller adjust a resolved argument (e.g. the seed inside
        ``train_args``) before calling ``self.autofunc(*args, **kwargs)``.
        """
        new_args = []
        new_kwds = {}
        for i, arg in enumerate(self.args):
            if isinstance(arg, Hyperparameter):
                new_args.append(space[prefix + arg.name])
            else:
                new_args.append(self.args[i](space, prefix + f"{i}:"))
        for k, v in self.kwds.items():
            if isinstance(v, Hyperparameter):
                new_kwds[k] = space[prefix + v.name]
            else:
                new_kwds[k] = self.kwds[k](space, prefix + f"{k}:")
        return new_args, new_kwds

    def __call__(self, space, prefix=""):
        new_args, new_kwds = self.resolve(space, prefix)
        return self.autofunc(*new_args, **new_kwds)


def _restore_autopartial(autofunc, args, kwds, cs):
    self = object.__new__(autopartial)
    self.autofunc = autofunc
    self.args     = args
    self.kwds     = kwds
    self.cs       = cs
    return self
