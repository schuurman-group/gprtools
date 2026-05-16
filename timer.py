"""
Lightweight wall-clock timing registry for profiling main routines.

Timings are *exclusive* (self time): when a timed routine calls another
timed routine, the nested time is attributed to the callee, not the
caller. A thin wrapper around an expensive routine therefore reports
near-zero time, while the expensive routine reports the full cost.

Usage:
    import timer as timer

    @timer.timed                      # decorate a routine
    def my_routine(...): ...

    with timer.region('block name'):  # time an arbitrary code block
        ...

    timer.summary()                   # print accumulated timings
    timer.reset()                     # clear the registry
"""
import time
import functools
from collections import defaultdict


# registry: name -> [exclusive_seconds, n_calls]
_registry = defaultdict(lambda: [0.0, 0])

# stack of child-time accumulators, one entry per currently-active
# timed scope. Each entry sums the inclusive time of nested timed calls.
_stack = []


#
def _enter():
    """
    begin a timed scope; push a fresh child-time accumulator and
    return the start time
    """
    _stack.append(0.0)
    return time.perf_counter()


#
def _exit(name, t0):
    """
    end a timed scope, recording exclusive (self) time under name and
    crediting this scope's inclusive time to its parent, if any
    """
    elapsed   = time.perf_counter() - t0
    child     = _stack.pop()
    exclusive = elapsed - child

    entry = _registry[name]
    entry[0] += exclusive
    entry[1] += 1

    # attribute our full (inclusive) time to the enclosing scope so its
    # exclusive time excludes the time spent here
    if _stack:
        _stack[-1] += elapsed


#
def timed(func):
    """
    Decorator: accumulate exclusive wall-clock time spent in func under
    its qualified name (e.g. 'BCM.gradient').
    """
    name = func.__qualname__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = _enter()
        try:
            return func(*args, **kwargs)
        finally:
            _exit(name, t0)

    return wrapper


#
class region:
    """
    Context manager: accumulate exclusive wall-clock time for an
    arbitrary code block under the supplied name.
    """
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self._t0 = _enter()
        return self

    def __exit__(self, *exc):
        _exit(self.name, self._t0)
        return False


#
def reset():
    """
    clear all accumulated timings
    """
    _registry.clear()
    _stack.clear()


#
def timings():
    """
    return the raw registry as a dict: name -> (exclusive_seconds, n_calls)
    """
    return {name: (tot, n) for name, (tot, n) in _registry.items()}


#
def summary(sort='total', print_fraction=0.95):
    """
    print a table of accumulated exclusive (self) timings

    sort           : 'total' (default), 'calls', 'mean', or 'name'
    print_fraction : only show the most expensive routines that together
                     account for this fraction of the total time; the
                     remainder are collapsed into a single omitted line
                     (1.0 prints every routine)
    """
    if not _registry:
        print('timer: no timings recorded.')
        return

    rows  = [(name, tot, n) for name, (tot, n) in _registry.items()]
    grand = sum(tot for _, tot, _ in rows)

    # select the most expensive routines covering print_fraction of the
    # total time -- selection is always by descending self time
    by_time = sorted(rows, key=lambda r: -r[1])
    kept    = []
    cum     = 0.
    if grand > 0.:
        thresh = print_fraction * grand
        for r in by_time:
            kept.append(r)
            cum += r[1]
            if cum >= thresh:
                break
    else:
        kept = list(by_time)
        cum  = grand

    n_omit = len(rows) - len(kept)
    t_omit = grand - cum

    # order the kept rows for display
    keys = {
        'total': lambda r: -r[1],
        'calls': lambda r: -r[2],
        'mean':  lambda r: -(r[1] / r[2] if r[2] else 0.),
        'name':  lambda r: r[0],
    }
    kept.sort(key=keys.get(sort, keys['total']))

    name_w = max(len('Routine'), max(len(r[0]) for r in kept))
    line   = '-' * (name_w + 40)

    print()
    print('Timing summary (exclusive / self time)')
    print(line)
    print(f"{'Routine':<{name_w}}  {'Calls':>8}  "
          f"{'Self (s)':>13}  {'Mean (ms)':>13}")
    print(line)
    for name, tot, n in kept:
        mean_ms = 1.e3 * tot / n if n else 0.
        print(f"{name:<{name_w}}  {n:>8d}  {tot:>13.4f}  {mean_ms:>13.3f}")
    print(line)
    if n_omit > 0:
        pct = 100. * t_omit / grand if grand > 0. else 0.
        print(f"({n_omit} routine(s) omitted: "
              f"{t_omit:.4f} s, {pct:.1f}% of total)")
    print(f"{'Total':<{name_w}}  {'':>8}  {grand:>13.4f}")
    print()
