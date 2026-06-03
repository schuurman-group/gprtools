"""
Lightweight timing registry for profiling main routines.

Both wall-clock and CPU time are tracked. Timings are *exclusive*
(self time): when a timed routine calls another timed routine, the
nested time is attributed to the callee, not the caller. A thin wrapper
around an expensive routine therefore reports near-zero time, while the
expensive routine reports the full cost.

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


# registry: name -> [wall_seconds, cpu_seconds, n_calls]
_registry = defaultdict(lambda: [0.0, 0.0, 0])

# stack of child-time accumulators, one [wall, cpu] entry per currently-
# active timed scope. Each entry sums the inclusive time of nested calls.
_stack = []


#
def _enter():
    """
    begin a timed scope; push a fresh child-time accumulator and
    return the (wall, cpu) start times
    """
    _stack.append([0.0, 0.0])
    return time.perf_counter(), time.process_time()


#
def _exit(name, t0):
    """
    end a timed scope, recording exclusive (self) wall and CPU time
    under name, and crediting this scope's inclusive time to its
    parent, if any
    """
    wall_t0, cpu_t0 = t0
    wall_elapsed = time.perf_counter() - wall_t0
    cpu_elapsed  = time.process_time() - cpu_t0

    child_wall, child_cpu = _stack.pop()

    entry = _registry[name]
    entry[0] += wall_elapsed - child_wall
    entry[1] += cpu_elapsed  - child_cpu
    entry[2] += 1

    # attribute our full (inclusive) time to the enclosing scope so its
    # exclusive time excludes the time spent here
    if _stack:
        _stack[-1][0] += wall_elapsed
        _stack[-1][1] += cpu_elapsed


#
def timed(func):
    """
    Decorator: accumulate exclusive wall and CPU time spent in func
    under its qualified name (e.g. 'BCM.gradient').
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
    Context manager: accumulate exclusive wall and CPU time for an
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
    return the raw registry as a dict:
    name -> (wall_seconds, cpu_seconds, n_calls)
    """
    return {name: (w, c, n) for name, (w, c, n) in _registry.items()}


#
def summary(sort='total', print_fraction=0.95):
    """
    print a table of accumulated exclusive (self) timings

    sort           : 'total' (wall, default), 'cpu', 'calls', 'mean'
                     (wall per call), or 'name'
    print_fraction : only show the most expensive routines that together
                     account for this fraction of the total wall time;
                     the remainder are collapsed into a single omitted
                     line (1.0 prints every routine)
    """
    if not _registry:
        print('timer: no timings recorded.')
        return

    rows  = [(name, w, c, n) for name, (w, c, n) in _registry.items()]
    grand_w = sum(r[1] for r in rows)
    grand_c = sum(r[2] for r in rows)

    # select the most expensive routines covering print_fraction of the
    # total wall time -- selection is always by descending wall time
    by_time = sorted(rows, key=lambda r: -r[1])
    kept    = []
    cum     = 0.
    if grand_w > 0.:
        thresh = print_fraction * grand_w
        for r in by_time:
            kept.append(r)
            cum += r[1]
            if cum >= thresh:
                break
    else:
        kept = list(by_time)
        cum  = grand_w

    n_omit = len(rows) - len(kept)
    w_omit = grand_w - cum

    # order the kept rows for display
    keys = {
        'total': lambda r: -r[1],
        'cpu':   lambda r: -r[2],
        'calls': lambda r: -r[3],
        'mean':  lambda r: -(r[1] / r[3] if r[3] else 0.),
        'name':  lambda r: r[0],
    }
    kept.sort(key=keys.get(sort, keys['total']))

    name_w = max(len('Routine'), max(len(r[0]) for r in kept))
    line   = '-' * (name_w + 69)

    print()
    print('Timing summary (exclusive / self time)')
    print(line)
    print(f"{'Routine':<{name_w}}  {'Calls':>8}  "
          f"{'Wall (s)':>12}  {'Wall %':>7}  "
          f"{'CPU (s)':>12}  {'CPU %':>7}  {'Mean (ms)':>12}")
    print(line)
    for name, w, c, n in kept:
        mean_ms = 1.e3 * w / n if n else 0.
        w_pct   = 100. * w / grand_w if grand_w > 0. else 0.
        c_pct   = 100. * c / grand_c if grand_c > 0. else 0.
        print(f"{name:<{name_w}}  {n:>8d}  "
              f"{w:>12.4f}  {w_pct:>7.2f}  "
              f"{c:>12.4f}  {c_pct:>7.2f}  {mean_ms:>12.3f}")
    print(line)
    if n_omit > 0:
        pct = 100. * w_omit / grand_w if grand_w > 0. else 0.
        print(f"({n_omit} routine(s) omitted: "
              f"{w_omit:.4f} s wall, {pct:.1f}% of total)")
    print(f"{'Total':<{name_w}}  {'':>8}  "
          f"{grand_w:>12.4f}  {'':>7}  {grand_c:>12.4f}")
    print()
