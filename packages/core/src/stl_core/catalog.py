"""Static catalog of generators: which queue owns each and where its code lives.

Modules are imported lazily so the catalog can be read in any environment.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field

from stl_core.generator import Generator


@dataclass(frozen=True)
class GeneratorInfo:
    name: str
    family: str
    queue: str
    module: str
    cpu_ok: bool = True
    params: dict = field(default_factory=dict)


_SC = "stl_synthcity.generators"
_SN = "stl_smartnoise.generators"
_ENTRIES = [
    GeneratorInfo("train_copy", "sanity_ceiling", "q.sdv", "stl_core.baselines"),
    GeneratorInfo("independent_marginals", "sanity_floor", "q.sdv", "stl_core.baselines"),
    GeneratorInfo("marginals", "sanity_floor", "q.synthcity", _SC),
    GeneratorInfo("gaussian_copula", "statistical", "q.sdv", "stl_sdv.generators"),
    GeneratorInfo("bayesian_network", "graphical", "q.synthcity", _SC),
    GeneratorInfo("ctgan", "deep", "q.synthcity", _SC),
    GeneratorInfo("tvae", "deep", "q.synthcity", _SC),
    GeneratorInfo("tabddpm", "diffusion", "q.synthcity", _SC, cpu_ok=False),
    GeneratorInfo("great", "llm", "q.synthcity", _SC, cpu_ok=False),
    # RQ2 epsilon sweep: smartnoise-synth MST, a vetted epsilon-DP synthesizer.
    GeneratorInfo("mst_eps0.1", "differential_privacy", "q.smartnoise", _SN,
                  params={"epsilon": 0.1}),
    GeneratorInfo("mst_eps1", "differential_privacy", "q.smartnoise", _SN,
                  params={"epsilon": 1.0}),
    GeneratorInfo("mst_eps10", "differential_privacy", "q.smartnoise", _SN,
                  params={"epsilon": 10.0}),
    # synthcity 0.2.12 PrivBayes is NOT epsilon-DP (numerics never binned, Laplace noise
    # scaled for probabilities but added to counts, zero cells unnoised): kept only as a
    # nominal-DP audit, never as an RQ2 data point.
    GeneratorInfo("privbayes_eps0.1", "nominal_dp_audit", "q.synthcity", _SC,
                  params={"epsilon": 0.1}),
    GeneratorInfo("privbayes_eps1", "nominal_dp_audit", "q.synthcity", _SC,
                  params={"epsilon": 1.0}),
    GeneratorInfo("privbayes_eps10", "nominal_dp_audit", "q.synthcity", _SC,
                  params={"epsilon": 10.0}),
]
CATALOG: dict[str, GeneratorInfo] = {e.name: e for e in _ENTRIES}


def is_available(name: str) -> bool:
    return importlib.util.find_spec(CATALOG[name].module.split(".")[0]) is not None


def load_generator(name: str, seed: int = 0, **overrides) -> Generator:
    if name not in CATALOG:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(CATALOG)}")
    info = CATALOG[name]
    if not is_available(name):
        raise ImportError(
            f"generator {name!r} needs module {info.module!r} (queue {info.queue}); "
            "run it in that worker's environment"
        )
    module = importlib.import_module(info.module)
    return module.build(name, seed=seed, **{**info.params, **overrides})
