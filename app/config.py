from dataclasses import dataclass, field


@dataclass
class GridParams:
    pitch: float = 2.0
    padding_mul: float = 3.0
    max_dim: int = 260
    band_mul: float = 0.75


@dataclass
class OctreeParams:
    enabled: bool = False
    max_level: int = 6
    balance_max_iters: int = 20


@dataclass
class ShrinkParams:
    iters: int = 25
    step: float = 0.35
    constraint_mul: float = 0.35
    lap_iters_per_step: int = 1
    lap_relax: float = 0.15


@dataclass
class AppParams:
    grid: GridParams = field(default_factory=GridParams)
    octree: OctreeParams = field(default_factory=OctreeParams)
    shrink: ShrinkParams = field(default_factory=ShrinkParams)