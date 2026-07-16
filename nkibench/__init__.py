"""
NKIGen-Bench task registry and loader.

Provides the NKIBenchTask dataclass and functions to load tasks by level.
Tasks are organized into three difficulty levels:
  - Level 1: Single operations (100 tasks)
  - Level 2: Fused operations (100 tasks)
  - Level 3: Full model components (50 tasks)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NKIBenchTask:
    """A single NKIGen-Bench evaluation task.

    Attributes:
        task_id: Unique identifier (e.g., "L1_matmul_001").
        name: Human-readable task name.
        level: Difficulty level (1, 2, or 3).
        category: Task category (e.g., "matrix_ops", "activations", "attention").
        model_class_code: Python source code string defining a PyTorch nn.Module
            subclass named ``Model`` that implements the baseline operation.
        get_inputs_code: Python source code string defining a function
            ``get_inputs() -> list[torch.Tensor]`` that returns the input tensors
            for the model.
        nki_metadata: Optional hints for NKI kernel generation, such as
            ``engine_hints`` (preferred engine), ``tile_suggestions``
            (recommended tile sizes), and ``memory_layout`` notes.
        expected_speedup_range: Tuple ``(low, high)`` giving the expected
            speedup range for a well-optimized NKI kernel over eager PyTorch.
    """

    task_id: str
    name: str
    level: int
    category: str
    model_class_code: str
    get_inputs_code: str
    nki_metadata: Dict = field(default_factory=dict)
    expected_speedup_range: Tuple[float, float] = (1.0, 5.0)

    def __post_init__(self):
        if self.level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, or 3, got {self.level}")


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, NKIBenchTask] = {}


def _register_tasks(tasks: List[NKIBenchTask]) -> None:
    """Register a list of tasks into the global registry."""
    for task in tasks:
        if task.task_id in TASK_REGISTRY:
            raise ValueError(f"Duplicate task_id: {task.task_id}")
        TASK_REGISTRY[task.task_id] = task


def load_tasks(level: Optional[int] = None) -> List[NKIBenchTask]:
    """Load NKIGen-Bench tasks, optionally filtered by difficulty level.

    Args:
        level: If provided, only return tasks of this level (1, 2, or 3).
            If ``None``, return all tasks.

    Returns:
        List of :class:`NKIBenchTask` instances, sorted by task_id.
    """
    # Lazy-import level modules to populate the registry on first call.
    _ensure_registry_populated()

    if level is not None:
        if level not in (1, 2, 3):
            raise ValueError(f"level must be 1, 2, or 3, got {level}")
        return sorted(
            [t for t in TASK_REGISTRY.values() if t.level == level],
            key=lambda t: t.task_id,
        )
    return sorted(TASK_REGISTRY.values(), key=lambda t: t.task_id)


_registry_populated = False


def _ensure_registry_populated() -> None:
    """Populate the registry from level modules (idempotent)."""
    global _registry_populated
    if _registry_populated:
        return

    from .level1 import get_level1_tasks
    from .level2 import get_level2_tasks
    from .level3 import get_level3_tasks

    _register_tasks(get_level1_tasks())
    _register_tasks(get_level2_tasks())
    _register_tasks(get_level3_tasks())

    _registry_populated = True
