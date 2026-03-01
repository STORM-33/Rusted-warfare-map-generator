from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


class WizardStep(IntEnum):
    COASTLINE = 0
    HILLS = 1
    HEIGHT_OCEAN = 2
    COMMAND_CENTERS = 3
    RESOURCES = 4
    FINALIZE = 5


@dataclass
class WizardState:
    # --- Inputs (set before step 1) ---
    initial_matrix: Optional[list] = None
    height: int = 160
    width: int = 160
    mirroring: str = "vertical"
    pattern: int = 1
    num_height_levels: int = 7
    num_ocean_levels: int = 3
    num_command_centers: int = 4
    num_resource_pulls: int = 12
    output_path: str = ""

    # --- Step 1: Coastline ---
    randomized_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    coastline_height_map: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Step 2: Hills ---
    wall_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Step 3: Height/Ocean ---
    perlin_seed: Optional[int] = None
    perlin_map: Optional[np.ndarray] = field(default=None, repr=False)
    height_map: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Step 4: Command Centers ---
    units_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    cc_positions: List[Tuple[int, int]] = field(default_factory=list)
    # Groups for undo: each entry is a list of (row, col) positions placed together
    cc_groups: List[List[Tuple[int, int]]] = field(default_factory=list)

    # --- Step 5: Resources ---
    items_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    resource_positions: List[Tuple[int, int]] = field(default_factory=list)
    resource_groups: List[List[Tuple[int, int]]] = field(default_factory=list)

    # --- Step 6: Finalize ---
    id_matrix: Optional[np.ndarray] = field(default=None, repr=False)

    # --- Tracking ---
    current_step: WizardStep = WizardStep.COASTLINE
    completed_step: int = -1  # -1 means no step completed yet

    def invalidate_from(self, step: WizardStep):
        """Clear data from the given step onward."""
        if step <= WizardStep.COASTLINE:
            self.randomized_matrix = None
            self.coastline_height_map = None
        if step <= WizardStep.HILLS:
            self.wall_matrix = None
        if step <= WizardStep.HEIGHT_OCEAN:
            self.perlin_seed = None
            self.perlin_map = None
            self.height_map = None
        if step <= WizardStep.COMMAND_CENTERS:
            self.units_matrix = None
            self.cc_positions = []
            self.cc_groups = []
        if step <= WizardStep.RESOURCES:
            self.items_matrix = None
            self.resource_positions = []
            self.resource_groups = []
        if step <= WizardStep.FINALIZE:
            self.id_matrix = None

        # Reset completed_step to one before the invalidated step
        self.completed_step = min(self.completed_step, int(step) - 1)
