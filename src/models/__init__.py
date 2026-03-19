from .smmile import SMMILe
from .attention import GatedAttention
from .instance_refinement import InstanceRefinement
from .nic import NICLayer

__all__ = ["SMMILe", "GatedAttention", "InstanceRefinement", "NICLayer"]

# Convenience re-export for pseudo-label selection used in training scripts
select_pseudo_labels = InstanceRefinement.select_pseudo_labels
