"""
Industry Profiles — future-proof multi-industry scaffolding.

Each industry plugs in via a single `IndustryProfile` object that carries
everything downstream tools need:

  - Per-department economics (base salary, fill days, replacement
    multiplier, safety criticality, hiring difficulty).
  - Intervention catalog (retention levers, cost/effect parameters).
  - Cohort thresholds (age / tenure / comp / engagement bands).
  - Cross-department adjacency matrix (for internal mobility).
  - Skills taxonomy + canonical role list.
  - Tier-1 employer list (pedigree detection).
  - Skill-cluster definitions (domain adjacency).
  - Synonym groups (acronym/phrasing equivalences).
  - Interview-question templates.

Swap the active industry via `set_industry(profile)`. All downstream
forecast_tools / talent_tools consult the active profile for their
domain constants. Today: energy. Tomorrow: finance, healthcare, tech,
manufacturing, retail, public sector — one plugin file each.

Architecture: each industry is a `dataclass` that extends `IndustryBase`
so adding a new vertical is a 200-line file, not a fork.

Reference: ESCO + O*NET cross-vertical frameworks provide pre-built
skill/occupation taxonomies that map cleanly onto this plugin interface.
"""

from .base import IndustryBase, set_industry, get_industry
from .energy import ENERGY
from .finance import FINANCE

# Registry — lookup by short name for the dashboard selector.
REGISTRY = {
    "energy":  ENERGY,
    "finance": FINANCE,
}

# Default = energy (the vertical this project validates first).
set_industry(ENERGY)

__all__ = [
    "IndustryBase", "ENERGY", "FINANCE", "REGISTRY",
    "set_industry", "get_industry",
]
