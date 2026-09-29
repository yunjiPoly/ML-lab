"""Resolver: turns OCR readings into a database-backed :class:`ResolutionResult`.

Public API::

    from app.services.resolver import Resolver, NameIndex, NameMatch
    from app.services.resolver import name_similarity, set_code_similarity, closest_set_code
"""

from app.services.resolver.matching import closest_set_code, name_similarity, set_code_similarity
from app.services.resolver.name_index import NameIndex, NameMatch
from app.services.resolver.resolver import Resolver

__all__ = [
    "Resolver",
    "NameIndex",
    "NameMatch",
    "name_similarity",
    "set_code_similarity",
    "closest_set_code",
]
