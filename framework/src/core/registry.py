"""
Shared helpers for small in-process registries.

The runtime, compiler, and monitor registries all expose a canonical name plus
aliases. Registration should be atomic: either every key belongs to the new
entry, or the registry stays unchanged.
"""

from typing import Iterable, MutableMapping, TypeVar


EntryT = TypeVar("EntryT")


class RegistryCollisionError(ValueError):
    """Raised when a registry key is already owned by another entry."""


def normalize_registry_key(value: str, kind: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{kind} registry key must be a string")
    key = value.strip().lower()
    if not key:
        raise ValueError(f"{kind} registry key must not be empty")
    return key


def register_entry_keys(
    registry: MutableMapping[str, EntryT],
    entry: EntryT,
    keys: Iterable[str],
    kind: str,
) -> None:
    normalized_keys: list[str] = []
    seen: set[str] = set()

    for raw_key in keys:
        key = normalize_registry_key(raw_key, kind)
        if key in seen:
            continue
        normalized_keys.append(key)
        seen.add(key)

    for key in normalized_keys:
        existing = registry.get(key)
        if existing is None or existing == entry:
            continue
        existing_name = getattr(existing, "name", repr(existing))
        entry_name = getattr(entry, "name", repr(entry))
        raise RegistryCollisionError(
            f"{kind} registry key '{key}' already belongs to "
            f"'{existing_name}', cannot register '{entry_name}'"
        )

    for key in normalized_keys:
        registry[key] = entry
