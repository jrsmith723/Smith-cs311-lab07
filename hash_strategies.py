"""
Lab 7: The Collision Resolver -- starter.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()  # sentinel marking a deleted open-addressing slot


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert, or update in place if `key` already exists. Resize (double + rehash) once load factor > 0.75."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]

        while current is not None:
            if current.key == key:
                current.value = value
                return
            current = current.next

        new_node = _ChainNode(key, value)
        new_node.next = self._buckets[index]
        self._buckets[index] = new_node
        self._count += 1

        if self._count / len(self._buckets) > 0.75:
            old_buckets = self._buckets
            self._buckets = [None] * (len(old_buckets) * 2)

            for node in old_buckets:
                current = node

                while current is not None:
                    next_node = current.next

                    index = hash(current.key) % len(self._buckets)
                    current.next = self._buckets[index]
                    self._buckets[index] = current

                    current = next_node

    def get(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]

        while current is not None:
            if current.key == key:
                return current.value
            current = current.next

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key`. Raise KeyError if missing."""
        index = hash(key) % len(self._buckets)

        current = self._buckets[index]
        previous = None

        while current is not None:
            if current.key == key:
                if previous is None:
                    self._buckets[index] = current.next
                else:
                    previous.next = current.next

                self._count -= 1
                return

            previous = current
            current = current.next

        raise KeyError(key)


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (double + rehash) once load factor > 0.7."""
        if (self._count + 1) / len(self._keys) > 0.7:
            old_keys = self._keys
            old_values = self._values

            self._keys = [None] * (len(old_keys) * 2)
            self._values = [None] * (len(old_values) * 2)
            self._count = 0

            for i in range(len(old_keys)):
                if old_keys[i] is not None and old_keys[i] is not _TOMBSTONE:
                    self.insert(old_keys[i], old_values[i])  # type: ignore

        index = hash(key) % len(self._keys)
        first_tombstone = None

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                if first_tombstone is not None:
                    index = first_tombstone

                self._keys[index] = key
                self._values[index] = value
                self._count += 1
                return

            if current_key is _TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = index

            elif current_key == key:
                self._values[index] = value
                return

            index = (index + 1) % len(self._keys)

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        index = hash(key) % len(self._keys)

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                return self._values[index]  # type: ignore

            index = (index + 1) % len(self._keys)

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone (not None) so later probes don't stop early. Raise KeyError if missing."""
        index = hash(key) % len(self._keys)

        for _ in range(len(self._keys)):
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                self._keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return

            index = (index + 1) % len(self._keys)

        raise KeyError(key)


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.

    Pitfall to design around: with a power-of-2 table size, the probe
    sequence (idx + i^2) mod size does NOT reach every slot -- it can
    cycle through only about half of them, so the table can appear
    "full" and raise/loop forever even though empty slots exist
    elsewhere. Two standard fixes, pick one:
      (a) use a PRIME table size (so the quadratic sequence covers all
          slots whenever load factor < 1), or
      (b) resize proactively -- check load factor BEFORE attempting an
          insert's probe sequence, not only after a successful insert.
    Using both is safest.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (grow + rehash) once load factor > 0.7 -- see the pitfall note above."""
        if (self._count + 1) / len(self._keys) > 0.7:
            old_keys = self._keys
            old_values = self._values

            new_size = len(old_keys) * 2 + 1
            self._keys = [None] * new_size
            self._values = [None] * new_size
            self._count = 0

            for i in range(len(old_keys)):
                if old_keys[i] is not None and old_keys[i] is not _TOMBSTONE:
                    self.insert(old_keys[i], old_values[i])  # type: ignore

        start = hash(key) % len(self._keys)
        first_tombstone = None

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                if first_tombstone is not None:
                    index = first_tombstone

                self._keys[index] = key
                self._values[index] = value
                self._count += 1
                return

            if current_key is _TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = index

            elif current_key == key:
                self._values[index] = value
                return

        raise RuntimeError("Hash table is full")

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        start = hash(key) % len(self._keys)

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                return self._values[index]  # type: ignore

        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone. Raise KeyError if missing."""
        start = hash(key) % len(self._keys)

        for i in range(len(self._keys)):
            index = (start + i * i) % len(self._keys)
            current_key = self._keys[index]

            if current_key is None:
                raise KeyError(key)

            if current_key is not _TOMBSTONE and current_key == key:
                self._keys[index] = _TOMBSTONE
                self._values[index] = None
                self._count -= 1
                return

        raise KeyError(key)
