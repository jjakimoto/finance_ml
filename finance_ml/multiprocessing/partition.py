import numpy as np


def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, descend=False):
    parts = [0]
    num_threads = min(num_threads, num_atoms)
    for num in range(num_threads):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads)
        part = 0.5 * (-1 + np.sqrt(part))
        parts.append(part)
    if descend:
        # Computational decreases as index increases
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    parts = np.round(parts).astype(int)
    return parts