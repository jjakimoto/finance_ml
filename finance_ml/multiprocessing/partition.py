import numpy as np


def linear_parts(num_atoms, num_threads):
    """Linear partitions

    Parameters
    ----------
    num_atoms: int
        The number of data points
    num_threads: int
        The number of partitions to split

    Returns
    -------
    array: indices of start and end
    """
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


def nested_parts(num_atoms, num_threads, descend=False):
    """Nested partitions

    Parameters
    ----------
    num_atoms: int
        The number of data points
    num_threads: int
        The number of partitions to split
    descent: bool, (default False)
        If True, the size of partitions are decreasing

    Returns
    -------
    array: indices of start and end
    """
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