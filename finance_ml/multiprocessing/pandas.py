import pandas as pd

from .partition import linear_parts, nested_parts
from .utils import process_jobs


def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1,
                  linear_mols=True,
                  descend=False, **kwargs):
    """Return multiprocessed results

    Parameters
    ----------
    func: function object
    pd_obj: list
        pd_obj[0]: the name of parameters to be parallelized
        pd_obj[1]: parameters to be parallelized
    mp_batches: int
        The number of batches processed for each thread
    linear_mols: bool
        If True, use linear partition
        If False, use nested partition
    descend: bool
        The parameter for nested partitions
    kwargs: optional parameters of `func`

    Returns
    -------
    The same type as the output of func
    """
    if linear_mols:
        parts = linear_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches, descend)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]: parts[i]], 'func': func}
        job.update(kwargs)
        jobs.append(job)
    outputs = process_jobs(jobs, num_threads=num_threads)
    # You can use either of pd.Series or pd.DatFrame
    if isinstance(outputs[0], pd.Series):
        df = pd.Series()
    elif isinstance(outputs[0], pd.DataFrame):
        df = pd.DataFrame()
    else:
        return outputs
    # The case of multiple threads
    for output in outputs:
        df = df.append(output)
    df = df.sort_index()
    return df