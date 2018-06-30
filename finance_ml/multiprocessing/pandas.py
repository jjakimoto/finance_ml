import pandas as pd


from .partition import lin_parts, nested_parts
from .utils import process_jobs


def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True,
                  descend=False, **kwargs):
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches, descend)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]: parts[i]], 'func': func}
        job.update(kwargs)
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    if isinstance(out[0], pd.Series):
        df0 = pd.Series()
    elif isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    else:
        return out

    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0