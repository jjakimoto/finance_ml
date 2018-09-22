import time
from datetime import datetime
import sys
from copy import deepcopy
import multiprocessing as mp


def expand_call(kwargs):
    """Execute function from dictionary input"""
    func = kwargs['func']
    del kwargs['func']
    out = func(**kwargs)
    return out


def report_progress(job_idx, num_jobs, time0, task):
    """Report progress to system output"""
    msg = [float(job_idx) / num_jobs, (time.time() - time0) / 60.]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(datetime.fromtimestamp(time.time()))
    msg_ = time_stamp + ' ' + str(
        round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
           str(round(msg[1], 2)) + ' minutes. Remaining ' + str(
        round(msg[2], 2)) + ' minutes.'
    if job_idx < num_jobs:
        sys.stderr.write(msg_ + '\r')
    else:
        sys.stderr.write(msg_ + '\n')


def process_jobs(jobs, task=None, num_threads=24):
    """Execute parallelized jobs

    Parameters
    ----------
    jobs: list(dict)
        Each element contains `function` and its parameters
    task: str, optional
        The name of task. If not specified, function name is used
    num_threads, (default 24)
        The number of threads for parallelization

    Returns
    -------
    List: each element is results of each part
    """
    if task is None:
        task = jobs[0]['func'].__name__
    out = []
    if num_threads > 1:
        pool = mp.Pool(processes=num_threads)
        outputs = pool.imap_unordered(expand_call, jobs)
        time0 = time.time()
        # Execute programs here
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            report_progress(i, len(jobs), time0, task)
        pool.close()
        pool.join()
    else:
        for job in jobs:
            job = deepcopy(job)
            func = job['func']
            del job['func']
            out_ = func(**job)
            out.append(out_)
    return out