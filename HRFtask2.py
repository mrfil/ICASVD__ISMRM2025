
import numpy as np
from scipy.special import gamma
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def HRFtask2(onTime, N_tpt, TR, delay, ASL, plotTask):
    """
    Parameters:
    onTime : int
        Seconds stimuli was on/off for
    N_tpt : int
        Number of time points
    TR : float
        Repetition time (seconds)
    delay : float
        Delay in the HRF (seconds)
    ASL : int
        For ASL data, want control/tag pattern built into the task.
        Without ASL, use 0. For ASL control first, use 1. For ASL tag first, use 2.
    plotTask : bool
        Use to plot the task waveform
    
    Returns:
    task : numpy.ndarray
        Task waveform
    taskdiff : numpy.ndarray
        Derivative of task waveform
    """
    dt = 0.01
    t = np.arange(0, TR*N_tpt + dt, dt)
    tt = np.arange(TR, TR*N_tpt + TR, TR) + TR/2

    alpha, beta = 6, 1
    hrf = (t**(alpha-1) * np.exp(-t/beta)) / (beta**alpha * gamma(alpha))
    hrf = hrf / np.max(hrf)

    boxcar = np.zeros_like(t)
    boxcar[np.mod(t, onTime*2) > 20] = 1

    convolved_signal = np.convolve(boxcar, hrf)
    task = convolved_signal[:len(t)]

    # Shift by the delay
    task = np.pad(task, (int(delay/dt), 0))[:len(t)]
    
    # Interpolate to match time series samples
    interp_func = interp1d(t, task, kind='linear', fill_value='extrapolate')
    task = interp_func(tt)
    task = task / np.max(task)

    taskdiff = np.diff(task)
    taskdiff = np.append(taskdiff, 0)
    taskdiff = taskdiff / np.max(taskdiff)

    if ASL == 1:
        task = task * np.tile([1, -1], N_tpt//2)
        taskdiff = taskdiff * np.tile([1, -1], N_tpt//2)
    elif ASL == 2:
        task = task * np.tile([-1, 1], N_tpt//2)
        taskdiff = taskdiff * np.tile([-1, 1], N_tpt//2)

    if plotTask:
        plt.figure()
        plt.plot(tt, task, label='task')
        plt.plot(tt, taskdiff, label='task derivative')
        plt.xlabel('time (sec)')
        plt.title('task waveform')
        plt.legend()
        plt.show()

    return task, taskdiff

# Example usage:
# onTime = 20
# N_tpt = 40
# TR = 4
# delay = 6
# ASL = 2
# plotTask = True
# task, taskdiff = HRFtask2(onTime, N_tpt, TR, delay, ASL, plotTask)


