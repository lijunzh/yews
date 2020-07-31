import numpy as np
import pandas as pd
from yews.cpic import pick

def detects2table(results_dict, wl, g, include_all=False, starttime=None):
    '''
    Converts dictionary of results from detect() function to pandas DataFrame
    object. Columns include starttime, endtime, p probability and s probability
    for each window. If include_all==True, all windows will be included in the
    table, (not just those detected probability > threshold). Default is False.
    If starttime argument is specified, the starttime and endtime columns will
    contain datetime strings. Else, these columns contain values which are the
    number of seconds since the start of the array.

    Inputs:
        results_dict: dictionary of results from detect function
        detected_windows_only: Boolean
        starttime: obspy UTCDateTime object

    Output:
        df: pandas data frame with probabilities of detection for each window
    '''
    data = []
    cols = ('window start', 'window end', 'p prob', 's prob')
    for i in range(len(results_dict['detect_p'])):
        if results_dict['detect_p'][i] or results_dict['detect_s'][i] \
        or include_all:
            # log row in data table
            if starttime:
                window_start = str(starttime + i*g)   # UTCDateTime object
                window_end = str(starttime + i*g + wl)
            else:
                window_start = i*g   # time in seconds since start of array
                window_end = window_start + wl
            p_prob = results_dict['detect_p'][i]
            s_prob = results_dict['detect_s'][i]
            row_entry = (window_start, window_end, p_prob, s_prob)
            data.append(dict(zip(cols, row_entry)))
    df = pd.DataFrame(data)
    df = df[list(cols)]
    return df

def find_runs_with_gaps(results_dict, max_gap):
    '''
    Find runs within results_dict from detect function where either the
    detection probability for p or s is above the probability threshold,
    allowing for max_gap 0's in between detected windows.

    Inputs:
        results_dict: dictionary of results from yews detect function
        max_gap: max number of consecutive 0's allowable within runs

    Output:
        run_indices: list of pairs describing start and end of run windows
    '''
    scan_for_start = True
    zero_count = 0
    run_indices = []
    for i in range(len(results_dict['detect_p'])):
        if scan_for_start:
            if results_dict['detect_p'][i] or results_dict['detect_s'][i]:
                start_index = i
                most_recent_nonzero = i
                scan_for_start = False
        else:
            if results_dict['detect_p'][i] or results_dict['detect_s'][i]:
                most_recent_nonzero = i
                zero_count = 0
            else:
                if zero_count == max_gap:
                    run_indices.append([start_index, most_recent_nonzero])
                    zero_count = 0
                    scan_for_start = True
                else:
                    zero_count += 1
    return run_indices

def yield_pick_windows(array, fs, wl, g, results_dict, max_gap, buffer):
    '''
    Yield windows to run cpic picker over given detection results.
    '''
    run_indices = find_runs_with_gaps(results_dict, max_gap)
    for run in run_indices:
        start_index = int(fs*(run[0]*g - buffer))
        if start_index < 0:
            start_index = 0
        end_index = int(fs*(run[1]*g + wl + buffer))
        if end_index > (len(array[0]) - 1):
            end_index = len(array[0]) - 1
        yield start_index, array[:, start_index:end_index]

def generate_picks(array, model, transform, fs, wl, g_detect, g_pick,
                   results_dict, max_gap, buffer):
    p_picks = []
    p_confs = []
    s_picks = []
    s_confs = []
    for start_index, window in yield_pick_windows(array, fs, wl, g_detect,
                                               results_dict, max_gap, buffer):
        pick_results = pick(window, fs, wl, model, transform, g_pick)
        if type(pick_results['p']) == np.ndarray:
            for i in range(len(pick_results['p'])):
                picktime = pick_results['p'][i]
                p_picks.append(start_index/fs + picktime)
                p_confs.append(pick_results['p_conf'][i])
        if type(pick_results['s']) == np.ndarray:
            for i in range(len(pick_results['s'])):
                picktime = pick_results['s'][i]
                s_picks.append(start_index/fs + picktime)
                s_confs.append(pick_results['s_conf'][i])
    return {'p picks': p_picks, 'p confs': p_confs, 's picks': s_picks,
            's confs': s_confs}

def picks2table(picks, starttime=None):
    data = []
    cols = ('phase', 'pick time', 'confidence')
    for i in range(len(picks['p picks'])):
        phase = 'p'
        if starttime:
            picktime = str(starttime + picks['p picks'][i])
        else:
            picktime = picks['p picks'][i]
        conf = picks['p confs'][i]
        row_entry = (phase, picktime, conf)
        data.append(dict(zip(cols, row_entry)))
    for i in range(len(picks['s picks'])):
        phase = 's'
        if starttime:
            picktime = str(starttime + picks['s picks'][i])
        else:
            picktime = picks['s picks'][i]
        conf = picks['s confs'][i]
        row_entry = (phase, picktime, conf)
        data.append(dict(zip(cols, row_entry)))
    df = pd.DataFrame(data)
    df = df[list(cols)]
    return df
