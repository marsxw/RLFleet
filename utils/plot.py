import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()


def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, no_legend=False, legend_loc='best', color=None,
              xlabel=None, ylabel=None, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)

    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, errorbar='sd', estimator=np.mean, **kwargs)

    # Use custom x and y labels if provided, otherwise use default column names
    plt.xlabel(xlabel if xlabel else xaxis)
    plt.ylabel(ylabel if ylabel else value)

    if not no_legend:
        plt.legend(loc=legend_loc)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def get_datasets(logdir, condition=None):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger. 

    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                with open(os.path.join(root, 'config.json')) as config_file:
                    config = json.load(config_file)
                    if 'exp_name' in config:
                        exp_name = config['exp_name']
            except FileNotFoundError:
                print(f"No file named config.json in {root}")

            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns), 'Unit', unit)
            exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
            exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
            exp_data.insert(len(exp_data.columns), 'Performance', exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is, 
           pull data from it; 
        2) if not, check to see if the entry is a prefix for a 
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1] == '/':
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            prefix = logdir.split('/')[-1]
            logdirs += [osp.join(basedir, x) for x in os.listdir(basedir) if prefix in x]

    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not (x in log) for x in exclude)]

    print('Plotting from...\n' + '='*DIV_LINE_WIDTH)
    for logdir in logdirs:
        print(logdir)
    print('='*DIV_LINE_WIDTH)

    assert not legend or (len(legend) == len(logdirs)), "Legend must match number of log directories."

    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg)
    else:
        for log in logdirs:
            data += get_datasets(log)
    return data


def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', no_legend=False, legend_loc='best',
               save_name=None, xlimit=-1, color=None, xlabel=None, ylabel=None):
    data = get_all_datasets(all_logdirs, legend, select, exclude)
    values = values if isinstance(values, list) else [values]
    condition = 'Condition2' if count else 'Condition1'
    for value in values:
        plt.figure()
        plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, no_legend=no_legend,
                  legend_loc=legend_loc, color=color, xlabel=xlabel, ylabel=ylabel)
        if xlimit > 0:
            plt.xlim(0, xlimit)

    if save_name is not None:
        fig = plt.gcf()
        fig.savefig(save_name)
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=10)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    parser.add_argument('--no-legend', action='store_true')
    parser.add_argument('--legend-loc', type=str, default='best')
    parser.add_argument('--save-name', type=str, default=None)
    parser.add_argument('--xlimit', type=int, default=-1)
    parser.add_argument('--color', '-color', nargs='*')

    # New arguments for custom x and y labels
    parser.add_argument('--xlabel', type=str, default=None, help="Custom label for x-axis")
    parser.add_argument('--ylabel', type=str, default=None, help="Custom label for y-axis")

    args = parser.parse_args()

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, no_legend=args.no_legend, legend_loc=args.legend_loc,
               save_name=args.save_name, xlimit=args.xlimit, color=args.color,
               xlabel=args.xlabel, ylabel=args.ylabel)


if __name__ == "__main__":
    main()
