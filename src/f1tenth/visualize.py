import matplotlib.pyplot as plt
from matplotlib import rc
from copy import deepcopy
import numpy as np
import pickle
import glob
import sys
import os

def main():
    fig_size = 5
    window_size = 500
    interp_steps = 1000
    steps = int(1.5e6)
    env_name = "F1Tenth"
    algo_list = []

    item_list = ['score']
    algo_list.append({
        'name': 'test',
        'logs': [f'results/test/{i}' for i in range(1,2)]
    })
    draw(env_name, item_list, algo_list, fig_size, window_size, interp_steps, is_horizon=False, steps=steps)


def draw(env_name, item_list, algo_list, fig_size, window_size, interp_steps, is_horizon=True, steps=int(1e6)):
    if is_horizon:
        fig, ax_list = plt.subplots(nrows=1, ncols=len(item_list), figsize=(fig_size*len(item_list), fig_size))
    else:
        fig, ax_list = plt.subplots(nrows=len(item_list), ncols=1, figsize=(fig_size*1.3, fig_size*len(item_list)))
    if len(item_list) == 1:
        ax_list = [ax_list]
    if is_horizon == True:
        plt.suptitle(f'{env_name}', fontsize="x-large", style='bold')
    elif is_horizon == False:
        plt.suptitle(f'{env_name}\n', fontsize="x-large", fontweight='bold')
        
    for item_idx in range(len(item_list)):
        ax = ax_list[item_idx]
        item_name = item_list[item_idx]
        min_value = np.inf
        max_value = -np.inf
        for algo_idx in range(len(algo_list)):
            algo_dict = algo_list[algo_idx]
            algo_name = algo_dict['name']
            algo_logs = algo_dict['logs']
            algo_dirs = ['{}/logs/{}_log'.format(dir_item, item_name.replace('total_', '').replace('metric', 'score')) for dir_item in algo_logs]
            linspace, means, stds = parse(algo_dirs, item_name, window_size, interp_steps)
            ax.plot(linspace, means, lw=2, label=algo_name)
            ax.fill_between(linspace, means - stds, means + stds, alpha=0.15)
            max_value = max(max_value, np.max(means + stds))
            min_value = min(min_value, np.max(means - stds))

        ax.set_xlabel('Steps')
        prefix, postfix = "", ""
        fontsize = "x-large"
        if item_idx == 0:
            ax.legend(loc='upper left', ncol=1, borderaxespad=0.)

        if item_name == "metric":
            ax.set_title(f'{prefix}Score{postfix}', fontsize=fontsize)
        elif item_name == "score":
            ax.set_title(f'{prefix}Reward Sum{postfix}', fontsize=fontsize)
        elif item_name == "cv":
            ax.set_title(f'{prefix}CV{postfix}', fontsize=fontsize)
            ax.set_ylim(0, max_value)
        elif item_name == "cost":
            ax.set_title(f'{prefix}Cost{postfix}', fontsize=fontsize)
            ax.set_ylim(0, max_value)
        elif item_name == "total_cv":
            ax.set_title(f'{prefix}Total CV{postfix}', fontsize=fontsize)
            ax.set_ylim(0, max_value)
        else:
            ax.set_title(item_name)
        ax.set_xlim(0, steps)
        ax.grid()

    fig.tight_layout()
    save_dir = "./imgs"
    item_names = '&'.join(item_list)
    env_name = env_name.replace(' ', '')
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    plt.savefig(f'{save_dir}/{env_name}_{item_names}.png')


def parse(algo_dirs, item_name, window_size, interp_steps):
    algo_datas = []
    min_linspace = None
    min_len = np.inf
    print(f'[parsing] {algo_dirs}')
    for algo_dir in algo_dirs:
        record_paths = glob.glob('./{}/*.pkl'.format(algo_dir))
        record_paths.sort()
        record = []
        for record_path in record_paths:
            with open(record_path, 'rb') as f:
                record += pickle.load(f)

        if item_name == "metric":
            cv_record_paths = glob.glob('./{}/*.pkl'.format(algo_dir.replace('score', 'cv')))
            cv_record_paths.sort()
            cv_record = []
            for record_path in cv_record_paths:
                with open(record_path, 'rb') as f:
                    cv_record += pickle.load(f)

        steps = [0]
        data = [0.0]
        for step_idx in range(len(record)):
            steps.append(steps[-1] + record[step_idx][0])
            if item_name == 'metric':
                data.append(record[step_idx][1]/(cv_record[step_idx][1] + 1))
            elif 'total' in item_name:
                data.append(data[-1] + record[step_idx][1])
            else:
                data.append(record[step_idx][1])

        linspace = np.linspace(steps[0], steps[-1], int((steps[-1]-steps[0])/interp_steps + 1))
        if min_len > len(linspace):
            min_linspace = linspace[:]
            min_len = len(linspace)
        interp_data = np.interp(linspace, steps, data)
        algo_datas.append(interp_data)

    algo_len = min([len(data) for data in algo_datas])
    algo_datas = [data[:algo_len] for data in algo_datas]

    smoothed_means, smoothed_stds = smoothing(algo_datas, window_size)
    return min_linspace, smoothed_means, smoothed_stds

def smoothing(data, window_size):
    means = []
    stds = []
    for i in range(1, len(data[0]) + 1):
        if i < window_size:
            start_idx = 0
        else:
            start_idx = i - window_size
        end_idx = i
        concat_data = np.concatenate([item[start_idx:end_idx] for item in data])
        a = np.mean(concat_data)
        b = np.std(concat_data)
        means.append(a)
        stds.append(b)
    return np.array(means), np.array(stds)

if __name__ == "__main__":
    main()