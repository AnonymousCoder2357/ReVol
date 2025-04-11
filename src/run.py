"""
Run main.py N times with different random seeds and GPUs.
"""
import argparse
import multiprocessing
import os

import numpy as np
import subprocess
from multiprocessing import Pool
import datetime


def parse_args():
    """
    Parse command line arguments.

    The other arguments not defined in this function are directly passed to main.py. For instance,
    an option like "--beta 1" is given directly to the main script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)))
    parser.add_argument('--gpus', type=int, nargs='+', default=list(range(4)))
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_known_args()


def run_command(args):
    """
    Run main.py with a suitable GPU given as an argument.

    :param args: the pair of a command and a list of GPUs.
    :return: None.
    """
    command, gpu_list = args
    gpu_idx = int(multiprocessing.current_process().name.split('-')[-1]) - 1
    gpu = gpu_list[gpu_idx % len(gpu_list)]
    command += ['--device', str(gpu)]
    subprocess.call(command)

def parse_fm_arg(unknown):
    if '--fm' in unknown:
        fm_index = unknown.index('--fm') + 1
        if fm_index < len(unknown):
            return unknown[fm_index]
    return 'lstm'


def main():
    """
    Main function to run main.py with multiple jobs.

    :return: None.
    """
    args, unknown = parse_args()
    assert args.data is not None
    print(args,unknown)
    out_path = '{}/{}'.format(args.out, args.data)
    args_list = []
    for seed in args.seeds:
        command = ['python', 'main.py', '--silent',
                   '--data', args.data,
                   '--seed', str(seed),
                   '--out', out_path]
        args_list.append((command + unknown, args.gpus))

    with Pool(len(args.gpus) * args.workers) as pool:
        pool.map(run_command, args_list)

    values = []
    for seed in args.seeds:
        values.append(np.loadtxt(os.path.join(out_path, str(seed), 'out.tsv'), delimiter='\t'))
    avg = np.stack(values, axis=0).mean(axis=0)
    std = np.stack(values, axis=0).std(axis=0)
    for a, s in zip(avg, std):
        print('{:.4f}\t{:.4f}'.format(a, s), end='\t\t\t')
    print()


    os.makedirs(os.path.join(out_path,'res'), exist_ok=True)
    file_name = "%s_%s.txt"%(args.data,parse_fm_arg(unknown))


    with open(os.path.join(out_path,'res',file_name), 'w') as f:
        f.write(f"Args: {args}\n")
        
        f.write(f"Unknown: {unknown}\n")

        np.savetxt(f,np.stack(values, axis=0),delimiter='\t',fmt='%.4f')
        f.write("\n\n")
        np.savetxt(f,np.stack([avg,std], axis=0),delimiter='\t',fmt='%.4f')

if __name__ == '__main__':
    main()
