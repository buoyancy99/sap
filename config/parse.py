import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', '-w', type=int, default=1, help='environment index, see readme for examples')
    parser.add_argument('--nodeath', action='store_true', help='no done signal ablation')
    parser.add_argument('--plan_step', '-p', type=int, default=10, help='planning step ablation')
    parser.add_argument('--store_video', action='store_true')
    parser.add_argument('--mbhp', action='store_true', help='mbhp reward baseline')

    args = parser.parse_args()

    return args

