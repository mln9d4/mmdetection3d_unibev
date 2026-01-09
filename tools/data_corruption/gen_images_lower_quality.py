import argparse
import os




def parse_arguments():
    parser = argparse.ArgumentParser(description='Lowering camera image quality')
    # parser.add_argument('-c', '--n_cpus', help='number of CPUs that should be used', type=int, default=mp.cpu_count())
    # parser.add_argument('-f', '--n_features', help='number of point features', type=int, default=4)
    parser.add_argument('-r', '--root_folder', help='root folder of dataset',
                        default='./data/nuscenes')
    parser.add_argument('-d', '--dst_folder', help='save folder of dataset',
                        default='./save_root/camera_lower_quality/set1')  # ['light','moderate','heavy']
    parser.add_argument('-b', '--down_scale_factor', help='float to determine loss of information', type=float, default=1.0)
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()