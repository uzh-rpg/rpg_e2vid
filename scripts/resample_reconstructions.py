import numpy as np
from matplotlib import pyplot as plt
import argparse
import glob
from os.path import basename, join, exists
from os import makedirs
import math
import shutil


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return (idx - 1), array[idx - 1]
    else:
        return idx, array[idx]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Pick images in a folder containing timestamped images so that the resulting video has a fixed frame rate (used defined)')

    parser.add_argument('-i', '--input_folder', required=True, type=str)
    parser.add_argument('-o', '--output_folder', required=True, type=str)
    parser.add_argument('-r', '--framerate', default=1000.0, type=float)
    args = parser.parse_args()

    output_folder = args.output_folder
    if not exists(output_folder):
        makedirs(output_folder)

    # list all images in the folder
    images = [f for f in glob.glob(join(args.input_folder, "*.png"), recursive=False)]
    images = sorted(images)
    print('Found {} images'.format(len(images)))

    # read timestamps (and check there is one timestamp per image...)
    stamps = np.loadtxt(join(args.input_folder, 'timestamps.txt'))
    stamps = np.sort(stamps)
    np.savetxt(join(args.input_folder, 'timestamps_sorted.txt'), stamps)
    assert(len(stamps) == len(images))

    # find the closest image to each element in [t0, t0 + dt, t0 + 2 * dt, t0 + 3 * dt, ...]
    # where t0 = stamps[0]
    dt = 1.0 / args.framerate
    t = stamps[0]
    img_index, _ = find_nearest(stamps, t)

    i = 0
    while t <= stamps[-1]:
        t += dt
        img_index, _ = find_nearest(stamps, t)
        path_to_img = images[img_index]
        shutil.copyfile(path_to_img, join(output_folder, 'frame_{:010d}.png'.format(i)))
        i += 1
