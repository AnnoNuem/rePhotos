import json


def read_lines(filename):
    try:
        f = open(filename, "r")
    except IOError:
        print("Could not open line file.")
        return [], []

    with f:
        [src_lines, dst_lines] = json.load(f)

    return src_lines, dst_lines


def write_lines(src_lines, dst_lines, filename):
    try:
        f = open(filename, "w")
    except IOError:
        print("Could not save lines to file.")

    with f:
        json.dump([src_lines, dst_lines], f)


def read_points(filename):
    try:
        f = open(filename, "r")
    except IOError:
        print("Could not open point file.")
        return [], []

    with f:
        [src_points, dst_points] = json.load(f)
        src_points = [tuple(point) for point in src_points]
        dst_points = [tuple(point) for point in dst_points]

    return src_points, dst_points


def write_points(src_points, dst_points, filename):
    try:
        f = open(filename, "w")
    except IOError:
        print("Could not save points to file.")

    with f:
        json.dump([src_points, dst_points], f)
