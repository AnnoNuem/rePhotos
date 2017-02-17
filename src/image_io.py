#!/usr/bin/env python
"""
Writes and saves point and line lists from and to json files
"""
import json


def read_lines(filename):
    """Reads lines from files.

    Parameters
    ----------
    filename : string
        Name of file.

    Returns
    -------
    src_lines : list
        List of lines in src image. Empty list if unable to open file.
    dst_lines : list
        List of lines in dst image. Empty list if unable to open file.

    """
    try:
        f = open(filename, "r")
    except IOError:
        print("Could not open line file.")
        return [], []

    with f:
        [src_lines, dst_lines] = json.load(f)

    return src_lines, dst_lines


def write_lines(src_lines, dst_lines, filename):
    """Writes linelists to file.

    Parameters
    ----------
    src_lines : list
        List of lines in src image.
    dst_lines : list
        List of lines in dst image.
    filename : string
        Name of file.

    Returns
    -------

    """
    try:
        f = open(filename, "w")
    except IOError:
        print("Could not save lines to file.")

    with f:
        json.dump([src_lines, dst_lines], f)


def read_points(filename):
    """Reads points from files.

    Parameters
    ----------
    filename : string
        Name of file.

    Returns
    -------
    src_points : list
        List of points in src image. Empty list if unable to open file.
    dst_points : list
        List of points in dst image. Empty list if unable to open file.

    """
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
    """Writes pointlists to file.

    Parameters
    ----------
    src_points : list
        List of points in src image.
    dst_points : list
        List of points in dst image.
    filename : string
        Name of file.

    Returns
    -------

    """
    try:
        f = open(filename, "w")
    except IOError:
        print("Could not save points to file.")

    with f:
        json.dump([src_points, dst_points], f)
