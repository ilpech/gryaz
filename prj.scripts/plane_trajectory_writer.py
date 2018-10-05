import os
import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from lib.modules import (
    Point, dist,
    yamls_dir, ungzip,
    choose_sense_and_shots
)



def trajectory_writer(src_folder, dst_folder):
    yamls_in_directory = yamls_dir(src_folder)
    yamls_in_directory = sorted(yamls_in_directory)
    season_title = src_folder[-7:]
    for yaml_in_directory in yamls_in_directory:
        file_name = os.path.basename(yaml_in_directory)[:11]
        try:
            yml = ungzip(yaml_in_directory)
        except EOFError:
            continue
        except OSError:
            continue
        try:
            sense, shots = choose_sense_and_shots(yml, season_title)
        except TypeError:
            print('Episode {} does not have any shots'.format(episode_title))
            continue
        old_index = 0
        index = 1
        empty_kml_flag = True

        points = []

        while index < len(shots):
            if (
                shots[old_index][sense]['senseData']['east'] == 0.0 or
                shots[old_index][sense]['senseData']['nord'] == 0.0 or
                shots[index][sense]['senseData']['east'] == 0.0 or
                shots[index][sense]['senseData']['nord'] == 0.0
                ):
                old_index += 1
                index += 1
                continue
            else:

                first_point = Point(
                            shots[old_index][sense]['senseData']['nord'],
                            shots[old_index][sense]['senseData']['east'],
                            shots[old_index][sense]['senseData']['alt'])

                second_point = Point(
                            shots[index][sense]['senseData']['nord'],
                            shots[index][sense]['senseData']['east'],
                            shots[index][sense]['senseData']['alt'])

            if (
                dist(first_point, second_point) >= 0.01 and
                dist(first_point, second_point) < 0.1
                ):

                points.append(str(first_point.coords()[0]) + ' ' + str(first_point.coords()[1]))
                points.append(str(second_point.coords()[0]) + ' ' + str(second_point.coords()[1]))
                first_point = second_point
                old_index = index
                empty_flag = False
            index += 1

        points = set(points)

        if len(points) == 0:
            continue

        os.chdir(dst_folder)
        filename = file_name + '.trj'
        f = open(filename, 'w')
        for p in points:
            f.write(str(p) + '\n')

        f.close()


trajectory_writer('/testdata/trm/inp/trm.044', '/testdata/trm/trj')
