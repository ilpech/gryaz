#brief: draw season's trajectory
#usage: python3 traj_writer.py
#authors: pichugin
from lib.modules import (
    yamls_dir, ungzip,
    choose_sense_and_shots,
    vector_sum, mercator
    )
import matplotlib.pyplot as plt
import numpy as np
import os

iSeason = 51
serial  = 'skl'

SRC_FOLDER = "/testdata/{0:s}/inp/{1:s}.{2:03d}/".format(serial, serial, iSeason)
print (SRC_FOLDER)

def get_gm_coords_on_season(src_folder, draw_flag=False):
    yamls_in_directory = yamls_dir(src_folder)
    yamls_in_directory = sorted(yamls_in_directory)
    al_lat = []
    al_lon = []
    for yaml_in_directory in yamls_in_directory:
        try:
            yml = ungzip(yaml_in_directory)
        except EOFError:
            continue
        except OSError:
            continue
        season_title = "test"
        sense, shots = choose_sense_and_shots(yml, season_title)
        note = ''
        lat = []
        lon = []
        velocity = 0.0
        index = 1
        while index < len(shots):
            if shots[index][sense]['senseData']['speed'] == 0:
                vl = shots[index][sense]['senseData']['vl']
                vf = shots[index][sense]['senseData']['vf']
                velocity = vector_sum(vl, vf) * 3.6
            else:
                velocity = shots[index][sense]['senseData']['speed'] * 3.6
            if (
                shots[index][sense]['senseData']['nord'] != 0.0 and
                shots[index][sense]['senseData']['east'] != 0.0 and
                velocity >= 10
                ):
                lat.append(format(shots[index][sense]['senseData']['nord'], ".8f"))
                lon.append(format(shots[index][sense]['senseData']['east'], ".8f"))
            else:
                pass
            index += 1
        bad_en_flag = False
        if len(lat) <= 2:
            bad_en_flag = True
        i = 1
        while i < len(lat):
            if lat[i-1] == lat[i] or lon[i-1] == lon[i]:
                bad_en_flag = True
            i += 1

        if bad_en_flag:
            continue
        for i in range(len(lat)):
            al_lat.append(lat[i])
            al_lon.append(lon[i])

    if len(al_lat) != 0:
        lat0 = al_lat.pop(0)
        lon0 = al_lon.pop(0)
        lat = np.array(al_lat, dtype=float)
        lon = np.array(al_lon, dtype=float)
        x, y = mercator(lat,lon, float(lat0), float(lon0))

        if(draw_flag):
            plt.plot(x,y,'--')
            plt.grid(True)
            plt.show()

        return x, y

    else:
        print("No good points founded")

get_gm_coords_on_season(SRC_FOLDER, True)
