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

def get_data_on_season(src_folder):
    yamls_in_directory = yamls_dir(src_folder)
    yamls_in_directory = sorted(yamls_in_directory)
    season_lat = []
    season_lon = []
    season_speed = []
    season_time = []
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
        speed_t = []
        grab_msecs = []
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
                grab_msecs.append(shots[index][sense]['grabMsec'] / 360000)
                speed_t.append(velocity)
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
            season_lat.append(lat[i])
            season_lon.append(lon[i])
            season_speed.append(speed_t[i])
            season_time.append(grab_msecs[i])

    if len(season_lat) != 0:
        return season_lat, season_lon, season_speed, season_time
    else:
        print("No good points founded")

import numpy as np

s_lat_data, s_lon_data, s_speed_data, s_timestamps = get_data_on_season(SRC_FOLDER)

if len(s_lat_data) != 0:
    lat0 = s_lat_data.pop(0)
    lon0 = s_lon_data.pop(0)
    lat = np.array(s_lat_data, dtype=float)
    lon = np.array(s_lon_data, dtype=float)
    x, y = mercator(lat,lon, float(lat0), float(lon0))


    plt.plot(x,y,'--')
    plt.grid(True)
    plt.show()


plt.plot(s_timestamps, s_speed_data)
plt.xlabel('t [h]')
plt.ylabel('speed [km/h]')
# plt.plot(np.diff(s_speed_data, s_timestamps-1))
plt.show()
