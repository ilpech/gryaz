#brief: график радиусов ориентированной кривизный по времени [1/R(t)]
#       для использования менять SRC_FOLDER
#usage: python3 circle_fit_season.py
#authors: pichugin
import os
import sys
import numpy as np
from lib.modules import (
    yamls_dir, ungzip,
    choose_sense_and_shots,
    vector_sum, mercator
    )
import matplotlib.pyplot as plt

iSeason = 51
serial  = 'skl'

SRC_FOLDER = "/testdata/{0:s}/inp/{1:s}.{2:03d}/".format(serial, serial, iSeason)
print (SRC_FOLDER)

def circle_fit(x, y):
    covmat = np.cov(x, y)
    cx2x = np.cov(x * x, x)[0, 1]
    cy2x = np.cov(y * y, x)[0, 1]
    cx2y = np.cov(x * x, y)[0, 1]
    cy2y = np.cov(y * y, y)[0, 1]
    cx = 0.5 * (cx2x + cy2x)
    cy = 0.5 * (cx2y + cy2y)
    c = [[cx], [cy]]
    try:
        center_lsm = np.linalg.solve(covmat, c)
    except np.linalg.linalg.LinAlgError:
        return 0, 0, 0
    means = np.array([[np.mean(x)], [np.mean(y)]])
    R = np.math.sqrt(np.linalg.norm(center_lsm - means)**2 + np.var(x) + np.var(y))
    return center_lsm[0], center_lsm[1], R


def smart_window_stack(a, b):
    start_p = []
    end_p = []
    good_hords = []
    start = 0
    end = 0
    for i in range(len(b)):
        end = i
        hord = (a[end] - a[start], b[end] - b[start]) # координаты хорды
        hord_l = (hord[0]**2 + hord[1]**2)**(1/2) # длина хорды

        if hord_l > 1.0 and hord_l < 100.0:
            # вычисляем "средний вектор" - между серединой окна и проекцией на хорду
            midl_wind = (a[int((start + end)/2)], b[int((start + end)/2)]) # центр окна
            x = np.array(midl_wind)
            line = [(a[start], b[start]), (a[end], b[end])]
            u = np.array(line[0])
            v = np.array(line[1])
            n = v - u
            n /= np.linalg.norm(n, 2)
            P = u + n*np.dot(x - u, n) # проекция на хорду

            midl_v = (midl_wind[0] - P[0],
                      midl_wind[1] - P[1])
            midl_v_l = (midl_v[0]**2 + midl_v[1]**2)**(1/2) # длина ср вект

            if midl_v_l > 6.0:
                start_p.append(start)
                end_p.append(end)
                good_hords.append(hord)
                start += 5
        elif hord_l > 100.0:
            start += 5
            continue
        else:
            continue
    return start_p, end_p, good_hords


season_title = "{0:s}.{1:03d}".format(serial, iSeason)
print(season_title)
yamls_in_directory = yamls_dir(SRC_FOLDER)
yamls_in_directory = sorted(yamls_in_directory)
al_lat = []
al_lon = []
al_t = []
al_v = []
for yaml_in_directory in yamls_in_directory:
    try:
        yml = ungzip(yaml_in_directory)
    except EOFError:
        continue
    except OSError:
        continue

    sense, shots = choose_sense_and_shots(yml, season_title)
    note = ''
    lat = []
    lon = []
    t   = []
    v = []
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
            velocity >= 1
            ):
            t.append(shots[index][sense]['grabMsec'])
            lat.append(format(shots[index][sense]['senseData']['nord'], ".8f"))
            lon.append(format(shots[index][sense]['senseData']['east'], ".8f"))
            v.append(velocity)
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
        al_v.append(v[i])
        al_t.append(t[i])
        al_lat.append(lat[i])
        al_lon.append(lon[i])

if len(al_lat) != 0:
    lat0 = al_lat.pop(0)
    lon0 = al_lon.pop(0)
    lat = np.array(al_lat, dtype=float)
    lon = np.array(al_lon, dtype=float)
    x, y = mercator(lat,lon, float(lat0), float(lon0))
    print("{} good points founded".format(len(x)))

else:
    print("No good points founded")
    sys.exit()

plt.figure()
plt.title("empty track {}".format(season_title))
plt.plot(x, y, '-o') # график пустого проезда с точками измерений
plt.grid(True)
plt.axis('Equal')

plt.figure()
plt.title(season_title)
plt.plot(x, y, '--')
plt.grid(True)
plt.axis('Equal')

R_d = []
t_d = []
s_d = []
start_p, end_p, hords = smart_window_stack(x,y)

for i in range(len(start_p)):
    c0, c1, R = circle_fit(x[start_p[i]:end_p[i]], y[start_p[i]:end_p[i]])

    if R > 18.0:
        circle = plt.Circle((c0, c1), R, fill=False, label=R)
        r = (c0 - x[start_p[i]], c1 - y[start_p[i]])

        plt.gcf().gca().add_artist(circle)
        plt.plot([c0, x[int((end_p[i]+start_p[i])/2)]], [c1, y[int((end_p[i]+start_p[i])/2)]], '--') # радиусы
        plt.plot([x[start_p[i]],x[end_p[i]]], [y[start_p[i]],y[end_p[i]]], 'g-') # hords

        if np.linalg.det((hords[i],r)) < 0: # ориентированная кривизна
            R = -R
        print("R {} on window from point {} to point {}".format(R, start_p[i], end_p[i]))
        R_d.append(1/R)
        t_d.append(al_t[start_p[i]])
        v_av = (al_v[start_p[i]] + al_v[end_p[i]])/2
        t_delta = (al_t[end_p[i]] - al_t[start_p[i]]) / 3600
        s_d.append(v_av * t_delta)


for i in range(len(s_d) + 1):
    try:
        s_d[i+1] += s_d[i]
    except IndexError:
        pass


plt.figure()
plt.title("{} 1 {:03d}".format(season_title, len(yamls_in_directory)))
plt.plot(s_d, R_d, 'o-')
plt.ylabel('1/R [meters]')
plt.xlabel('S [meters]')
plt.grid(True)

plt.show()
