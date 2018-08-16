import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt


def pos(t, x1, v1, b, c):
    return x1 + v1*t + (t**2)*b/2 + (t**3)*c/6


def kinematic_interpolation(xytvv, times):
    """
    Interpolate using kinematic interpolation
    Perform kinematic path interpolation on a movement dataset. Kinematic path interpolation was introduced in the
    paper Long (2015). Kinematic interpolation is appropriate for fast moving objects, recorded with relatively high
    resolution tracking data.
    Kinematic interpolation requires the user to input the coordinates of the anchor points between which the
    interpolation is occurring, as well as initial and final velocities associate with the anchor points. In practice,
    these velocities may be explicitly known, or estimated from the tracking data.
    Long, JA (2015) Kinematic interpolation of movement data. International Journal of Geographical Information Science.
    DOI: 10.1080/13658816.2015.1081909.
    :param xytvv: a 2x5 array containing the coordinates, times, and initial and final velocities (as 2D vectors)
    of the two points to be interpolated between, often termed the anchor points. Each row of the array should be
    arranged as x, y, t, vx, vy.
    :param times: a single time (POSIX or numeric), or list of times, to be interpolated for. The times must lie between
    those of the points in xytvv.
    :return: The function returns a dataframe (with nrow = len(t)) corresponding to the interpolated locations.
    """

    x1 = xytvv[0, 0:2]
    x2 = xytvv[1, 0:2]
    t1 = xytvv[0, 2]
    t2 = xytvv[1, 2]
    v1 = xytvv[0, 3:]
    v2 = xytvv[1, 3:]

    print (v1,v2)

    t = t2 - t1
    t_s = times - t1
    print(t_s)

    ax = np.array([ [(t**2)/2, (t**3)/6], [float(t), (t**2)/2] ])
    bx = [x2[0]-x1[0]-v1[0]*t, v2[0]-v1[0]]
    coef_x = np.linalg.solve(ax, bx)

    ay = ax
    by = [x2[1]-x1[1]-v1[1]*t, v2[1]-v1[1]]
    coef_y = np.linalg.solve(ay, by)

    x = pos(t_s, x1[0], v1[0], coef_x[0], coef_x[1])
    y = pos(t_s, x1[1], v1[1], coef_y[0], coef_y[1])

    return pnd.DataFrame({'x': x, 'y': y, 't': times})


if __name__ == '__main__':

    #contrived_data = {'x': [0, 0, 10, 13], 'y': [-3, 0, 10, 10], 't':[0, 1, 6, 7] }
    contrived_data = np.array([[0, 0, 10, 13], [-3, 0, 10, 10], [0, 1, 6, 7]])
    speeds = np.array([[2,3],[3,5]])
    contrived_data = contrived_data.transpose()
    xyt = contrived_data
    xyt = contrived_data[1:3,:]
    # print(xyt)
    xytvv = np.append(xyt, speeds, axis=1)
    print(xytvv)
    # times = [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.5]
    times = np.linspace(0, 7, 20)
    # print(times)

    interpolated_points = kinematic_interpolation(xytvv, times)
    print(interpolated_points)

    #Plotting
    plt.figure()
    contrived_data_points = pnd.DataFrame({'x': [0, 0, 10, 13], 'y': [-3, 0, 10, 10], 't':[0, 1, 6, 7]})
    all_points = contrived_data_points.append(interpolated_points).sort_values('t')
    plt.plot(all_points['x'], all_points['y'], 'bo-')
    plt.plot(interpolated_points['x'], interpolated_points['y'], 'go-')
    plt.show()
