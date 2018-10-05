from matplotlib import pyplot

p1 = [152, 140]
p2 = [260, 60]

p1_ = [1, 1]
p2_ = [1, 1]

x1 = p1[0]
x2 = p2[0]
y1 = p1[1]
y2 = p2[1]

# x_t =
t1 = 0
t2 = 134

step = 1

x_t = []
y_t = []

for t in range(t1, t2, step):
    temp_x_t = x1 + t + ( 3*(x2 - x1)/(t2**2) - (2 * p1_[0]/ t2) - (p2_[0]/ t2) ) * (t ** 2)
    temp_x_t += ( 2*(x1 - x2)/(t2**3) + p1_[0]/(t2**2) + p2_[0]/(t2**2)) * (t**3)
    x_t.append(temp_x_t)

    temp_y_t = y1 + t + ( 3*(y2 - y1)/(t2**2) - (2 * p1_[0]/ t2) - (p2_[0]/ t2) ) * (t ** 2)
    temp_y_t += ( 2*(y1 - y2)/(t2**3) + p1_[0]/(t2**2) + p2_[0]/(t2**2)) * (t**3)
    y_t.append(temp_y_t)

pyplot.plot(x_t, y_t)
pyplot.xlabel('x')
pyplot.ylabel('y')
pyplot.show()

# from matplotlib import pyplot
# import math
#
# p1 = [152, 140]
# p2 = [260, 60]
# p3 = [350, 140]
#
# p1_ = [1, 1]
# p2_ = [1, 1]
# p3_ = [1, 1]
#
# x1 = p1[0]
# x2 = p2[0]
# x3 = p3[0]
# y1 = p1[1]
# y2 = p2[1]
# y3 = p3[1]
#
# # x_t =
# t1 = 0
# t2 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
# t3 = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
# print(t3)
#
#
# step = 1
#
# x_t = []
# y_t = []
#
# for t in range(t1, int(t2), step):
#     temp_x_t = x1 + t + ( 3*(x2 - x1)/(t2**2) - (2 * p1_[0]/ t2) - (p2_[0]/ t2) ) * (t ** 2)
#     temp_x_t += ( 2*(x1 - x2)/(t2**3) + p1_[0]/(t2**2) + p2_[0]/(t2**2)) * (t**3)
#     x_t.append(temp_x_t)
#
#     temp_y_t = y1 + t + ( 3*(y2 - y1)/(t2**2) - (2 * p1_[0]/ t2) - (p2_[0]/ t2) ) * (t ** 2)
#     temp_y_t += ( 2*(y1 - y2)/(t2**3) + p1_[0]/(t2**2) + p2_[0]/(t2**2)) * (t**3)
#     y_t.append(temp_y_t)
#
# t2 = 0
#
# for t in range(int(t2), int(t3), step):
#     temp_x_t = x2 + t + ( 3*(x3 - x2)/(t3**2) - (2 * p2_[0]/ t3) - (p3_[0]/ t3) ) * (t ** 2)
#     temp_x_t += ( 2*(x2 - x3)/(t3**3) + p2_[0]/(t3**2) + p3_[0]/(t3**2)) * (t**3)
#     x_t.append(temp_x_t)
#
#     temp_y_t = y2 + t + ( 3*(y3 - y2)/(t3**2) - (2 * p2_[0]/ t3) - (p3_[0]/ t3) ) * (t ** 2)
#     temp_y_t += ( 2*(y2 - y3)/(t3**3) + p2_[0]/(t3**2) + p3_[0]/(t3**2)) * (t**3)
#     y_t.append(temp_y_t)
#
# pyplot.plot(x_t, y_t)
# pyplot.xlabel('x')
# pyplot.ylabel('y')
# pyplot.show()
