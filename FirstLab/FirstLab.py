import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt

from numpy.random.mtrand import normal

file = open("model_1.obj")
v = []
vt = []
vn = []
fv = []
fvt = []
fvn = []
#v/vt/vn
for s in file:
    spl = s.split()
    if spl[0] == "vt":
        vt.append(list(map(float, spl[1:])))
    if spl[0] == "vn":
        vn.append(list(map(float, spl[1:])))
    if spl[0] == "v":  # ОООО ZZZZZZZ ZZZ ГОЙДА
        v.append(list(map(float, spl[1:])))

    if spl[0] == "f":
        v_index = [int(sp.split("/")[0]) for sp in spl[1:]]
        vt_index = [int(sp.split("/")[1]) for sp in spl[1:]]
        vn_index = [int(sp.split("/")[2]) for sp in spl[1:]]
        fv.append(v_index)
        fvt.append(vt_index)
        fvn.append(vn_index)


def draw_line(img_mat, x0, y0, x1, y1, count):
    step = 1.0/count
    for t in np.arange(0, 1, step):
        x = round((1 - t)*x0 + t*x1)
        y = round((1 - t)*y0 + t*y1)
        img_mat[y, x] = 255

def draw_line1(img_mat, x0, y0, x1, y1, count):
    count = sqrt((x0-x1)**2 + (y0-y1)**2)
    step = 1.0/count
    for t in np.arange(0,1, step):
        x = round((1 - t)*x0 + t*x1)
        y = round((1 - t)*y0 + t*y1)
        img_mat[y, x] = 255

def draw_line2(img_mat, x0, y0, x1, y1, count): # рисует половину т.к.
    # for в туза не идёт
    for x in range(x0, x1):
        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = 255

def draw_line4(img_mat, x0, y0, x1, y1, count): # теперь идёт в туза
    # не дорисоывает тк y изменяется быстрее x
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        img_mat[y, x] = 255

def draw_line5(img_mat, x0, y0, x1, y1, count):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(x0, x1):
        t = (x - x0)/(x1 - x0)
        y = round((1.0 - t)*y0 + t*y1)
        if xchange:
            img_mat[x, y] = 255
        else:
            img_mat[y, x] = 255


# def draw_line5(img_mat, x0, y0, x1, y1, count):
#     xchange = False
#     if abs(x0 - x1) < abs(y0 - y1):
#         x0, y0 = y0, x0
#         x1, y1 = y1, x1
#         xchange = True
#
#     if x0 > x1:
#         x0, x1 = x1, x0
#         y0, y1 = y1, y0
#
#     for x in range(x0, x1):
#         t = (x - x0)/(x1 - x0)
#         y = round((1.0 - t)*y0 + t*y1)
#         if xchange:
#             img_mat[x, y] = 255
#         else:
#             img_mat[y, x] = 255


def draw_line6(img_mat, x0, y0, x1, y1, count):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            img_mat[x, y] = 255
        else:
            img_mat[y, x] = 255
        derror += dy
        if derror > 0.5:
            derror -= 1.0
            y += y_update

def draw_line7(img_mat, x0, y0, x1, y1, count):
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2.0*(x1-x0)*abs(y1 - y0)/(x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            img_mat[x, y] = 255
        else:
            img_mat[y, x] = 255
        derror += dy
        if derror > (2.0*(x1 - x0) * 0.5):
            derror -= 2.0*(x1 - x0)* 1.0
            y += y_update


def draw_line8(img_mat, x0, y0, x1, y1, count): # полностью целочисленный метод, работает как мой адвокат порекомендовал не продолжать шутку
    xchange = False
    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1

    for x in range(x0, x1):
        if xchange:
            img_mat[x, y] = 255
        else:
            img_mat[y, x] = 255
        derror += dy
        if derror > (x1 - x0):
            derror -= 2*(x1 - x0)
            y += y_update

def scaleCoords(coords):
    return (int(9000*coords[0] + 1000), int(9000*coords[1] + 1000))

def draw_polygon(img_mat, fv):
    x1, y1 = scaleCoords(v[fv[0] - 1])
    x2, y2 = scaleCoords(v[fv[1] - 1])
    x3, y3 = scaleCoords(v[fv[2] - 1])

    draw_line8(img_mat, x1, y1, x2, y2, 1)
    draw_line8(img_mat, x2, y2, x3, y3, 1)
    draw_line8(img_mat, x3, y3, x1, y1, 1)


img_mat = np.zeros((2000, 2000, 3), dtype=np.uint8)



#for k in range(13):
#    x0, y0 = 100,100
#    x1, y1 = int(x0 + 95*cos(2*pi*k/13)), int(y0 + 95*sin(2*pi*k/13))
#    draw_line8(img_mat, x0, y0, x1, y1, 95)
#for vert in v:
#    vx, vy = int(10000*vert[0] + 1000), int(10000*vert[1] + 1000)
#    img_mat[vy, vx] = 255

for pg in fv:
    draw_polygon(img_mat, pg)


# (u, v) -> (x, y)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.show()
img.save('img.png')