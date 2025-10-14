import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt
from random import randint, random


width, height = 2000, 2000

def calculateBarycentric(x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) /((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)) 
    lambda2 = 1.0 - lambda0 - lambda1 
    return (lambda0, lambda1, lambda2)

def calcNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return ((y1 - y2) * (z1 - z0) - (z1 - z0) * (y1 - y0), -((x1 - x2) * (z1 - z0) - (z1 - z2) * (x1 - x0)), (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))

def getXYMaxMin(x0, y0, x1, y1, x2, y2):
    xmin = min(x0, x1, x2)
    ymin = min(y0, y1, y2)
    xmax = max(x0, x1, x2)
    ymax = max(y0, y1 ,y2)
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax < 0): xmax = 0
    if (ymax < 0): ymax = 0
    if (xmin > width): xmin = width
    if (xmax > width): xmax = width
    if (ymin > height): ymin = height
    if (ymax > height): ymax = height
    
    return (int(xmin), int(ymin), int(xmax), int(ymax))

def cosLight(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    normal = calcNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    dot = normal[2]
    lenNormal = sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    return dot/lenNormal

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

def draw_line8(img_mat, x0, y0, x1, y1, count): # полностью целочисленныйa метод, работает как мой адвокат порекомендовал не продолжать шутку
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
    return (9000*coords[0] + width/2, 9000*coords[1] + height/2, 9000*coords[2] + width/2)

def draw_polygon(img_mat, fv):
    x1, y1 = scaleCoords(v[fv[0] - 1])
    x2, y2 = scaleCoords(v[fv[1] - 1])
    x3, y3 = scaleCoords(v[fv[2] - 1])

    x1, y1, x2, y2, x3, y3 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)

    draw_line8(img_mat, x1, y1, x2, y2, 1)
    draw_line8(img_mat, x2, y2, x3, y3, 1)
    draw_line8(img_mat, x3, y3, x1, y1, 1)


def draw_triangle_model(img_mat, z_buffer, fv):
    x1, y1, z1 = scaleCoords(v[fv[0] - 1])
    x2, y2, z2 = scaleCoords(v[fv[1] - 1])
    x3, y3, z3 = scaleCoords(v[fv[2] - 1])


    cs = cosLight(x1, y1, z1, x2, y2, z2, x3, y3, z3)
    light = -255 * cs
    if cs < 0:
        drawTriangle(img_mat, z_buffer, x1, y1, z1, x2, y2, z2, x3, y3, z3, (light, 156, light))


def drawTriangle(img_mat, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2, color):
    xmin, ymin, xmax, ymax = getXYMaxMin(x0, y0, x1, y1, x2, y2)
    for x in range(xmin, xmax + 1):
        for y in range(ymin, ymax + 1):
            lambda0, lambda1, lambda2 = calculateBarycentric(x, y, x0, y0, x1, y1, x2, y2)
            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if z < z_buffer[y][x]:
                    z_buffer[y][x] = z
                    img_mat[y][x] = color


img_mat = np.zeros((width, height, 3), dtype=np.uint8)
z_buffer = np.zeros((width, height))
z_buffer[0: width] = 10000


for pg in fv:
    #draw_polygon(img_mat, pg)
    draw_triangle_model(img_mat, z_buffer, pg)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)

img.show()
img.save('img.png')