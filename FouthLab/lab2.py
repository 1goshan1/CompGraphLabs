import numpy as np
from PIL import Image, ImageOps
from math import sin, cos, pi, sqrt

width, height = 2000, 2000

def rotate(x, y, z, a, b, c, tx, ty, tz):
    rx = np.array([[1, 0, 0], [0, cos(a), sin(a)], [0, -sin(a), cos(a)]])
    ry = np.array([[cos(b), 0, sin(b)], [0, 1, 0], [-sin(b), 0, cos(b)]])
    rz = np.array([[cos(c), -sin(c), 0], [sin(c), cos(c), 0], [0, 0, 1]])

    v = np.array([[x], [y], [z]])

    v = np.dot(rz, v)
    v = np.dot(ry, v)
    v = np.dot(rx, v)

    v = v + np.array([[tx], [ty], [tz]])

    return list(np.transpose(v))[0]

def calculateBarycentric(x, y, x0, y0, x1, y1, x2, y2):
    denominator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    if abs(denominator) < 1e-10:
        return (0, 0, 0)

    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denominator
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denominator
    lambda2 = 1.0 - lambda0 - lambda1
    return (lambda0, lambda1, lambda2)

def calcNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    v1 = np.array([x1 - x0, y1 - y0, z1 - z0])
    v2 = np.array([x2 - x0, y2 - y0, z2 - z0])

    normal = np.cross(v1, v2)

    length = np.linalg.norm(normal)
    if length > 0:
        normal = normal / length

    return normal

def getXYMaxMin(x0, y0, x1, y1, x2, y2):
    xmin = max(0, min(int(x0), int(x1), int(x2)))
    ymin = max(0, min(int(y0), int(y1), int(y2)))
    xmax = min(width - 1, max(int(x0), int(x1), int(x2)))
    ymax = min(height - 1, max(int(y0), int(y1), int(y2)))

    return (xmin, ymin, xmax, ymax)

def projective(ax, ay, u0, v0, x, y, z):
    return (ax * x / z + u0, ay * y / z + v0, z)

def scaleCoords(coords):
    return projective(1500, 1500, width / 2, height / 2, coords[0], coords[1], coords[2])

def compute_vertex_normals(v, fv):
    vertex_normals = [np.array([0.0, 0.0, 0.0]) for _ in range(len(v))]
    vertex_counts = [0] * len(v)

    for face in fv:
        v0 = face[0] - 1
        v1 = face[1] - 1
        v2 = face[2] - 1

        x0, y0, z0 = v[v0]
        x1, y1, z1 = v[v1]
        x2, y2, z2 = v[v2]

        normal = calcNormal(x0, y0, z0, x1, y1, z1, x2, y2, z2)

        vertex_normals[v0] += normal
        vertex_normals[v1] += normal
        vertex_normals[v2] += normal

        vertex_counts[v0] += 1
        vertex_counts[v1] += 1
        vertex_counts[v2] += 1

    # Нормализация нормалей вершин
    for i in range(len(v)):
        if vertex_counts[i] > 0:
            vertex_normals[i] /= vertex_counts[i]
            length = np.linalg.norm(vertex_normals[i])
            if length > 1e-10:
                vertex_normals[i] /= length

    return vertex_normals

def compute_vertex_intensity(normal, light_dir=np.array([0.0, 0.0, 1.0])):
    dot_product = np.dot(normal, light_dir)
    return dot_product

def drawTriangleTextured(img_mat, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2, 
                        i0, i1, i2, u0, v0, u1, v1, u2, v2, texture):
    xmin, ymin, xmax, ymax = getXYMaxMin(x0, y0, x1, y1, x2, y2)

    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            lambda0, lambda1, lambda2 = calculateBarycentric(x, y, x0, y0, x1, y1, x2, y2)

            if lambda0 >= 0 and lambda1 >= 0 and lambda2 >= 0:
                z = lambda0 * z0 + lambda1 * z1 + lambda2 * z2

                if z < z_buffer[y][x]:
                    # вычисляем координаты текстуры и преобразуем их в пиксельные
                    u_texture = lambda0 * u0 + lambda1 * u1 + lambda2 * u2
                    v_texture = lambda0 * v0 + lambda1 * v1 + lambda2 * v2
                    tx = int(u_texture * (width_texture - 1))
                    ty = int((1 - v_texture) * (height_texture - 1))  # инвертируем V-координату
                
                    texture_color = texture[ty, tx]

                    intensity = -(lambda0 * i0 + lambda1 * i1 + lambda2 * i2)
                    intensity = max(0, min(1, intensity))
                    
                    color = (texture_color * intensity)
                    
                    img_mat[y][x] = color
                    z_buffer[y][x] = z

def draw_triangle_model(img_mat, z_buffer, fv, vertex_normals, vt, texture, ft):
    v0_idx = fv[0] - 1
    v1_idx = fv[1] - 1
    v2_idx = fv[2] - 1

    # текстурные индексы координат не совпадают с вершинами
    vt0_idx = ft[0] - 1
    vt1_idx = ft[1] - 1
    vt2_idx = ft[2] - 1

    x1, y1, z1 = v[v0_idx]
    x2, y2, z2 = v[v1_idx]
    x3, y3, z3 = v[v2_idx]

    sx1, sy1, sz1 = scaleCoords([x1, y1, z1])
    sx2, sy2, sz2 = scaleCoords([x2, y2, z2])
    sx3, sy3, sz3 = scaleCoords([x3, y3, z3])

    i0 = compute_vertex_intensity(vertex_normals[v0_idx])
    i1 = compute_vertex_intensity(vertex_normals[v1_idx])
    i2 = compute_vertex_intensity(vertex_normals[v2_idx])

    u0, v0 = vt[vt0_idx]
    u1, v1 = vt[vt1_idx]
    u2, v2 = vt[vt2_idx]

    drawTriangleTextured(img_mat, z_buffer, sx1, sy1, sz1, sx2, sy2, sz2, sx3, sy3, sz3, 
                        i0, i1, i2, u0, v0, u1, v1, u2, v2, texture)

file = open("model_1.obj")
v = []
fv = []
vt = []
ft = [] # индексы текстурных координат  

for s in file:
    spl = s.split()
    if spl[0] == "v":
        coords = list(map(float, spl[1:]))
        coords[0], coords[1], coords[2] = rotate(coords[0], coords[1], coords[2], 0, pi + pi / 4, 0, 0, -0.05, 0.1)
        v.append(coords)
    if spl[0] == "f":
        v_indices = []
        vt_indices = []
        for sp in spl[1:]:
            parts = sp.split('/')
            v_indices.append(int(parts[0]))
            if len(parts) > 1 and parts[1] != '':
                vt_indices.append(int(parts[1]))
                
        fv.append(v_indices)
        ft.append(vt_indices)
    if spl[0] == "vt":
        coords = list(map(float, spl[1:3]))
        vt.append(coords)

vertex_normals = compute_vertex_normals(v, fv)

img_mat = np.zeros((height, width, 3), dtype=np.uint8)
z_buffer = np.full((height, width), float('inf'))

texture_img = Image.open('bunny-atlas.jpg')
texture = np.array(texture_img)
width_texture = texture.shape[1]  # ширина текстуры
height_texture = texture.shape[0]  # высота текстуры

for i in range(len(fv)):
    pg = fv[i]
    pt = ft[i]
    draw_triangle_model(img_mat, z_buffer, pg, vertex_normals, vt, texture, pt)

img = Image.fromarray(img_mat, mode='RGB')
img = ImageOps.flip(img)
img.save('img_textured.png')
img.show()