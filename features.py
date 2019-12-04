import sys, os, operator, time, math, datetime, json
import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations

DATA = 'data'
NUM = 15

def generate(path):
    start = time.time()
    vec1s = []
    vec2s = []
    vec3s = []
    vec4s = []
    for dir in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(os.path.join(DATA, dir))):
            points = extract_points(os.path.join(DATA, dir, filename))
            points = impute_occlusions(points)
            points = select_points(points)
            vec1s.append(np.append(points.flatten(), int(dir)))
            vec2s.append(np.append(center(points), int(dir)))
            vec3s.append(np.append(distance(points), int(dir)))
            vec4s.append(np.append(angles(points), int(dir)))
    mat1 = np.array(vec1s)
    mat2 = np.array(vec2s)
    mat3 = np.array(vec3s)
    mat4 = np.array(vec4s)
    
    np.savetxt('mat1.csv', mat1, delimiter=',', fmt = '%.6f')    
    np.savetxt('mat2.csv', mat2, delimiter=',', fmt = '%.6f')    
    np.savetxt('mat3.csv', mat3, delimiter=',', fmt = '%.6f')    
    np.savetxt('mat4.csv', mat4, delimiter=',', fmt = '%.6f')    

    end = time.time()
    print("Generated features in " + str(datetime.timedelta(seconds=int(end-start))))

def extract_points(filename):
    with open(filename) as file:
        jtext = json.load(file)
        points = np.array(jtext['people'][0]['pose_keypoints_2d'])
        points = np.resize(points, (25, 3))
        points = np.delete(points, 2, 1)
    return points

def impute_occlusions(points):
    right = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24]
    left = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
    for i in range(len(right)):
        if np.sum(points[right[i]]) == 0:
            points[right[i]] = points[left[i]]
        if np.sum(points[left[i]]) == 0:
            points[left[i]] = points[right[i]] 
    return points

def select_points(points):
    points = points[:15]
    return points

def center(points):
    points = points - points[0]
    vec = points.flatten()
    return vec

def distance(points):
    mat = distance_matrix(points, points)
    iu = np.triu_indices(NUM, 1)
    mat = mat[iu]
    vec = mat.flatten()
    return vec

def angles(points):
    comb = combinations(range(0, NUM), 3)
    mat = distance_matrix(points, points)
    angs = []
    for i in comb:
        a = mat[i[0]][i[1]]
        b = mat[i[0]][i[2]]
        c = mat[i[1]][i[2]]
        angs.append(np.arccos((a*a + b*b - c*c)/(2*a*b)))
        angs.append(np.arccos((a*a + c*c - b*b)/(2*a*c)))
        angs.append(np.arccos((b*b + c*c - a*a)/(2*b*c)))
    vec = np.array(angs)
    return vec

generate(DATA)
