import argparse
import re
import pickle
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument("file", help="point clouds file path")

def load_h5(h5_filename):
  f = h5py.File(h5_filename)
  data = f['data'][:]
  label = f['label'][:]
  return (data, label)

def pyplot_draw_point_cloud(points):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


args = parser.parse_args()
if re.match(r'.*\.off', args.file):
    with open(args.file, "r") as f:
        lines = f.readlines()
        points = []
        for line in lines[1:]:
            line = [float(i) for i in line.split()]
            if len(line)<=3:
                points.append(line)
            else:
                break
        points = np.asarray(points)
        assert points.shape[1] == 3

elif re.match(r'.*\.pkl', args.file):
    with open(args.file, "rb") as f:
        points = pickle.load(f)

pyplot_draw_point_cloud(points)
