import numpy as np
from vctoolkit import *
from mano_np import MANOModel
from vctoolkit.visgl import TriMeshViewer
from transforms3d.axangles import axangle2mat


def dump_scans():
  with open('./official_model/MANO_LEFT.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
  basis = np.array(data['hands_components'])
  mean = np.array(data['hands_mean'])
  left = np.matmul(np.array(data['hands_coeffs']), basis) + mean
  left = np.reshape(left, [-1, 15, 3])

  with open('./official_model/MANO_RIGHT.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
  basis = np.array(data['hands_components'])
  mean = np.array(data['hands_mean'])
  right = np.matmul(np.array(data['hands_coeffs']), basis) + mean
  right = np.reshape(right, [-1, 15, 3])
  right *= np.array([[[1, -1, -1]]])

  axangles = np.concatenate([left, right])
  print(axangles.shape)

  np.save('./axangles.npy', axangles)


def produce_video():
  axangles = np.load('./axangles.npy')
  mesh = MANOModel('./model.pkl')
  view_mat = np.matmul(
    axangle2mat([1, 0, 0], -np.pi/2), axangle2mat([0, 1, 0], -np.pi/2)
  )
  verts = []
  for a in axangles:
    a = np.concatenate([np.array([[0, 0, 0]]), a], 0)
    v = mesh.set_params(pose_abs=a)
    verts.append(np.matmul(view_mat, v.T).T)

  viewer = TriMeshViewer()
  viewer.run(verts, mesh.faces, './all_scans_view2.avi', 10)
