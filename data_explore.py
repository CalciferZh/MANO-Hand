import numpy as np
from vctoolkit import *
from mano_np import MANOModel
from vctoolkit.visgl import TriMeshViewer
from transforms3d.axangles import axangle2mat


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
