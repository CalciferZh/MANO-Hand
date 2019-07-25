import pickle
import numpy as np


def dump_model(src_path, dst_path):
  with open(src_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    output = {}
    output['pose_pca_basis'] = np.array(data['hands_components'])
    output['pose_pca_mean'] = np.array(data['hands_mean'])
    output['J_regressor'] = data['J_regressor'].toarray()
    output['skinning_weights'] = np.array(data['weights'])
    # how mesh is affected by pose
    output['mesh_pose_basis'] = np.array(data['posedirs'])
    output['mesh_shape_basis'] = np.array(data['shapedirs'])
    output['mesh_template'] = np.array(data['v_template'])
    output['faces'] = np.array(data['f'])
    output['parents'] = data['kintree_table'][0].tolist()
    output['parents'][0] = None

  with open(dst_path, 'wb') as f:
    pickle.dump(output, f)


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


if __name__ == '__main__':
  dump_scans()
