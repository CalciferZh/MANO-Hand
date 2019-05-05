import pickle
import numpy as np


def dump_model(src_path, dst_path):
  with open(src_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    output = {}
    output['pose_pca_basis'] = np.array(data['hands_components'])
    output['pose_pca_mean'] = np.array(data['hands_mean'])
    output['J_regressor'] = np.array(data['J_regressor'])
    output['weights'] = np.array(data['weights'])
    # how mesh is affected by pose
    output['mesh_pose_basis'] = np.array(data['posedirs'])
    output['mesh_shape_basis'] = np.array(data['shapedirs'])
    output['mesh_template'] = np.array(data['v_template'])
    output['faces'] = np.array(data['f'])
    output['parent'] = data['kintree_table'][0].tolist()
    output['parent'][0] = None

  with open(dst_path, 'wb') as f:
    pickle.dump(output, f)


if __name__ == '__main__':
  dump_model('./official_model/model_left.pkl', './model_left.pkl')

