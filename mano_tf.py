import numpy as np
import tensorflow as tf
import pickle


def axangle_to_mat(axangle):
  angle = tf.maximum(
    tf.linalg.norm(axangle, axis=-1, keepdims=True), np.finfo(np.float64).eps
  )
  axis = axangle / angle
  x, y, z = tf.split(axis, 3, -1)
  cos = tf.cos(angle)
  o = tf.zeros([angle.shape[0], 1], tf.float64)
  mat = tf.reshape(tf.stack([o, -z, y, z, o, -x, -y, x, o], -1), [-1, 3, 3])
  I = tf.eye(3, dtype=tf.float64)
  dot = tf.matmul(tf.expand_dims(axis, 2), tf.expand_dims(axis, 1))
  R = tf.expand_dims(cos, -1) * tf.expand_dims(I, 0) + \
      tf.expand_dims((1 - cos), -1) * dot + \
      tf.expand_dims(tf.sin(angle), -1) * mat

  return R


def with_zeros(x):
  """
  Nx3x4 + [0, 0, 0, 1] -> Nx4x4
  """
  n = x.get_shape().as_list()[0]
  o = np.tile(
    np.array([0, 0, 0, 1], dtype=np.float64).reshape([1, 1, 4]), [n, 1, 1]
  )
  return tf.concat([x, o], axis=1)


def pack(x):
  """
  0 + Nx16x4x1 -> Nx16x4x4
  """
  n = x.get_shape().as_list()[0]
  o = tf.zeros([n, 16, 4, 3], dtype=tf.float64)
  return tf.concat([o, x], 3)


def mano_model(pose, model_path):
  """
  Compute mano's verts and joints given pose parameter in a batch manner.

  Parameters
  ----------
  pose: A batch of poses, shape [N, 16, 3]

  model_path: Path to load MANO model, dumped by `dump_model.py`
  """
  with open(model_path, 'rb') as f:
    params = pickle.load(f)
  j_regressor = tf.constant(params['J_regressor'].astype(np.float64)) # 16x778
  skinning_weights = params['skinning_weights'].astype(np.float64) # 778x16
  mesh_template = params['mesh_template'].astype(np.float64) # 778x3
  parents = params['parents']
  N = pose.shape[0]
  n_joints = 16

  joints = tf.tile(
    tf.expand_dims(tf.matmul(j_regressor, mesh_template), 0), [N, 1, 1]
  ) # Nx16x3
  pose = tf.reshape(pose, [-1, 3])
  rot_mat = tf.reshape(axangle_to_mat(pose), [N, n_joints, 3, 3]) # Nx16x3x3

  G = [None] * n_joints
  G[0] = with_zeros(
    tf.concat(
      [rot_mat[:, 0], tf.expand_dims(joints[:, 0], 2)], axis=2
    )
  )
  for j in range(1, n_joints):
    p = parents[j]
    G[j] = tf.matmul(
      G[p],
      with_zeros(
        tf.concat(
          [rot_mat[:, j], tf.expand_dims(joints[:, j] - joints[:, p], 2)],
          axis=2
        )
      )
    )
  G = tf.stack(G, axis=1) # Nx16x4x4
  G = G - pack(
    tf.matmul(
      G,
      tf.expand_dims(
        tf.concat([joints, tf.zeros([N, n_joints, 1], dtype=tf.float64)], 2),
        -1
      )
    )
  )

  T = tf.einsum(
    'vj, njab -> nvab',
    tf.constant(skinning_weights, dtype=tf.float64), G
  ) # Nx778x4x4

  rest_verts = tf.concat(
    [
      mesh_template,
      tf.constant(np.ones((mesh_template.shape[0], 1)), dtype=tf.float64)
    ], 1
  ) # 778x4

  posed_verts = tf.einsum('nvhw, vw -> nvh', T, rest_verts)[..., :3]
  return posed_verts


if __name__ == '__main__':
  mano_model(None, './model.pkl')
