import numpy as np
import pickle


class MANOModel():
  """
  MANO model (hand only).
  * Call `set_params` to set pose and shape params.
  * Call `export_obj` to export to obj file.
  """
  def __init__(self, model_path):
    """
    Parameter
    ---------
    model_path: Path to model parameters dumped by `dump_model.py`.
    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.pose_pca_basis = params['pose_pca_basis']
      self.pose_pca_mean = params['pose_pca_mean']

      self.J_regressor = params['J_regressor']

      self.skinning_weights = params['skinning_weights']

      self.mesh_pose_basis = params['mesh_pose_basis']
      self.mesh_shape_basis = params['mesh_shape_basis']
      self.mesh_template = params['mesh_template']

      self.faces =  params['faces']

      self.parents = params['parents']

    self.n_joints = 16
    self.n_shape_params = 10

    self.pose = np.zeros((self.n_joints, 3))
    self.shape = np.zeros(self.n_shape_params)
    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def set_params(self, pose_abs=None, pose_pca=None, shape=None):
    """
    Parameters
    ---------
    pose_abs: "Absolute" pose. Relative rotation of each joint in axis-angle
    (in radian). Shape [16, 3].

    pose_pca: Parameterized pose, coefficients for PCA-ed pose. Shape [N],
    0 < N <= 45.

    shape: Coefficients for PCA-ed shape. Shape [N], 0 < N <= 10.

    Return
    ------
    Updated vertices.
    """
    if pose_abs is not None:
      self.pose = pose_abs
    if pose_pca is not None:
      self.pose = np.dot(
        np.expand_dims(pose_pca, 0), self.pose_pca_basis[:pose_pca.shape[0]]
      )[0] + self.pose_pca_mean
      self.pose = np.reshape(self.pose, [self.n_joints - 1, 3])
      self.pose = np.concatenate([np.zeros([1, 3]), self.pose], 0)
    if shape is not None:
      self.shape = shape
    self.update()
    return self.verts.copy()

  def update(self):
    # how shape affect mesh
    v_shaped = self.mesh_template + self.mesh_shape_basis.dot(self.shape)
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0), (self.R.shape[0] - 1, 3, 3)
    )
    # how pose affect mesh
    v_posed = v_shaped + self.mesh_pose_basis.dot((self.R[1:] - I_cube).ravel())
    # world transformation of each joint
    G = np.empty((self.n_joints, 4, 4))
    G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
    for i in range(1, self.n_joints):
      G[i] = G[self.parents[i]].dot(self.with_zeros(
          np.hstack([
            self.R[i],
            (self.J[i, :] - self.J[self.parents[i], :]).reshape([3, 1])
          ])
      ))
    # remove the transformation due to the rest pose
    G = G - self.pack(np.matmul(
        G,
        np.hstack([self.J, np.zeros([self.n_joints, 1])]) \
          .reshape([self.n_joints, 4, 1])
    ))
    # transformation of each vertex
    T = np.tensordot(self.skinning_weights, G, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    self.verts = \
      np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]

  def rodrigues(self, r):
    """
    Rodrigues' rotation formula that turns axis-angle vector into rotation
    matrix in a batch-ed manner.

    Parameter:
    ----------
    r: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Return:
    -------
    Rotation matrix of shape [batch_size, 3, 3].
    """
    theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
    # avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).eps)
    r_hat = r / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
      z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
      r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
      -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
    ).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0), [theta.shape[0], 3, 3]
    )
    A = np.transpose(r_hat, axes=[0, 2, 1])
    B = r_hat
    dot = np.matmul(A, B)
    R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return R

  def with_zeros(self, x):
    """
    Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Parameter:
    ---------
    x: Matrix to be appended.

    Return:
    ------
    Matrix after appending of shape [4,4]

    """
    return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, x):
    """
    Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
    manner.

    Parameter:
    ----------
    x: Matrices to be appended of shape [batch_size, 4, 1]

    Return:
    ------
    Matrix of shape [batch_size, 4, 4] after appending.

    """
    return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

  def export_obj(self, path):
    """
    Save the SMPL model into .obj file.

    Parameter:
    ---------
    path: Path to save.

    """
    with open(path, 'w') as fp:
      for v in self.verts:
        fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
      for f in self.faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))



if __name__ == '__main__':
  model = MANOModel('./model.pkl')
  # np.random.seed(9608)
  # pose_pca = np.random.uniform(-1, 1, 12)
  # model.set_params(pose_pca=pose_pca)
  model.export_obj('./hand.obj')
