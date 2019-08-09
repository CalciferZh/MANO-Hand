import mano_np
import mano_tf
import numpy as np
import tensorflow as tf
from vctoolkit import Timer


def is_close(a, b):
  if a.shape != b.shape:
    print('different shape: %s vs %s' % (str(a.shape), str(b.shape)))
    return False
  return np.allclose(a, b)


def test():
  #NOTE: before testing make sure in tf all dtype is float64
  batch_size = 4096
  np.random.seed(9608)
  axangles = np.random.uniform(size=[batch_size, 16, 3])
  timer = Timer()

  pose_ph = tf.placeholder(tf.float64, [batch_size, 16, 3])
  vert_op = mano_tf.mano_model(pose_ph, './model.pkl')
  sess = tf.Session()
  timer.tic()
  tf_verts = sess.run(vert_op, {pose_ph: axangles})
  timer.toc()
  print(timer.interval)

  np_verts = []
  np_model = mano_np.MANOModel('./model.pkl')
  timer.tic()
  for i in range(batch_size):
    verts = np_model.set_params(pose_abs=axangles[i])
    np_verts.append(verts)
  timer.toc()
  np_verts = np.stack(np_verts)
  print(timer.interval)

  print(np_verts.shape)
  print(tf_verts.shape)

  print(is_close(np_verts, tf_verts))


if __name__ == '__main__':
  test()
