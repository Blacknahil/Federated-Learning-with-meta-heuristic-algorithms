import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

os.environ["TF_USE_LEGACY_KERAS"] = "1"
tf.compat.v1.disable_resource_variables()
# Compatibility shim: some environments (Keras 3) remove
# `tf.compat.v1.layers.dense`. Monkey-patch it to use
# `tf.keras.layers.Dense` so older code (mclr.py) continues to work
# without modifying model source files.
try:
    has_dense = hasattr(tf.compat.v1, 'layers') and hasattr(tf.compat.v1.layers, 'dense')
except Exception:
    has_dense = False

if not has_dense:
    # create a simple namespace for layers if missing
    import types
    layers_ns = getattr(tf.compat.v1, 'layers', types.SimpleNamespace())

    def dense_compat(inputs, units, kernel_regularizer=None, name=None, **kwargs):
        layer = tf.keras.layers.Dense(units=units, kernel_regularizer=kernel_regularizer, name=name)
        return layer(inputs)

    layers_ns.dense = dense_compat
    tf.compat.v1.layers = layers_ns

from flearn.utils.model_utils import read_data
from flearn.models.mnist.mclr import Model
from flearn.optimizer.pgd import PerturbedGradientDescent

def load_central_dataset(train_dir, test_dir):
    clients, groups, train_data, test_data = read_data(train_dir, test_dir)
    # merge all client train data
    X_list, y_list = [], []
    for cid, d in train_data.items():
        X_list.append(np.array(d['x']))
        y_list.append(np.array(d['y']))
    X_train = np.vstack(X_list)
    y_train = np.concatenate(y_list)

    X_list, y_list = [], []
    for cid, d in test_data.items():
        X_list.append(np.array(d['x']))
        y_list.append(np.array(d['y']))
    X_test = np.vstack(X_list)
    y_test = np.concatenate(y_list)

    return {'x': X_train, 'y': y_train}, {'x': X_test, 'y': y_test}

def main():
    train_path = os.path.join('data', 'mnist', 'data', 'train')
    test_path = os.path.join('data', 'mnist', 'data', 'test')
    train, test = load_central_dataset(train_path, test_path)

    lr = 0.03
    mu = 1.0
    epochs = 20
    batch_size = 32

    optimizer = PerturbedGradientDescent(learning_rate=lr, mu=mu)
    model = Model(10, optimizer, seed=0)

    print('Central training on full MNIST: epochs=%d, batch_size=%d, lr=%.4f, mu=%.4f' % (epochs, batch_size, lr, mu))
    soln, comp = model.solve_inner(train, num_epochs=epochs, batch_size=batch_size)
    acc, loss = model.test(test)
    print('Test accuracy (central):', acc)
    print('Test loss (central):', loss)

    model.close()

if __name__ == '__main__':
    main()
