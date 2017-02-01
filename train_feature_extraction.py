import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
with open('train.p', mode='rb') as f:
    train = pickle.load(f)

X_train = train['features']
y_train = train['labels']
n_input = len(X_train)
n_classes = len(set(y_train))

# TODO: Split data into training and validation sets.

X_input, X_valid, y_input, y_valid = train_test_split(X_train, y_train, test_size=0.20)

# TODO: Define placeholders and resize operation.

X = tf.placeholder(tf.float32, [None, 32, 32, 3], 'X')
y = tf.placeholder(tf.int32, [None], 'y')
y_one_hot = tf.one_hot(y, n_classes)
X_resized = tf.image.resize_images(X, [227, 227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(X_resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = [fc7.get_shape().as_list()[-1], n_classes]
W = tf.Variable(tf.truncated_normal(shape))
b = tf.Variable(tf.truncated_normal([n_classes]))
fc8 = tf.nn.xw_plus_b(fc7, W, b)
logits = fc8

# TODO: Define loss, training, accuracy operations.
learning_rate = 0.0001
epoch = 2
batch_size = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1)), tf.float32))

# TODO: Train and evaluate the feature extraction model.

def evaluate_metrics(XX, yy):
    sess = tf.get_default_session()
    len_X = len(XX)
    acc, total_acc, l, n_batches = 0., 0., 0., 0.
    for b in range(0, len_X, batch_size):
        end = b + batch_size
        batch_X, batch_y = XX[b:end], yy[b:end]
        l += sess.run(loss, feed_dict = {X: batch_X, y: batch_y})
        total_acc += sess.run(accuracy, feed_dict = {X: batch_X, y: batch_y})
        n_batches += 1
    return total_acc/n_batches, l


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epoch):
        X_shuffled, y_shuffled = shuffle(X_input, y_input)
        for b in range(0, X_shuffled.shape[0], batch_size):
            end = b + batch_size
            batch_X, batch_y = X_shuffled[b:end], y_shuffled[b:end]
            sess.run(training_operation, feed_dict={X: batch_X, y: batch_y})
        training_accuracy, training_loss = evaluate_metrics(X_input, y_input)
        validation_accuracy, validation_loss = evaluate_metrics(X_valid, y_valid)
        print("metrics at epoch ", e, training_accuracy, training_loss, validation_accuracy, validation_loss)

