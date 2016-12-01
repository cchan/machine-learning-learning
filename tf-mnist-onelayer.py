import tensorflow as tf
import input_data

# Load all the data as one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# In other words, it'll look like this: [image of a 2]: [0 0 1 0 0 0 0 0 0 0]

# Create a placeholder, filled with feed_dict later
x = tf.placeholder(tf.float32, [None, 784])
# 

# Create variables which may be changed
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Define the single-layer softmax NN using matrix multiplication
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Another placeholder, filled with feed_dict later
y_ = tf.placeholder(tf.float32, [None, 10])

# Cross entropy, a fitness measure, 
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Let TensorFlow's magic work, gradient-descent-optimizing the cross-entropy.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Start a session and init all its variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Now run the training step, 1000 times, on random 100-element subsets of the dataset
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Get a vector of whether the one-hot answer matches, for each run.
# The variable y is filled with a lot of answers from before.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

# Reduce the correct_prediction to an accuracy percent.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Run the "accuracy" op, which will feed the data to the session and make things work.
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



