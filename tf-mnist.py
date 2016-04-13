import tensorflow as tf
import input_data


# Load all the data as one-hot
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create a placeholder for raw data (image), to be filled with feed_dict later
x = tf.placeholder(tf.float32, [None, 784])

# Create variables which may be changed
W1 = tf.Variable(tf.zeros([784,100]))
b1 = tf.Variable(tf.zeros([100]))
W2 = tf.Variable(tf.zeros([100,10]))
b2 = tf.Variable(tf.zeros([10]))

# Define the two-layer softmax NN using matrix multiplication
y = tf.nn.softmax(tf.matmul(x, W1) + b1)
prediction = tf.nn.softmax(tf.matmul(y, W2) + b2)

# Another placeholder for raw data (class assigned), filled with feed_dict later
label = tf.placeholder(tf.float32, [None, 10])

# fitness measure
cost = tf.reduce_mean(tf.square(label- prediction))

# Let TensorFlow's magic work, gradient-descent-minimizing the cost.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)



# Start a session and init all its variables
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Now run the training step, 1000 times, on random 100-element subsets of the dataset
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, label: batch_ys})

# Get a vector of whether the one-hot answer matches, for each run.
# The variable prediction is filled with a lot of answers from before.
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(label, 1))

# Reduce the correct_prediction to an accuracy percent.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Run the "accuracy" op, which will feed the data to the session and make things work.
print sess.run(accuracy, feed_dict={x: mnist.test.images, label: mnist.test.labels})

