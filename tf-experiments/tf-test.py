import tensorflow as tf

# Say hello.
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

# Some simple math.
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))
