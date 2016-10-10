from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#Here x and y_ aren't specific values
#Here we assign it a shape of [None, 784], where 784 is the dimensionality of a single flattened 28 by 28 pixel
#None indicates that the first dimension, corresponding to the batch size, can be of any size
#y_ will also consist of a 2d tensor, where each row is a one-hot 10-dimensional vector indicating which digit class (zero through nine)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
#they are placehoders because we will give them values when  learnng from images

W = tf.Variable(tf.zeros([784,10])) #initialize W,b (what you want to know)
b = tf.Variable(tf.zeros([10]))# to zeros

sess.run(tf.initialize_all_variables())

# implement our regression mode

#y =  softmaxvalue structure that will adapt W and b
y = tf.nn.softmax(tf.matmul(x,W) + b)

#We can specify a loss function just as easily
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#Note that tf.reduce_sum sums across all classes and tf.reduce_mean takes the average over these sums.

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Now our model is in train_step, now we have to train it

# We load 100 training examples in each training iteration
for i in range(1000):
  batch = mnist.train.next_batch(100) # 100 imagenes de misnt
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})#entrenas con x=batch[0] donde etan los pixeles, y=batch[1] con la etiqueta
  #Ademas de que en x e y_ se van a guardar las iamgenes con las que has entrenado y al final tendras en y_ los ultimos
  #valores de y_ predichos y las ultiams x con las que has entrenado (y es un vector de 10 elementos, donde dentro tendra
  #probabilidades de ser una clase u otra

#Defines valor de la prediccion correcta, que es si la supuesta etiqueta de verdad es lo que se predice
#y es lo que sale del softmax, y_ es la etiqueta de verdad
# tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean.
# # For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Ahora evaluas accuracy (que es una operation de tensor, con las imagenes de test
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
