import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

tf.__version__

# loading & preprocessing the dataset

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
# we're generating pics based on train set and they will be evaluated so we don't need test set

x_train.shape

y_train.shape

i = np.random.randint(0, 60000)
print(y_train[i])
plt.imshow(x_train[i], cmap='gray');

# normalizing and reshaping the data: when working with tensorflow and ConvNets we need another dimension added

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

x_train.shape

x_train[0].min(), x_train[0].max()

x_train = (x_train - 127.5) / 127.5   # this normalization converts the data so it ranges from -1 to 1
# /255 made the data range from 0 to 1

x_train[0].min(), x_train[0].max()

buffer_size = 60000  # number of pictures
batch_size = 256

type(x_train)

x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)
# converting the data to tensorflow type

type(x_train)

x_train

# building the generator part of the GAN

def build_generator():
  network = tf.keras.Sequential()

  network.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,))) # 7*7*256 cuz the tensorflow website has recommended so, we're not gonna predict any value so we don't use bias
  network.add(layers.BatchNormalization())
  network.add(layers.LeakyReLU())  # leaky relu covers values (-1, 1)

  network.add(layers.Reshape((7, 7, 256)))  # changing the vector to matrix/ tensor

  # 7 * 7 * 128
  network.add(layers.Conv2DTranspose(128, (5, 5), padding='same', use_bias=False)) # to increase the dimension of data, the opposite of normal conv2D, 128 = num of filters, (5, 5) = kernel size
  network.add(layers.BatchNormalization())
  network.add(layers.LeakyReLU())

  # 14 * 14 * 64 (size increases cuz conv2d transpose works exactly the opposite of conv2d on layer dimensions)
  network.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
  network.add(layers.BatchNormalization())
  network.add(layers.LeakyReLU())

  # 28 * 28 * 1
  network.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')) # tanh covers data in range (-1, 1)

  network.summary()
  return network

generator = build_generator()

noise = tf.random.normal([1, 100])  # generating random noise by use of normal distribution in size a vector with 100 random numbers

noise

# we're gonna send the noise/ random numbers to the generator

generated_image = generator(noise, training=False) # we only train to adjust weights, here we only generate pictures from noise

generated_image.shape

plt.imshow(generated_image[0, :, :, 0], cmap='gray'); # [0, :, :, 0] means that we only get the information in ":" columns

# building the discriminator

def build_discriminator():
  network = tf.keras.Sequential()

  # 14 * 14 * 64
  network.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
  network.add(layers.LeakyReLU())
  network.add(layers.Dropout(0.3))

  # 7 * 7 * 128
  network.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
  network.add(layers.LeakyReLU())
  network.add(layers.Dropout(0.3))

  network.add(layers.Flatten())  # flattening the data to vector
  network.add(layers.Dense(1))  # one last unit in the output

  network.summary()
  return network

discriminator = build_discriminator()

discriminator(generated_image, training=False)

# calculating the loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
 # from_logits=True to turn the logits that discriminator created to statistics

def discriminator_loss(expected_output, fake_output):
  real_loss = cross_entropy(tf.ones_like(expected_output), expected_output) # ones_like(expected_output) because the real image's values are close to 1 and we compare them to a matrix of ones
  fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)  # zeros_like(fake_output) because the fake image's values are close to 0 and we compare them to a matrix of zeros
  total_loss = real_loss + fake_loss
  return total_loss

def generator_loss(fake_output):
  return cross_entropy(tf.ones_like(fake_output), fake_output) # ones_like(fake_output) cause we're comparing the fake image loss to 1 to see how well it's doing

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001) # when coding a GAN it's a good idea to set the learning_rate to a very small value
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

# training the model (GAN)

epochs = 50
noise_dim = 100
num_images_to_generate = 16

batch_size, noise_dim

@tf.function # to store the values of variables(gradients) globally, since we're gonna run the model several epochs
def train_steps(images):
  noise = tf.random.normal([batch_size, noise_dim])
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(noise, training = True)  # training=True -> we want to update the weights

    expected_output = discriminator(images, training = True)
    fake_output = discriminator(generated_images, training = True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(expected_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))  # zip to count 2 variables as 1
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

test_images = tf.random.normal([num_images_to_generate, noise_dim])

test_images.shape

def train(dataset, epochs, test_images):
  for epoch in range(epochs):
    for image_batch in dataset:
      #print(image_batch.shape)
      train_steps(image_batch)

    print('Epoch: ', epoch + 1)
    generated_images = generator(test_images, training = False)
    fig = plt.figure(figsize=(10,10))
    for i in range(generated_images.shape[0]):
      plt.subplot(4,4,i+1)
      plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
    plt.show()

train(x_train, epochs, test_images)





