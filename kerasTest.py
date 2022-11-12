import tensorflow as tf

#helper libs
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#indecies relate to labels of data, with the actual classifications being the strings inside the list. 
#Maybe use an np.array for this instead?
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(f'Train images: {train_images.shape}\t{len(train_images)}')
#print(f'Train images: {type(train_images)}')
#print(f'Train labels: {train_labels.shape}\t{len(train_labels)}')
#print(f'Train labels: {type(train_labels)}')


#scale the pixel values from 0-255 to 0-1
#remember since these are np.arrays the division applies to each element
train_images = train_images / 255
test_images = test_images/255

#display the first 25 training images to verify the scaling worked
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()


#time to build the model:
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
#    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#need to compile the model to set more params
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

#TRAINING TIME
model.fit(train_images, train_labels, epochs=7)

#TESTING
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f'\nTest Accuracy: {test_acc}')

#lets make some predictions!
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)

#print(f'raw prediction form: {predictions[0]}')
#print(f'most confident prediction: {np.argmax(predictions[0])}')


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_labeli.set_color('blue')

i=0
for i in range(10):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions[i],  test_labels)
    plt.show()

