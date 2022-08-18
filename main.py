import tensorflow as tf
import numpy as np

def onehot_encode(image, label):
    return image, tf.one_hot(label, depth=10, dtype=tf.float32) # hardcord

def binary_encode(labels):
    bit_length = 4
    binary_labels = np.zeros((len(labels), bit_length), dtype=np.float32)
    for i in range(len(labels)):
        label = labels[i]
        for l in reversed(range(bit_length)): # 大きいビットからチェック
            if 0 <= label - 2 ** l: # 着目ビット値を引いても0以上 = ビットを立てられる
                binary_labels[i, l] = 1.0
                label = label - 2 ** l
    return binary_labels

batch_size = 256

mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()
image_train, image_test = image_train / 255.0, image_test / 255.0
image_train, image_test = image_train[:,:,:,tf.newaxis].astype("float32"), image_test[:,:,:,tf.newaxis].astype("float32")

oh_train_ds = tf.data.Dataset.from_tensor_slices((image_train, label_train)).map(onehot_encode).shuffle(10000).batch(batch_size)
oh_test_ds = tf.data.Dataset.from_tensor_slices((image_test, label_test)).map(onehot_encode).batch(32)

bin_train_ds = tf.data.Dataset.from_tensor_slices((image_train, binary_encode(label_train))).shuffle(10000).batch(batch_size)
bin_test_ds = tf.data.Dataset.from_tensor_slices((image_test, binary_encode(label_test))).batch(32)


EPOCHS = 10

# def my_cnn(inputs, output_dim):
#     # imitate resnet
#     x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.Conv2D(128, kernel_size=3)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.LeakyReLU()(x)
# 
#     x = tf.keras.layers.GlobalAveragePooling2D()(x)
# 
#     x = tf.keras.layers.Dense(128)(x)
#     x = tf.keras.layers.LeakyReLU()(x)
#     x = tf.keras.layers.Dense(output_dim)(x)
# 
#     return x

def my_cnn(inputs, output_dim):
    x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dense(output_dim)(x)

    return x

oh_inputs = tf.keras.Input(shape=(28, 28, 1))
oh_output = tf.keras.layers.Activation("softmax")(my_cnn(oh_inputs, 10))
oh_model = tf.keras.models.Model(inputs=oh_inputs, outputs=oh_output)

oh_loss = tf.keras.losses.CategoricalCrossentropy()
oh_optimizer = tf.keras.optimizers.Adam()

oh_train_loss = tf.keras.metrics.Mean(name='oh_train_loss')
oh_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='oh_train_accuracy')
oh_test_loss = tf.keras.metrics.Mean(name='oh_test_loss')
oh_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='oh_test_accuracy')


bin_inputs = tf.keras.Input(shape=(28, 28, 1))
bin_output = tf.keras.layers.Activation("sigmoid")(my_cnn(bin_inputs, 4))
bin_model = tf.keras.models.Model(inputs=bin_inputs, outputs=bin_output)

bin_loss = tf.keras.losses.BinaryCrossentropy()
bin_optimizer = tf.keras.optimizers.Adam()

bin_train_loss = tf.keras.metrics.Mean(name='bin_train_loss')
bin_train_accuracy = tf.keras.metrics.Accuracy(name='bin_train_accuracy')
bin_test_loss = tf.keras.metrics.Mean(name='bin_test_loss')
bin_test_accuracy = tf.keras.metrics.Accuracy(name='bin_test_accuracy')


@tf.function()
def oh_train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = oh_model(images, training=True)
        loss = oh_loss(labels, preds)
    gradients = tape.gradient(loss, oh_model.trainable_variables)
    oh_optimizer.apply_gradients(zip(gradients, oh_model.trainable_variables))

    oh_train_loss(loss)
    oh_train_accuracy(labels, preds)

@tf.function()
def oh_test_step(images, labels):
    preds = oh_model(images, training=False)
    loss = oh_loss(preds, labels)

    oh_test_loss(loss)
    oh_test_accuracy(labels, preds)

@tf.function()
def bin_train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = bin_model(images, training=True)
        loss = bin_loss(labels, preds)
    gradients = tape.gradient(loss, bin_model.trainable_variables)
    bin_optimizer.apply_gradients(zip(gradients, bin_model.trainable_variables))

    bin_train_loss(loss)
    bin_train_accuracy(labels, preds)

@tf.function()
def bin_test_step(images, labels):
    preds = bin_model(images, training=False)
    loss = bin_loss(preds, labels)

    bin_test_loss(loss)
    bin_test_accuracy(labels, preds)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    oh_train_loss.reset_states()
    oh_train_accuracy.reset_states()
    oh_test_loss.reset_states()
    oh_test_accuracy.reset_states()

    for images, labels in oh_train_ds:
        oh_train_step(images, labels)

    for test_images, test_labels in oh_test_ds:
        oh_test_step(test_images, test_labels)

    bin_train_loss.reset_states()
    bin_train_accuracy.reset_states()
    bin_test_loss.reset_states()
    bin_test_accuracy.reset_states()

    for images, labels in bin_train_ds:
        bin_train_step(images, labels)

    for test_images, test_labels in bin_test_ds:
        bin_test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, \n'
        f'\tonehot Loss: {oh_train_loss.result()}, '
        f'onehot Accuracy: {oh_train_accuracy.result() * 100}, '
        f'onehot Test Loss: {oh_test_loss.result()}, '
        f'onehot Test Accuracy: {oh_test_accuracy.result() * 100}\n'
        f'\tbinary Loss: {bin_train_loss.result()}, '
        f'binary Accuracy: {bin_train_accuracy.result() * 100}, '
        f'binary Test Loss: {bin_test_loss.result()}, '
        f'binary Test Accuracy: {bin_test_accuracy.result() * 100}'
    )