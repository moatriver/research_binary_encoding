import tensorflow as tf
import numpy as np

class BinaryTrainer():
    def __init__(self):
        self.init_datasets()
        
        self.EPOCHS = 10

        bin_inputs = tf.keras.Input(shape=(28, 28, 1))
        bin_output = tf.keras.layers.Activation("sigmoid")(self.my_cnn(bin_inputs, 4))
        self.bin_model = tf.keras.models.Model(inputs=bin_inputs, outputs=bin_output)

        self.bin_loss = tf.keras.losses.BinaryCrossentropy()
        self.bin_optimizer = tf.keras.optimizers.Adam()

        self.bin_train_loss = tf.keras.metrics.Mean(name='bin_train_loss')
        self.bin_train_accuracy = tf.keras.metrics.BinaryAccuracy(name='bin_train_accuracy')
        self.bin_test_loss = tf.keras.metrics.Mean(name='bin_test_loss')
        self.bin_test_accuracy = tf.keras.metrics.BinaryAccuracy(name='bin_test_accuracy')

    def binary_encode(self, labels):
        bit_length = 4
        binary_labels = np.zeros((len(labels), bit_length), dtype=np.float32)
        for i in range(len(labels)):
            label = labels[i]
            for l in reversed(range(bit_length)): # 大きいビットからチェック
                if 0 <= label - 2 ** l: # 着目ビット値を引いても0以上 = ビットを立てられる
                    binary_labels[i, l] = 1.0
                    label = label - 2 ** l
        return binary_labels

    def init_datasets(self):
        batch_size = 256
        mnist = tf.keras.datasets.mnist

        (image_train, label_train), (image_test, label_test) = mnist.load_data()
        image_train, image_test = image_train / 255.0, image_test / 255.0
        image_train, image_test = image_train[:,:,:,tf.newaxis].astype("float32"), image_test[:,:,:,tf.newaxis].astype("float32")

        self.bin_train_ds = tf.data.Dataset.from_tensor_slices((image_train, self.binary_encode(label_train))).shuffle(10000).batch(batch_size)
        self.bin_test_ds = tf.data.Dataset.from_tensor_slices((image_test, self.binary_encode(label_test))).batch(32)

    def my_cnn(self, inputs, output_dim):
        x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(output_dim)(x)

        return x

    
    @tf.function()
    def bin_train_step(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self.bin_model(images, training=True)
            loss = self.bin_loss(labels, preds)
        gradients = tape.gradient(loss, self.bin_model.trainable_variables)
        self.bin_optimizer.apply_gradients(zip(gradients, self.bin_model.trainable_variables))
    
        self.bin_train_loss(loss)
        self.bin_train_accuracy(labels, preds)
    
    @tf.function()
    def bin_test_step(self, images, labels):
        preds = self.bin_model(images, training=False)
        loss = self.bin_loss(preds, labels)
    
        self.bin_test_loss(loss)
        self.bin_test_accuracy(labels, preds)

    def train(self):
        for epoch in range(self.EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.bin_train_loss.reset_states()
            self.bin_train_accuracy.reset_states()
            self.bin_test_loss.reset_states()
            self.bin_test_accuracy.reset_states()

            for images, labels in self.bin_train_ds:
                self.bin_train_step(images, labels)

            for test_images, test_labels in self.bin_test_ds:
                self.bin_test_step(test_images, test_labels)

            print(
                f'Epoch {epoch + 1}, '
                f'\tbinary Loss: {self.bin_train_loss.result()}, '
                f'binary Accuracy: {self.bin_train_accuracy.result() * 100}, '
                f'binary Test Loss: {self.bin_test_loss.result()}, '
                f'binary Test Accuracy: {self.bin_test_accuracy.result() * 100}'
            )