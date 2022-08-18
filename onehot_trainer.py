import tensorflow as tf

class OneHotTrainer():
    def __init__(self):
        self.init_datasets()

        self.EPOCHS = 10
        
        oh_inputs = tf.keras.Input(shape=(28, 28, 1))
        oh_output = tf.keras.layers.Activation("softmax")(self.my_cnn(oh_inputs, 10))
        self.oh_model = tf.keras.models.Model(inputs=oh_inputs, outputs=oh_output)

        self.oh_loss = tf.keras.losses.CategoricalCrossentropy()
        self.oh_optimizer = tf.keras.optimizers.Adam()

        self.oh_train_loss = tf.keras.metrics.Mean(name='oh_train_loss')
        self.oh_train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='oh_train_accuracy')
        self.oh_test_loss = tf.keras.metrics.Mean(name='oh_test_loss')
        self.oh_test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='oh_test_accuracy')

    def onehot_encode(self, image, label):
        return image, tf.one_hot(label, depth=10, dtype=tf.float32) # hardcord

    def init_datasets(self):
        batch_size = 256
        mnist = tf.keras.datasets.mnist

        (image_train, label_train), (image_test, label_test) = mnist.load_data()
        image_train, image_test = image_train / 255.0, image_test / 255.0
        image_train, image_test = image_train[:,:,:,tf.newaxis].astype("float32"), image_test[:,:,:,tf.newaxis].astype("float32")

        self.oh_train_ds = tf.data.Dataset.from_tensor_slices((image_train, label_train)).map(self.onehot_encode).shuffle(10000).batch(batch_size)
        self.oh_test_ds = tf.data.Dataset.from_tensor_slices((image_test, label_test)).map(self.onehot_encode).batch(32)

    def my_cnn(self, inputs, output_dim):
        x = tf.keras.layers.Conv2D(32, kernel_size=3)(inputs)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dense(output_dim)(x)

        return x

    @tf.function()
    def oh_train_step(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self.oh_model(images, training=True)
            loss = self.oh_loss(labels, preds)
        gradients = tape.gradient(loss, self.oh_model.trainable_variables)
        self.oh_optimizer.apply_gradients(zip(gradients, self.oh_model.trainable_variables))

        self.oh_train_loss(loss)
        self.oh_train_accuracy(labels, preds)

    @tf.function()
    def oh_test_step(self, images, labels):
        preds = self.oh_model(images, training=False)
        loss = self.oh_loss(preds, labels)

        self.oh_test_loss(loss)
        self.oh_test_accuracy(labels, preds)

    def train(self):
        for epoch in range(self.EPOCHS):
            # Reset the metrics at the start of the next epoch
            self.oh_train_loss.reset_states()
            self.oh_train_accuracy.reset_states()
            self.oh_test_loss.reset_states()
            self.oh_test_accuracy.reset_states()
        
            for images, labels in self.oh_train_ds:
                self.oh_train_step(images, labels)
        
            for test_images, test_labels in self.oh_test_ds:
                self.oh_test_step(test_images, test_labels)
        
            print(
                f'Epoch {epoch + 1}, '
                f'\tonehot Loss: {self.oh_train_loss.result()}, '
                f'onehot Accuracy: {self.oh_train_accuracy.result() * 100}, '
                f'onehot Test Loss: {self.oh_test_loss.result()}, '
                f'onehot Test Accuracy: {self.oh_test_accuracy.result() * 100}'
            )