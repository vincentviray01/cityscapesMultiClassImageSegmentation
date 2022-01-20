import tensorflow as tf

class ConvPoolBlockWithoutTranspose(tf.keras.Model):
    def __init__(self, kernel_size, numFilters1, numFilters2, dropoutRate=0.7, pool_size=2, useMaxPool=True):
        super(ConvPoolBlockWithoutTranspose, self).__init__()

        self.conv2a = tf.keras.layers.Conv2D(numFilters1, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.leakyRelu = tf.keras.layers.LeakyReLU()
        # self.dropout = tf.keras.layers.Dropout(dropoutRate)
        self.conv2b = tf.keras.layers.Conv2D(numFilters2, (kernel_size, kernel_size), activation = 'relu', kernel_initializer='he_normal', padding='same')
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()
        self.useMaxPool = useMaxPool
        if useMaxPool == True:
            self.maxPool = tf.keras.layers.MaxPooling2D((pool_size, pool_size))


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.batchNorm(x)#, training=training)
        x = self.leakyRelu(x)
        # x = self.dropout(x)
        x = self.conv2b(x)
        x = self.batchNorm2(x)#, training=training)
        x = self.leakyRelu2(x)
        if self.useMaxPool:
            maxPoolX = self.maxPool(x)
            return x, maxPoolX

        return x

class ConvPoolBlockWithTranspose(tf.keras.Model):
    def __init__(self, kernel_size, transposeFilters, numFilters1, numFilters2, dropoutRate=0.7, transposeStrides=2):
        super(ConvPoolBlockWithTranspose, self).__init__()

        self.conv2Transpose = tf.keras.layers.Conv2DTranspose(transposeFilters, (kernel_size, kernel_size), strides = (transposeStrides, transposeStrides), padding='same')
        self.batchNorm = tf.keras.layers.BatchNormalization()
        self.leakyRelu = tf.keras.layers.LeakyReLU()
        self.concatenate = tf.keras.layers.Concatenate(axis=3)
        self.conv2a = tf.keras.layers.Conv2D(numFilters1, (kernel_size, kernel_size), kernel_initializer='he_normal', padding='same')
        self.batchNorm2 = tf.keras.layers.BatchNormalization()
        self.leakyRelu2 = tf.keras.layers.LeakyReLU()
        # self.dropout = tf.keras.layers.Dropout(dropoutRate)
        self.conv2b = tf.keras.layers.Conv2D(numFilters2, (kernel_size, kernel_size), activation = 'relu', kernel_initializer='he_normal', padding='same')
        self.batchNorm3 = tf.keras.layers.BatchNormalization()
        self.leakyRelu3 = tf.keras.layers.LeakyReLU()
        


    def call(self, input_tensor, leftHalfInput, training=False):
        x = self.conv2Transpose(input_tensor)
        x = self.batchNorm(x)
        x = self.leakyRelu(x)
        x = self.concatenate([x, leftHalfInput])
        x = self.conv2a(x)
        x = self.batchNorm2(x)#, training=training)
        x = self.leakyRelu2(x)
        # x = self.dropout(x)
        x = self.conv2b(x)
        x = self.batchNorm3(x)#, training=training)
        x = self.leakyRelu3(x)

        return x