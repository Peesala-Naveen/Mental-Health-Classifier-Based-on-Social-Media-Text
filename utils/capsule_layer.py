from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(1, self.input_dim, self.num_capsule * self.dim_capsule),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = K.shape(inputs)[0]

        u_hat = K.dot(inputs, self.W[0])

        u_hat = K.reshape(
            u_hat,
            (batch_size, -1, self.num_capsule, self.dim_capsule)
        )

        u_hat = K.mean(u_hat, axis=1)

        return u_hat

    # ✅ ADD HERE (INSIDE CLASS, LAST METHOD)
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "routings": self.routings
        })
        return config