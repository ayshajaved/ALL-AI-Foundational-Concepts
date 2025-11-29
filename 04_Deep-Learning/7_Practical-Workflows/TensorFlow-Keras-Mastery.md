# TensorFlow/Keras Mastery

> **Production-ready Deep Learning** - The Keras Functional API and Custom Loops

---

## 🔗 Functional API

More flexible than `Sequential`. Handles non-linear topologies (ResNets, Inception).

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2d(32, 3, activation="relu")(inputs)
x = layers.MaxPooling2d(2)(x)

# Residual connection
residual = x
x = layers.Conv2d(32, 3, activation="relu", padding="same")(x)
x = layers.add([x, residual])

outputs = layers.Dense(10)(layers.Flatten()(x))

model = keras.Model(inputs=inputs, outputs=outputs)
```

---

## 🛠️ Custom Layers

```python
class MyLayer(layers.Layer):
    def __init__(self, units=32):
        super(MyLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros",
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

---

## 🔄 Custom Training Loop

```python
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function  # Compiles to graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value
```

---

**TensorFlow: Scalable and deployable!**
