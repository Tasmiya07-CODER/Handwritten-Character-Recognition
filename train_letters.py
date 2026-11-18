

import os
import tensorflow as tf
import tensorflow_datasets as tfds

print("Loading EMNIST Letters dataset...")

# Load training and testing datasets
train_data = tfds.load(
    "emnist/letters",
    split="train",
    as_supervised=True
)

test_data = tfds.load(
    "emnist/letters",
    split="test",
    as_supervised=True
)

# Correct image orientation and prepare labels
def preprocess(image, label):
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int64) - 1
    return image, label

AUTOTUNE = tf.data.AUTOTUNE

train_data = (
    train_data
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(10000)
    .batch(128)
    .prefetch(AUTOTUNE)
)

test_data = (
    test_data
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(AUTOTUNE)
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(26, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train model
print("Training alphabet recognition model...")

model.fit(
    train_data,
    epochs=5,
    validation_data=test_data,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(test_data, verbose=0)

print(f"Test loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Save model separately from digit model
os.makedirs("models", exist_ok=True)

model.save("models/emnist_letters.keras")

print("Alphabet model saved successfully!")
