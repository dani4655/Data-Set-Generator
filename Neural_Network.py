from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

import data_set_generator


def split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=None):
    # First, split the dataset into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(
        dataset, labels, test_size=test_size, random_state=random_state)

    # Then, split the training set into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=validation_size / (1 - test_size), random_state=random_state)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels


# Generate dataset
num_images = 10000
image_size = 128
dataset, labels = data_set_generator.generate_dataset(num_images, image_size)

# Split the dataset into training, validation, and test sets
(train_images, val_images, test_images, train_labels, val_labels, test_labels) =\
    split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=42)

# Preprocess the datasets
train_images = train_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
val_images = val_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
