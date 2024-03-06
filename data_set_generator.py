import cv2
import numpy as np
import matplotlib
from keras.src.utils import split_dataset
from tensorflow.keras import layers, models

# Specify Matplotlib backend
matplotlib.use('TkAgg')  # Change to another backend if needed


def generate_bubble(image_size):
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    center = (np.random.randint(20, image_size - 20), np.random.randint(20, image_size - 20))
    axes = (np.random.randint(10, 20), np.random.randint(5, 15))
    cv2.ellipse(img, center, axes, 0, 0, 360, 255, -1)
    return img


def generate_scratch(image_size):
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Randomly choose start and end points for the line
    start_point = (np.random.randint(10, image_size - 20), np.random.randint(10, image_size - 20))
    end_point = (np.random.randint(10, image_size - 20), np.random.randint(10, image_size - 20))

    # Draw the line
    cv2.line(img, start_point, end_point, 255, 3)

    return img

def generate_dirt_stain(image_size):
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Ensure that the center, radius, and frequency are valid
    center = (np.random.randint(20, image_size - 20), np.random.randint(20, image_size - 20))
    num_ellipses = np.random.randint(2, 5)  # Randomly choose the number of ellipses

    for _ in range(num_ellipses):
        # Randomly determine the size and orientation of each ellipse
        major_axis = np.random.randint(5, 15)
        minor_axis = np.random.randint(3, 8)
        angle = np.random.uniform(0, 2 * np.pi)

        # Randomly determine the position of each ellipse
        offset_x = np.random.randint(-5, 5)
        offset_y = np.random.randint(-5, 5)

        # Calculate the ellipse parameters
        ellipse_center = (center[0] + offset_x, center[1] + offset_y)
        ellipse_axes = (major_axis, minor_axis)

        # Draw the ellipse
        cv2.ellipse(img, ellipse_center, ellipse_axes, angle, 0, 360, 255, -1)

    return img


def generate_dataset(num_images, image_size):
    dataset = []
    labels = []

    for _ in range(num_images):
        defect_type = np.random.choice(['bubble', 'scratch', 'dirt_stain'])
        if defect_type == 'bubble':
            img = generate_bubble(image_size)
            label = 0  # 0 represents bubble
        elif defect_type == 'scratch':
            img = generate_scratch(image_size)
            label = 1  # 1 represents scratch
        else:
            img = generate_dirt_stain(image_size)
            label = 2  # 2 represents dirt stain

        dataset.append(img)
        labels.append(label)

    return np.array(dataset), np.array(labels)


# from sklearn.model_selection import train_test_split
#
#
# def split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=None):
#     # First, split the dataset into training and test sets
#     train_images, test_images, train_labels, test_labels = train_test_split(
#         dataset, labels, test_size=test_size, random_state=random_state)
#
#     # Then, split the training set into training and validation sets
#     train_images, val_images, train_labels, val_labels = train_test_split(
#         train_images, train_labels, test_size=validation_size / (1 - test_size), random_state=random_state)
#
#     return train_images, val_images, test_images, train_labels, val_labels, test_labels
#
#
# # Generate dataset
# num_images = 10000
# image_size = 128
# dataset, labels = generate_dataset(num_images, image_size)
#
# # Split the dataset into training, validation, and test sets
# (train_images, val_images, test_images, train_labels, val_labels, test_labels) =\
#     split_dataset(dataset, labels, test_size=0.2, validation_size=0.2, random_state=42)
#
# # Preprocess the datasets
# train_images = train_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
# val_images = val_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
# test_images = test_images.reshape(-1, image_size, image_size, 1).astype('float32') / 255.0
#
# # Define a simple CNN model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(3, activation='softmax')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
#
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(test_images, test_labels)
# print("Test Loss:", test_loss)
# print("Test Accuracy:", test_accuracy)
