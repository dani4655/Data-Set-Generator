import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

    # Start with a random point
    start_point = (np.random.randint(10, image_size - 10), np.random.randint(10, image_size - 10))

    # Randomly determine the number of line segments
    num_segments = np.random.randint(3, 8)

    for _ in range(num_segments):
        # Randomly determine the length and angle of each segment
        length = np.random.randint(5, 20)
        angle = np.random.uniform(0, 2 * np.pi)

        # Calculate the endpoint of the segment
        end_point = (
            int(start_point[0] + length * np.cos(angle)),
            int(start_point[1] + length * np.sin(angle))
        )

        # Draw the line segment
        cv2.line(img, start_point, end_point, 255, 3)

        # Update the start point for the next segment
        start_point = end_point

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

# Example usage:
num_images = 1000
image_size = 64
dataset, labels = generate_dataset(num_images, image_size)

# Visualize a few examples
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(dataset[i], cmap='gray')
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis('off')
    plt.savefig(f'image_{i}.png')
plt.show()
