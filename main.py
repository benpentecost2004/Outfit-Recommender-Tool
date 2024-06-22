import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
from scipy.spatial.distance import cosine

# Change directory path to match path on your device
directory_path = '/Users/benpentecost/Documents/CodingProjects/PythonProjects/ClothingRecommenationTool/women fashion'
files = os.listdir(directory_path)

# Function to load and display an image
def show_img(file_path):
    image = Image.open(file_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Change directory path to match path on your device
image_directory = '/Users/benpentecost/Documents/CodingProjects/PythonProjects/ClothingRecommenationTool/women fashion'
image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', '.webp'))]

# Display the first image to understand its characteristics
first_img_path = os.path.join(directory_path, files[-2])
show_img(first_img_path)


base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess(img_path)
    features = extract(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))


def recommender(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess(input_image_path)
    input_features = extract(model, preprocessed_img)

    # calculate similarities and find the top N similar images
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    indices = np.argsort(similarities)[-top_n:]

    # filter out the input image index from indices
    indices = [idx for idx in indices if idx != all_image_names.index(input_image_path)]

    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    for i, idx in enumerate(indices[:top_n], start=1):
        # Change directory path to match path on your device
        image_path = os.path.join('/Users/benpentecost/Documents/CodingProjects/PythonProjects/ClothingRecommenationTool/women fashion', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
