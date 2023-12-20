import os
import sys
import pandas as pd 

from tqdm import tqdm_notebook

import torch
from torch.utils.data import Dataset
import clip
from PIL import Image

# Define constants
PARENT_DIR = os.path.dirname(os.getcwd())
SRC_DIR = os.path.join(PARENT_DIR, 'src')
ROOT_DIR = os.path.join(PARENT_DIR, "dataset/images/AffectNet_face")
EMOTIONS = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
EMOTION_DICT = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Sadness', 5: 'Surprise'}


sys.path.insert(0, SRC_DIR)
import read

class AffectNet(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

def starts_with_vowel(word):
    """Check if a word starts with a vowel."""
    vowels = 'aeiou'
    return word[0].lower() in vowels

def preprocess_images(root_dir):
    """Preprocess images and return image paths and class ids."""
    images = []
    class_id = []
    for i, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(root_dir, emotion)
        for image_name in os.listdir(emotion_dir):
            image_path = os.path.join(emotion_dir, image_name)
            images.append(image_path)
            class_id.append(i)
    return images, class_id


def create_text_embeddings(df_emotion, model, device):
    '''
    This function creates text embeddings for a list of words using a given model.
    
    Args:
        df_emotion (DataFrame): A DataFrame containing the words in the 'Word' column.
        model: The model used to encode the text inputs into feature vectors.
        device: The device on which the computations will be performed.
        
    Returns:
        dict: A dictionary mapping each word to its corresponding feature vector normalized.
    '''
    words = df_emotion.Word.tolist()
    queries = [f'an {word} face' if starts_with_vowel(word) else f'a {word} face' for word in words]

    # Tokenize queries
    text_inputs = clip.tokenize(queries).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_embedding = {words[n]: embedding for n, embedding in enumerate(text_features)}
    
    return text_embedding

def create_image_embeddings(img_dataset, model, preprocess, device):
    '''
    This function creates image embeddings for a dataset of images using a given model.
    
    Args:
        img_dataset (AffectNet): An instance of the AffectNet dataset class. 
        Each element in the dataset is a tuple containing an image and its corresponding label.
        model: The model used to encode the preprocessed images into feature vectors.
        preprocess: The preprocessing function applied to the images before encoding.
        device: The device on which the computations will be performed.
        
    Returns:
        list: A list of dictionaries, each containing the image path, label, and feature vector for an image.
    '''
    image_embeddings = []
    for i in tqdm_notebook(range(len(img_dataset)), desc='image'):
        image, label = img_dataset[i]
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        result = {'image_name': os.path.basename(img_dataset.image_paths[i]), 'label': label, 'embedding': image_features}

        image_embeddings.append(result)
     
    return image_embeddings


def compute_similarity_score(image_embeddings, text_embeddings):

    similarity_scores = {}


    for image_dict in tqdm_notebook(image_embeddings, desc='image'):
        image_name = image_dict['image_name']
        image_embedding = image_dict['embedding']
        
        similarity_scores[image_name] = {}
        
        
        for text, text_embedding in text_embeddings.items():
            
            similarity = (image_embedding @ text_embedding.T)
            similarity_scores[image_name][text] = similarity.item()
    
    return pd.DataFrame(similarity_scores)


def main():
    """Main function."""
    # Preprocess images
    images, class_id = preprocess_images(ROOT_DIR)

    # Get emotion words and pseudowords
    emotion_df = read.emotions_df()

    # Initialize image dataset
    img_dataset = AffectNet(images, class_id)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    
    text_embeddings = create_text_embeddings(emotion_df, model, device)
    image_embeddings = create_image_embeddings(img_dataset, model, preprocess, device)

    similarity_score = compute_similarity_score(image_embeddings, text_embeddings)

    return similarity_score


similarity_score = main()

similarity_score.to_csv('similarity_score.csv')
