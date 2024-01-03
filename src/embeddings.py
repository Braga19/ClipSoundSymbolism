import os
import numpy as np

from tqdm import tqdm_notebook

import torch
from torch.utils.data import Dataset
import clip
from PIL import Image

parent_dir = os.getcwd()
affectnet_dir = os.path.abspath(os.path.join(parent_dir, "dataset/images/AffectNet_face"))
EMOTIONS = ["Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"]
#EMOTION_DICT = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Sadness', 5: 'Surprise'}

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


def create_text_embeddings_clip(my_dict, model, device):
    '''
    This function creates text embeddings given a dictionnary of datasets and a given model and device.
    
    Args:
        my_dict (dictionary): A dictionary containing two dataset divided by real and pseudowords
        with words in the 'Word' column.
        model: The model (CLIP) used to encode the text inputs into feature vectors.
        device: The device on which the computations will be performed.
        
    Returns:
        dict: A dictionary with nested dictionnary mapping each word to its corresponding feature vector normalized.
    '''
    
    text_embeddings = {}

    for word_type, df in my_dict.items():

        words = df.Word.tolist()

        # Tokenize queries
        text_inputs = clip.tokenize(words).to(device)

        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        if word_type == 'real':
            label = df.Label.tolist()
            text_embedding_type = {f'{words[n]}_{label[n]}': embedding for n, embedding in enumerate(text_features)}
            text_embeddings[word_type] = text_embedding_type
        
        else:
            text_embedding_type = {words[n]: embedding for n, embedding in enumerate(text_features)}
            text_embeddings[word_type] = text_embedding_type

    
    return text_embeddings

def create_image_embeddings_clip(model, preprocess, device):
    '''
    This function creates image embeddings for a dataset of images using a given model.
    
    Args:
        img_dataset (AffectNet): An instance of the AffectNet dataset class. 
        Each element in the dataset is a tuple containing an image and its corresponding label.
        model: The model (CLIP) used to encode the preprocessed images into feature vectors.
        preprocess: The preprocessing function applied to the images before encoding.
        device: The device on which the computations will be performed.
        
    Returns:
        list: A list of dictionaries, each containing the image path, label, and feature vector for an image.
    '''
    
    images, class_id = preprocess_images(affectnet_dir)
    img_dataset = AffectNet(images, class_id)

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

def normalize_vector(vector):
    
    norm = np.linalg.norm(vector)
    if norm == 0: 
       return vector
    return vector / norm

def create_text_embeddings_ft(my_dict, model):
    """
    This function creates text embeddings given a dictionnary and a model

    Args:
        my_dict (dict): a dictionnary with two dataset divided by word type (real or pseudo)
        model: a fastText model to create feature vectors

    Returns:
        text_embeddings (dict): a dictionnary with nested dictionnary mapping each word to its corresponding feature vector normalized..
    """
    

    text_embeddings = {}

    for word_type, df in my_dict.items():

        word_list = df.Word.tolist()

        if word_type == 'real':
            label = df.Label.tolist()
            embedding = {f'{word_list[n]}_{label[n]}': normalize_vector(model.get_word_vector(word)) for n, word in enumerate(word_list)}
            text_embeddings[word_type] = embedding

        else:
            embedding = {word_list[n]: normalize_vector(model.get_word_vector(word)) for n, word in enumerate(word_list)}
            text_embeddings[word_type] = embedding


    return text_embeddings