import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import torch
from torch.nn import Softmax
from transformers import AutoImageProcessor, AutoModelForImageClassification

parent_dir = os.getcwd()
dataset_dir = os.path.abspath(os.path.join(parent_dir, 'dataset/images'))
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

import read

df_characters_annotated =  read.characters_df()

gender_df = read.get_subset('gender')
age_df = read.get_subset('age')

classifiers_gender = { 
    'crangana': {
        'extractor': AutoImageProcessor.from_pretrained("crangana/trained-gender"),
        'model': AutoModelForImageClassification.from_pretrained("crangana/trained-gender")
    }
}

classifiers_age = {
     'cranage': {
        'extractor': AutoImageProcessor.from_pretrained("crangana/trained-age"),
        'model': AutoModelForImageClassification.from_pretrained("crangana/trained-age"),
     },

    'nateraw': {
        'extractor': AutoImageProcessor.from_pretrained("nateraw/vit-age-classifier"),
        'model': AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
    },
    'ibombSwin': {
        'extractor': AutoImageProcessor.from_pretrained("ibombonato/swin-age-classifier"),
        'model': AutoModelForImageClassification.from_pretrained("ibombonato/swin-age-classifier")
    }
}

def load_image(path):
    try:
        return Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Error loading image {path}: {e}")

def compute_probabilities(model, extractor, img):
    inputs = extractor(img, return_tensors = 'pt')
    softmax = Softmax(dim=1)
    with torch.no_grad():
        logits = model(**inputs).logits
        prob_scores = softmax(logits)[0]
    return prob_scores

def predict_gender(img_dataset):

    root_dir = os.path.join(dataset_dir, img_dataset)

    results = []
    for classifier_name, classifier in tqdm(classifiers_gender.items(), desc='Classifier'):

        model = classifier['model']
        extractor = classifier['extractor']

        for name in tqdm(gender_df['name'].str.lower(), desc='Names'):

            for i in range(20):

                path = os.path.join(root_dir, name, f'{name}_{i}.jpeg')
                img = load_image(path)
                if img is None:
                    raise ValueError("img is None")
                prob_scores = compute_probabilities(model, extractor, img)
                if classifier_name == 'leilab' or classifier_name == 'crangana':
                    female_prob, male_prob = prob_scores[1].item(), prob_scores[0].item()
                else:
                    female_prob, male_prob = prob_scores[0].item(), prob_scores[1].item()
                result = {
                    'classifier': classifier_name,
                    'name': name,
                    'image_idx': i,
                    'female_prob': female_prob,
                    'male_prob': male_prob            
                }
                results.append(result)

    df_results = pd.DataFrame.from_dict(results)

    return df_results.to_csv(f'gender_predictions_{img_dataset}_2.csv', index=False)

def predict_age(img_dataset):

    root_dir = os.path.join(dataset_dir, img_dataset)

    results = []

    for classifier_name, classifier in tqdm(classifiers_age.items(), desc='Classifier'):

        model = classifier['model']
        extractor = classifier['extractor']

        for name in tqdm(age_df['name'].str.lower(), desc='Names'):
            
            for i in range(20):
                path = os.path.join(root_dir, name, f'{name}_{i}.jpeg')
                img = load_image(path)
                if img is None:
                    raise ValueError("img is None")
                prob_scores = compute_probabilities(model, extractor, img)
                if classifier_name == 'nateraw' or classifier_name == 'cranage':
                    young_prob, old_prob = prob_scores[0:4].sum().item(), prob_scores[4:].sum().item()
                else:
                    young_prob, old_prob = prob_scores[0:3].sum().item(), prob_scores[3:].sum().item()
                result = {
                    'classifier': classifier_name,
                    'name': name,
                    'image_idx': i,
                    'young_prob': young_prob,
                    'old_prob': old_prob
                }
                results.append(result)

    df_results = pd.DataFrame.from_dict(results)

    return df_results.to_csv(f'age_predictions_{img_dataset}_2.csv', index=False)


def main():
    img_generator = ['SD_face', 'vqgan_face']
    for generator in img_generator:
        
        predict_age(generator)
        


if __name__ == "__main__":

    main()
