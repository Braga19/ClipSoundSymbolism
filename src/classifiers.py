import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm_notebook
import torch
from torch.nn import Softmax
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

parent_dir = os.path.dirname(os.getcwd())
dataset_dir = os.path.join(parent_dir, 'dataset')
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)

import read

df_characters_annotated =  read.characters_df()

gender_df = read.get_subset('gender')
age_df = read.get_subset('age')

classifiers_gender = {
    'rizvan': {
        'extractor':  AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification"),
        'model': AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
    },

    'rizvan2': {
        'extractor': AutoFeatureExtractor.from_pretrained("rizvandwiki/gender-classification-2"),
        'model': AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification-2")
    },

    'leilab': {
        'extractor': AutoFeatureExtractor.from_pretrained("Leilab/gender_class"),
        'model': AutoModelForImageClassification.from_pretrained("Leilab/gender_class")
    }
}

classifiers_age = {
    'nateraw': {
        'extractor': AutoFeatureExtractor.from_pretrained("nateraw/vit-age-classifier"),
        'model': AutoModelForImageClassification.from_pretrained("nateraw/vit-age-classifier")
    },
    'ibombVit': {
        'extractor': AutoFeatureExtractor.from_pretrained("ibombonato/vit-age-classifier"),
        'model': AutoModelForImageClassification.from_pretrained("ibombonato/vit-age-classifier")

    },
    'ibombSwin': {
        'extractor': AutoFeatureExtractor.from_pretrained("ibombonato/swin-age-classifier"),
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
    for classifier_name, classifier in tqdm_notebook(classifiers_gender.items(), desc='Classifier'):

        model = classifier['model']
        extractor = classifier['extractor']

        for name in tqdm_notebook(gender_df['name'].str.lower(), desc='Names'):

            for i in range(20):

                path = os.path.join(root_dir, name, f'{name}_{i}.jpeg')
                img = load_image(path)
                if img is None:
                    continue
                prob_scores = compute_probabilities(model, extractor, img)
                if classifier_name == 'leilab':
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

    return df_results.to_csv(f'gender_predictions_{img_dataset}.csv', index=False)

def predict_age(img_dataset):

    root_dir = os.path.join(dataset_dir, img_dataset)

    results = []

    for classifier_name, classifier in tqdm_notebook(classifiers_age.items(), desc='Classifier'):

        model = classifier['model']
        extractor = classifier['extractor']

        for name in tqdm_notebook(age_df['name'].str.lower(), desc='Names'):
            
            for i in range(20):
                path = os.path.join(root_dir, name, f'{name}_{i}.jpeg')
                img = load_image(path)
                if img is None:
                    continue
                prob_scores = compute_probabilities(model, extractor, img)
                if classifier_name == 'nateraw':
                    young_prob, old_prob = prob_scores[0:4].sum().item(), prob_scores[6:].sum().item()
                else:
                    young_prob, old_prob = prob_scores[0:3].sum().item(), prob_scores[5:].sum().item()
                result = {
                    'classifier': classifier_name,
                    'name': name,
                    'image_idx': i,
                    'young_prob': young_prob,
                    'old_prob': old_prob
                }
                results.append(result)

    df_results = pd.DataFrame.from_dict(results)

    return df_results.to_csv(f'age_predictions_{img_dataset}.csv', index=False)
