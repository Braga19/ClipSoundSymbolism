import os
import sys
import pandas as pd
import fasttext

import torch
from torch.utils.data import Dataset
import clip
from PIL import Image

parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')
model_dir = os.path.join(parent_dir, 'models')
result_dir = os.path.join(parent_dir, 'results')

sys.path.insert(0, src_dir)
import read
import embeddings


real_words_df = read.emotional_words()
pseudowords_df = read.emotions_df()
pseudowords_df = pseudowords_df[pseudowords_df['Real Word Flag'] == 0].reset_index(drop=True)
words_dict = {'real': real_words_df,
              'pseudowords': pseudowords_df}

def compute_text_to_text_similarity(embeddings_dict):

    '''
    This function compute the similarity score between real and pseudowords already embedded with a given model

    Args:
        embedding_dict (dict): a dictionary with embeddings for real and pseudowords
    
    Output:
        similarity_score (dataset): a dataset with similarity score 
    '''
    pseudowords_embeddings = embeddings_dict['pseudowords']
    real_embeddings = embeddings_dict['real']
    
    similarity_scores = {}

    for real_word, real_embed in real_embeddings.items():
        
        similarity_scores[real_word] = {}
        
        for pseudoword, pseudo_embed in pseudowords_embeddings.items():
            
            similarity = (real_embed @ pseudo_embed.T)
            similarity_scores[real_word][pseudoword] = similarity.item()
    
    return pd.DataFrame(similarity_scores)


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    ft_model = fasttext.load_model(os.path.join(model_dir,'coca_clean_d300_w5_m2_M5_rho0.7953.bin'))


    text_embeddings_ft = embeddings.create_text_embeddings_ft(words_dict, ft_model)
    text_embeddings_clip = embeddings.create_text_embeddings_clip(words_dict, clip_model, device)
    
    similarity_ft = compute_text_to_text_similarity(text_embeddings_ft)
    similarity_clip = compute_text_to_text_similarity(text_embeddings_clip)

    return similarity_ft, similarity_clip

if __name__ == "__main__":

    similarity = main()
    output_dir = os.path.join(result_dir, 'cosine_similarity')
    similarity[0].to_csv(os.path.join(output_dir,'similarity_text_ft.csv'))
    similarity[1].to_csv(os.path.join(output_dir,'similarity_text_clip.csv'))