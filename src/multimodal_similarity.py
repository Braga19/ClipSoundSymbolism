import os
import sys
import pandas as pd

from tqdm import tqdm_notebook

import torch
import clip


parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')
model_dir = os.path.join(parent_dir, 'models')
result_dir = os.path.join(parent_dir, 'results')

sys.path.insert(0, src_dir)
import read
import embeddings

def compute_text_to_image_similarity(image_embeddings, text_embeddings):

    similarity_scores = {}

    for image_dict in tqdm_notebook(image_embeddings, desc='image'):
        image_name = image_dict['image_name']
        image_embedding = image_dict['embedding']
        
        similarity_scores[image_name] = {}        
        
        for query, embedding in text_embeddings.items():
            
            similarity = (image_embedding @ embedding.T)
            similarity_scores[image_name][query] = similarity.item()

    
    return pd.DataFrame(similarity_scores)


def main():
    
    df_sabbatino = read.sabbatino_et_al()
    emotional_words_df = read.emotional_words()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    text_embeddings_clip_sabbatino = embeddings.create_text_embeddings_sabbatino(df_sabbatino, clip_model, device)
    text_embeddings_clip_emotional_words = embeddings.create_text_embeddings_sabbatino(emotional_words_df, clip_model, device)
    image_embeddings = embeddings.create_image_embeddings_clip(clip_model, preprocess, device)

    similarity_score_clip_sabbatino = compute_text_to_image_similarity(image_embeddings, text_embeddings_clip_sabbatino)
    similarity_score_clip_emotional_words = compute_text_to_image_similarity(image_embeddings, text_embeddings_clip_emotional_words)

    return similarity_score_clip_sabbatino, similarity_score_clip_emotional_words


if __name__ == "__main__":

    similarity = main()
    output_dir = os.path.join(result_dir, 'cosine_similarity')
    similarity[0].to_csv(os.path.join(output_dir,'sabbatino_multimodal_similarity_clip.csv'))
    similarity[1].to_csv(os.path.join(output_dir,'emotional_words_multimodal_similarity_clip.csv'))
