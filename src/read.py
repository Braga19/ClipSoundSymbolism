import pandas as pd 
import os 

parent_dir = os.getcwd()
text_dataset_dir = os.path.join(parent_dir, 'dataset/texts')
image_dataset_dir = os.path.join(parent_dir, 'dataset/images')
results_dir = os.path.join(parent_dir, 'results')
classifiers_dir = os.path.join(results_dir, 'classification')

# Reading datasets 
def characters_df():

    df = pd.read_csv(os.path.join(text_dataset_dir, 'avgRatings_annotated.csv'))

    return df.copy()

def emotions_df():

    df = pd.read_csv(os.path.join(text_dataset_dir, 'nonsense-words-emotion-intensities.csv'), sep=';')

    return df.copy()

def emotional_words():

    df = pd.read_excel(os.path.join(text_dataset_dir, 'emotional_words.xlsx'))

    return df.copy()


# Modifying datasets for analysis
def get_subset(attribute):

    '''get subset of characters_df scored on specific attribute (gender or age)'''
    
    df = characters_df()
    df_attribute = df.loc[df['attribute'] == attribute, ['name', 'rating.mean', 'type']]

    return df_attribute

def get_merge_dataset(attribute, img_generator):

    '''
    Merging classifications dataset results with dataset characters annotated
    - attribute: gender or age (str)
    - img_generator: sd or vqgan (str)
    '''

    attribute_df = get_subset(attribute)
    attribute_df['name'] = attribute_df['name'].str.lower()
    
    file_path = os.path.join(classifiers_dir, f'{attribute}_predictions_{img_generator}_face.csv')
    classifier_df = pd.read_csv(file_path)

    return pd.merge(attribute_df, classifier_df, on='name')
 

def prepare_dataset_for_pearson(attribute, img_generator, prob_col1):

    '''Averaging results of merge dataset
    - attribute: gender or age (str)
    - img_generator: sd or vqgan (str)
    - prob_col1 (female_prob or old_prob)
    '''

    df = get_merge_dataset(attribute, img_generator)
    
    return df.groupby(['name', 'classifier'], as_index=False).agg({f'{prob_col1}': 'mean','type': 'first', 'rating.mean': 'first'})
    
def get_similarity_score_avg():

    '''
    Get the average of the cosine similarity for each words and pseudowords and return a pandas df groupby emotions
    '''
    df = pd.read_csv(os.path.join(results_dir, 'cosine_similarity/similarity_score.csv'))

    means = []
    for i in range(1, df.shape[1], 500):
        
        means.append(df.iloc[:, i:i+500].mean(axis=1))
    
    emotion_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Sadness', 5: 'Surprise'}

    similarity_mean = {}

    for n, mean_list in enumerate(means):

        similarity_mean[emotion_dict[n]] = {}

        for i, mean in enumerate(mean_list):

            similarity_mean[emotion_dict[n]][df.iloc[i,0]] =  mean
    
    similarity_mean_df = pd.DataFrame.from_dict(similarity_mean)

    similarity_mean_df = similarity_mean_df.reset_index().rename(columns={'index': 'Word'})

    return similarity_mean_df

def dataset_for_pearson_emotion():

    '''
    Merge the similarity_score_avg with the df_emotion for computing pearson correlation
    '''

    df_similarity_avg = get_similarity_score_avg()
    df_emotion = emotions_df()

    # x gold-standard y predicted
    merged_df =  df_emotion.merge(df_similarity_avg, on='Word').drop(columns=['IDs', 'ARPA Pron'])

    return merged_df

