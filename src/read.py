import pandas as pd 
import os 

parent_dir = os.path.dirname(os.getcwd())
text_dataset_dir = os.path.join(parent_dir, 'dataset/texts')
image_dataset_dir = os.path.join(parent_dir, 'dataset/images')
results_dir = os.path.join(parent_dir, 'results')
classifiers_dir = os.path.join(results_dir, 'classification')

def characters_df():

    df = pd.read_csv(os.path.join(text_dataset_dir, 'avgRatings_annotated.csv'))

    return df.copy()

def emotions_df():

    df = pd.read_csv(os.path.join(text_dataset_dir, 'nonsense-words-emotion-intensities.csv'), sep=';')

    return df.copy()

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
    


