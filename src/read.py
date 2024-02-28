import pandas as pd 
import os 

#directories
parent_dir = os.getcwd()
text_dataset_dir = os.path.join(parent_dir, 'dataset/texts')
image_dataset_dir = os.path.join(parent_dir, 'dataset/images')
results_dir = os.path.join(parent_dir, 'results')
classifiers_dir = os.path.join(results_dir, 'classification')

### Reading dataset 
def characters_df():

    df = pd.read_csv(os.path.join(text_dataset_dir, 'avgRatings_annotated.csv'))

    return df.copy()

### Modifying datasets for analysis
def get_subset(attribute):

    '''get subset of characters_df scored on specific attribute (gender or age)'''
    
    df = characters_df()
    df_attribute = df.loc[df['attribute'] == attribute, ['name', 'rating.mean', 'type']].drop_duplicates()
    df_attribute['name'] = df_attribute['name'].str.lower()
    df_attribute = df_attribute.reset_index(drop=True)
    return df_attribute

def get_merge_dataset(attribute, img_generator):

    '''
    Merging classifications dataset results with dataset characters annotated
    - attribute: gender or age (str)
    - img_generator: sd or vqgan (str)
    '''

    attribute_df = get_subset(attribute)
    
    file_path = os.path.join(classifiers_dir, f'{attribute}_predictions_{img_generator}_face.csv')
    classifier_df = pd.read_csv(file_path)

    return pd.merge(classifier_df, attribute_df, on='name', how='left')
 

def prepare_dataset_for_pearson(attribute, img_generator, prob_col1):

    '''Averaging results of merge dataset
    - attribute: gender or age (str)
    - img_generator: sd or vqgan (str)
    - prob_col1 (female_prob or old_prob)
    '''

    df = get_merge_dataset(attribute, img_generator)
    
    return df.groupby(['name', 'classifier'], as_index=False).agg({f'{prob_col1}': 'mean','type': 'first', 'rating.mean': 'first'})

