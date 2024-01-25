from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
import numpy as np
import pandas as pd
import os
import sys

parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')


sys.path.insert(0, src_dir)
import read

gender_classification = {'sd': read.get_merge_dataset('gender', 'sd'),
                         'vqgan': read.get_merge_dataset('gender', 'vqgan')}

age_classificaiton = {'sd': read.get_merge_dataset('age', 'sd'),
                      'vqgan': read.get_merge_dataset('age', 'vqgan')}


def add_prediction_column(df, prob_col1, prob_col2):
    '''given a dataset with probability scores, create a new column of the predicted class,
    where 0 is for prob_col1 and 1 is for prob_col2'''
    df['pred_class'] = np.where(df[prob_col1] > df[prob_col2], 0, 1)
    return df 

def process_classification_data_per_type(classification_data, prob_col1, prob_col2):
    '''Process classification data by adding prediction column and computing Fleiss' kappa per type'''
    fleiss_data = {}
    for img_generator, df in classification_data.items():
        # Add prediction column
        classification_data[img_generator] = add_prediction_column(df, prob_col1, prob_col2)
        # Compute Fleiss' kappa per type
        fleiss_data[img_generator] = df.pivot_table(index=['name', 'image_idx', 'type'], columns='classifier', values='pred_class')
    return fleiss_data

def compute_fleiss_kappa(df):
    '''Compute Fleiss' kappa for a given dataframe'''
    try:
        agg = aggregate_raters(df)
        fleiss_coeff = fleiss_kappa(agg[0], method='fleiss')
        return fleiss_coeff
    except Exception as e:
        print(f"Error computing Fleiss' kappa: {e}")
        raise e

def compute_and_save_fleiss_kappa_per_type(fleiss_data, filename):
    '''Compute Fleiss' kappa for each image generator per type and save to CSV'''
    fleiss_kappa_values = {}
    for img_generator, df in fleiss_data.items():
        fleiss_kappa_values[img_generator] = df.groupby('type').apply(compute_fleiss_kappa).to_dict()
    # Create DataFrame and save to CSV
    fleiss_df = pd.DataFrame(list(fleiss_kappa_values.items()), columns=['Image Generator', 'Fleiss Kappa'])
    
    # Convert the Fleiss Kappa column from a dictionary to separate columns
    fleiss_df = pd.json_normalize(fleiss_df['Fleiss Kappa'])
    fleiss_df['Image Generator'] = ['Stable Diffusion', 'VQGAN-CLIP']

    # Rearrange the columns
    fleiss_df = fleiss_df[['Image Generator', 'real', 'madeup', 'talking']]

    # Rename the columns and round decimals
    fleiss_df.columns = ['Image Generator', "Fleiss' Kappa (real)", "Fleiss' Kappa (madeup)", "Fleiss Kappa (talking)"]
    fleiss_df = round(fleiss_df, 2)
    fleiss_df.to_csv(filename, index=False)
    

def main():
    
    # Process gender classification data
    fleiss_gender = process_classification_data_per_type(gender_classification, 'female_prob', 'male_prob')

    # Process age classification data
    fleiss_age = process_classification_data_per_type(age_classificaiton, 'young_prob', 'old_prob')

    # Compute Fleiss' kappa for gender classification per type and save to CSV
    compute_and_save_fleiss_kappa_per_type(fleiss_gender, 'fleiss_gender.csv')

    # Compute Fleiss' kappa for age classification per type and save to CSV
    compute_and_save_fleiss_kappa_per_type(fleiss_age, 'fleiss_age.csv')

if __name__ == "__main__":

    main()






