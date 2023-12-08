import os
import sys
import pandas as pd 
import pingouin as pg 

parent_dir = os.path.dirname(os.getcwd())
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)
import read

def prepare_dataframes():
    gender_df = {
        'sd': read.prepare_dataset_for_pearson('gender', 'sd', 'female_prob'),
        'vqgan': read.prepare_dataset_for_pearson('gender', 'vqgan', 'female_prob')
    }

    age_df = {
        'sd': read.prepare_dataset_for_pearson('age', 'sd', 'old_prob'),
        'vqgan': read.prepare_dataset_for_pearson('age', 'vqgan', 'old_prob')
    }

    return gender_df, age_df

def compute_pearson(df_dict, column):
    pearson_all = pd.DataFrame()
    for img_generator, df in df_dict.items():
        grouped = df.groupby(['classifier', 'type'])

        # Compute Pearson correlation for each group
        for name, group in grouped:
            pearson_df = pg.corr(group['rating.mean'], group[column])
            pearson_df['image generator'] = img_generator
            pearson_df['classifier'] = name[0]
            pearson_df['type'] = name[1]
            pearson_all = pd.concat([pearson_all, pearson_df], ignore_index=True)

    return pearson_all

def format_dataframe(df):
    df.rename(columns={
        'n': 'Sample Size (n)',
        'r': 'Pearson Correlation (r)',
        'CI95%': '95% Confidence Interval',
        'p-val': 'P-value',
        'BF10': 'Bayes Factor',
        'power': 'Statistical Power',
        'image generator': 'Image Generator',
        'classifier': 'Classifier',
        'type': 'Type'
    }, inplace=True)

    df.set_index(['Image Generator', 'Classifier', 'Type'], inplace=True)

    return df

def write_to_csv(df, filename):
    df.to_csv(filename)


gender_df, age_df = prepare_dataframes()

gender_pearson = compute_pearson(gender_df, 'female_prob')
gender_pearson = format_dataframe(gender_pearson)
write_to_csv(gender_pearson, 'gender_pearson.csv')

age_pearson = compute_pearson(age_df, 'old_prob')
age_pearson = format_dataframe(age_pearson)
write_to_csv(age_pearson, 'age_pearson.csv')


