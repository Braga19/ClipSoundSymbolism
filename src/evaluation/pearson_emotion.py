import os
import sys
import pandas as pd 
import pingouin as pg 

parent_dir = os.path.dirname(os.getcwd())
src_dir = os.path.join(parent_dir, 'src')

sys.path.insert(0, src_dir)
import read


emotion_score_merge_df = read.dataset_for_pearson_emotion()

emotion_types = {'real' : emotion_score_merge_df[emotion_score_merge_df['Real Word Flag'] == 1],
              'pseudowords' : emotion_score_merge_df[emotion_score_merge_df['Real Word Flag'] == 0]}

def get_pearson(my_dict):

    '''
    This function compute the pearson correlation between emotions score

    Args: 
        A dictionnary with dataset score on emotion divided by type

    Returns:
        A dataset with the pearson correlation analysis
    '''
    emotion_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Sadness', 5: 'Surprise'}

    pearson_all = pd.DataFrame()

    for typ, df in my_dict.items():
        for emotion in emotion_dict.values():

            pearson_df = pg.corr(df[f'{emotion}_x'], df[f'{emotion}_y'])
            pearson_df['emotion'] = emotion
            pearson_df['type'] = typ

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
        'emotion': 'Emotion',
        'type': 'Type'
    }, inplace=True)

    df.set_index(['Emotion','Type'], inplace=True)

    return df

def main():

    correlation_df = get_pearson(emotion_types)

    correlation_df = format_dataframe(correlation_df)

    return correlation_df


correlation_df = main()

correlation_df.to_csv('emotion_pearson.csv')

