import os
import sys
import pandas as pd 
import pingouin as pg 


parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')
text_dataset_dir = os.path.join(parent_dir, 'dataset/texts')
results_dir = os.path.join(parent_dir, 'results')

sys.path.insert(0, src_dir)
import read


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

        if df.empty:
            continue
        
        for emotion in emotion_dict.values():

            gold_standard = df[f'{emotion}_x']
            predicted = df[f'{emotion}_y'].apply(lambda x: x[0])
        
            pearson_df = pg.corr(gold_standard, predicted)
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

    df.set_index(['Emotion', 'Type'], inplace=True)
    df = df.drop(columns=['Bayes Factor','Statistical Power'])
    return df

def read_pickle(file_name):
    return pd.read_pickle(os.path.join(results_dir, f'cosine_similarity/avg_similarity/{file_name}'))

def write_data(df, file_name):
    output_dir = os.path.abspath(os.path.join(results_dir, 'pearson'))
    df.to_csv(os.path.join(output_dir, file_name))

def main():
    

    similarity_dfs = ['avg_sabbatino_multimodal_similarity_clip.pkl', 'avg_textual_similarity_clip.pkl', 'avg_textual_similarity_ft.pkl']
    output_files = ['pearson_sabbatino_multimodal.csv', 'pearson_textual_clip.csv', 'pearson_textual_ft.csv']
    sabbatino_df = read.sabbatino_et_al()
    merged_dict = {}
    for similarity_df, output_file in zip(similarity_dfs, output_files):
        
        df = read_pickle(similarity_df)
        # x gold-standard y predicted
        merged_df = sabbatino_df.merge(df, on='Word').drop(columns=['IDs', 'ARPA Pron'])

        merged_dict['real'] = merged_df[merged_df['Real Word Flag'] == 1]
        merged_dict['pseudowords'] = merged_df[merged_df['Real Word Flag'] == 0]

        correlation_df = get_pearson(merged_dict)         
        correlation_df = format_dataframe(correlation_df)
        write_data(correlation_df, output_file)


if __name__ == "__main__":

   main()

