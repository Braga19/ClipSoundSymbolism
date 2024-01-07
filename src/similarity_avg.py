import pandas as pd 
import os 

parent_dir = os.getcwd()
text_dataset_dir = os.path.join(parent_dir, 'dataset/texts')
image_dataset_dir = os.path.join(parent_dir, 'dataset/images')
results_dir = os.path.join(parent_dir, 'results')


def format_datasets(mean_df, sd_df):


    mean_without_words= mean_df.iloc[:, 1:].round(3)
    sd_without_words = sd_df.iloc[:, 1:].round(3)
    
    merged_df = pd.DataFrame()

    for column in mean_without_words.columns:
        merged_df[column] = mean_without_words[column].combine(sd_without_words[column], lambda s1, s2: (s1, s2))

    
    merged_df.insert(0, 'Word', mean_df.iloc[:, 0])

    return merged_df

def get_avg_std_multimodal_similarity(df):

    '''
    Get the average and standard deviation of the cosine similarity for each words and pseudowords and return a pandas df groupby emotions
    '''

    means = []
    stds = []
    for i in range(1, df.shape[1], 500):
        
        means.append(df.iloc[:, i:i+500].mean(axis=1))
        stds.append(df.iloc[:, i:i+500].std(axis=1))
    
    emotion_dict = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Joy', 4: 'Sadness', 5: 'Surprise'}

    similarity_mean = {}
    similarity_std = {}

    for n, mean_list, std_list in zip(range(len(means)), means, stds):

        similarity_mean[emotion_dict[n]] = {}
        similarity_std[emotion_dict[n]] = {}

        for i, mean, std in zip(range(len(mean_list)), mean_list, std_list):

            similarity_mean[emotion_dict[n]][df.iloc[i,0]] =  mean
            similarity_std[emotion_dict[n]][df.iloc[i,0]] =  std
    
    similarity_mean_df = pd.DataFrame.from_dict(similarity_mean)
    similarity_std_df = pd.DataFrame.from_dict(similarity_std)

    similarity_mean_df = similarity_mean_df.reset_index().rename(columns={'index': 'Word'})
    similarity_std_df = similarity_std_df.reset_index().rename(columns={'index': 'Word'})

    return format_datasets(similarity_mean_df, similarity_std_df)

def get_avg_std_textual_similarity(df):
    '''
    Get average and standard deviation of the cosine similarity across emotions
    '''

    df_without_words = df.iloc[:, 1:]  
    df_without_words.columns = pd.MultiIndex.from_tuples([col.split('_') for col in df_without_words.columns])

    mean_df = df_without_words.stack().mean(axis=1).unstack()
    sd_df = df_without_words.stack().std(axis=1).unstack()

    
    mean_df.insert(0, 'Word', df.iloc[:, 0])
    sd_df.insert(0, 'Word', df.iloc[:, 0])
    
    return format_datasets(mean_df, sd_df)

def read_data(file_name):
    return pd.read_csv(os.path.join(results_dir, f'cosine_similarity/{file_name}'))

def get_avg_similarity(df, similarity_type):
    """Choose the right function depending on the similarity"""
    return get_avg_std_multimodal_similarity(df) if similarity_type == 'multimodal' else get_avg_std_textual_similarity(df)

def write_pickle(df, file_name):
    output_dir = os.path.abspath(os.path.join(results_dir, 'cosine_similarity/avg_similarity'))
    df.to_pickle(os.path.join(output_dir, file_name))

def main():

    files = ['sabbatino_multimodal_similarity_clip.csv', 'emotional_words_multimodal_similarity_clip.csv', 'similarity_text_clip.csv', 'similarity_text_ft.csv']
    types = ['multimodal','multimodal', 'textual', 'textual']
    output_files = ['avg_sabbatino_multimodal_similarity_clip.pkl','avg_emotional_words_multimodal_similarity_clip.pkl', 'avg_textual_similarity_clip.pkl', 'avg_textual_similarity_ft.pkl']

    for file, typ, output in zip(files, types, output_files):
        data = read_data(file)
        avg = get_avg_similarity(data, typ)
        write_pickle(avg, output)

if __name__ == "__main__":

   main()

