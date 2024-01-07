import os
import sys
import pandas as pd 
import numpy as np
import pathlib

import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.io as pio

parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')
results_dir = os.path.join(parent_dir, 'results')

sys.path.insert(0, src_dir)
import read





def correlation_scatterplot_emotion(merged_dict, model, modality):

    emotion_colors = {'Anger':'#d53e4f','Disgust':'#fc8d59','Fear':'#fee08b', 'Joy':'#e6f598','Sadness':'#99d594', 'Surprise':'#3288bd'}


    for emotion, color in emotion_colors.items():

        for typ, df in merged_dict.items():

            if df.empty:
                continue
            x = df[f'{emotion}_x']
            y = df[f'{emotion}_y'].apply(lambda x: x[0])
            word = df.Word

            xlimits = [0, 1]
            ylimits = [-1, 1]
            
            fig = go.Figure()
                        
            # Add a scatter trace with x, y, and name as text
            fig.add_trace(go.Scatter(name='Scatter', x=x, y=y, mode="markers", text=word, customdata=word, marker_color=color,
                                    hovertemplate='Word: <b>%{customdata}</b><extra></extra><br>Predicted Similarity: <b>%{y}</b><br>People Score: <b>%{x}</b>'))
            
            # Calculate the coefficients of the best fit line
            coefficients = np.polyfit(x, y, 1)
            # Generate y values for the best fit line
            y_fit = np.polyval(coefficients, x)
            # Add the best fit line to the plot
            fig.add_trace(go.Scatter(x=x, y=y_fit, mode="lines", name="Best fit", line_color=color))

            # Update the x and y axis limits and labels with some padding
            #padding = 0.1  # adjust this value to change the amount of padding
            #x_range = [xlimits[0] - (xlimits[1] - xlimits[0]) * padding, xlimits[1] + (xlimits[1] - xlimits[0]) * padding]
            #y_range = [ylimits[0] - (ylimits[1] - ylimits[0]) * padding, ylimits[1] + (ylimits[1] - ylimits[0]) * padding]
            
            # Update the x and y axis limits and labels
            fig.update_xaxes(range=xlimits, title="<b>people's score</b>", tickvals=[0, 1])
            fig.update_yaxes(range=ylimits, tickvals=[-1, 1])
        
            fig.update_yaxes(title= "<b>predicted similarity</b>",title_standoff=0)
            fig.update_layout(title_text=f"<b>Correlation between people's perception and {model} predictions about {emotion} degree of words</b>")

            filepath = os.getcwd()

            p = pathlib.Path(os.path.join(filepath, f'plots/emotions/{emotion}'))            
            p.mkdir(parents=True, exist_ok=True)
            pio.write_image(fig, os.path.join(filepath, f'plots/emotions/{emotion}/{modality}_{model}_{typ}.jpeg'), width=1200, height=600)
            pio.write_html(fig, os.path.join(filepath, f'plots/emotions/{emotion}/{modality}_{model}_{typ}.html'))

def read_pickle(file_name):
    return pd.read_pickle(os.path.join(results_dir, f'cosine_similarity/avg_similarity/{file_name}'))

def main():
    

    similarity_dfs = ['avg_sabbatino_multimodal_similarity_clip.pkl', 'avg_textual_similarity_clip.pkl', 'avg_textual_similarity_ft.pkl']
    models = ['Clip', 'Clip', 'FastText']
    modalities  = ['multimodal', 'textual', 'textual']
    sabbatino_df = read.sabbatino_et_al()
    merged_dict = {}
    for similarity_df, model, modality in zip(similarity_dfs, models, modalities):
        
        df = read_pickle(similarity_df)
        # x gold-standard y predicted
        merged_df = sabbatino_df.merge(df, on='Word').drop(columns=['IDs', 'ARPA Pron'])
        merged_dict['real'] = merged_df[merged_df['Real Word Flag'] == 1]
        merged_dict['pseudowords'] = merged_df[merged_df['Real Word Flag'] == 0]

        correlation_scatterplot_emotion(merged_dict, model, modality)


if __name__ == "__main__":

   main()