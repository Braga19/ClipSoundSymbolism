import os
import sys

parent_dir = os.getcwd()
src_dir = os.path.join(parent_dir, 'src')


sys.path.insert(0, src_dir)
import read

import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.io as pio

import numpy as np


gender_dict = {'Stable Diffusion' : read.prepare_dataset_for_pearson('gender', 'sd', 'female_prob'),
             'VQGAN-CLIP' : read.prepare_dataset_for_pearson('gender', 'vqgan', 'female_prob')}

age_dict = {'Stable Diffusion' : read.prepare_dataset_for_pearson('age', 'sd', 'old_prob'),
            'VQGAN-CLIP' : read.prepare_dataset_for_pearson('age', 'vqgan', 'old_prob')}

def correlation_scatterplot(df, img_generator,attribute, tick_top, tick_bottom, prob):

    unique_classifiers = list(set(df['classifier']))

    for classifier in unique_classifiers:

        df_classifier = df[df['classifier'] == classifier].reset_index(drop=True)

        x = df_classifier['rating.mean'].round(2)
        y = df_classifier[prob].round(2)
        name = df_classifier['name']
        type = df_classifier['type']

        xlimits = (-50, 50)
        ylimits = (0, 1)

        #color combination choosed with ColorBrewer for colorblind friendly visualization
        colors = {'real': '#d95f02', 'madeup': '#1b9e77', 'talking': '#7570b3'}
           
        unique_types = ['real', 'madeup', 'talking']
        
        # Create a subplot with 1 row and as many columns as there are unique types
        fig = sp.make_subplots(rows=1, cols=len(unique_types), subplot_titles=unique_types)
        
        # For each unique type
        for i, t in enumerate(unique_types):
            # Filter the data for the current type
            x_t = [x[j] for j in range(len(x)) if type[j] == t]
            y_t = [y[j] for j in range(len(y)) if type[j] == t]
            name_t = [name[j] for j in range(len(name)) if type[j] == t]

            color = colors[t]
            
            # Add a scatter trace with x, y, and name as text
            fig.add_trace(go.Scatter(name=t.capitalize(), x=x_t, y=y_t, mode="markers", text=[f"{n} ({t})" for n in name_t], marker_color=color, customdata=name_t,
                                    hovertemplate='Name: <b>%{customdata}</b><extra></extra><br>Classifier Probability: <b>%{y}</b><br>People Score: <b>%{x}</b>'), row=1, col=i+1,)
            
            # Calculate the coefficients of the best fit line
            coefficients = np.polyfit(x_t, y_t, 1)
            # Generate y values for the best fit line
            y_fit = np.polyval(coefficients, x_t)
            # Add the best fit line to the plot
            fig.add_trace(go.Scatter(x=x_t, y=y_fit, mode="lines", name="Best fit", line_color=color), row=1, col=i+1)

            # Update the x and y axis limits and labels with some padding
            padding = 0.1
            x_range = [xlimits[0] - (xlimits[1] - xlimits[0]) * padding, xlimits[1] + (xlimits[1] - xlimits[0]) * padding]
            y_range = [ylimits[0] - (ylimits[1] - ylimits[0]) * padding, ylimits[1] + (ylimits[1] - ylimits[0]) * padding]
            
            # Update the x and y axis limits and labels
            fig.update_xaxes(range=x_range, title="<b>people's score</b>", tickvals=[-50, 50], ticktext=[tick_bottom, tick_top], row=1, col=i+1)
            fig.update_yaxes(range=y_range, tickvals=[0, 1], ticktext=[tick_bottom, tick_top], row=1, col=i+1)
        
        fig.update_yaxes(title= "<b>classifier's prediction</b>",title_standoff=0, row=1, col=1)
        fig.update_layout(title_text=f"<b>{img_generator} - Correlation between people's perception and {classifier} predictions about {attribute} of characters' name</b>")


        pio.write_image(fig, os.path.join(parent_dir, f'plots/{attribute}/jpeg/{img_generator}_{classifier}_{attribute}_correlation_plot.jpeg'), width=1200, height=600)
        pio.write_html(fig, os.path.join(parent_dir, f'plots/{attribute}/html/{img_generator}_{classifier}_{attribute}_correlation_plot.html'))


def main():

    for img_gen, df in gender_dict.items():

        correlation_scatterplot(df, img_gen, 'gender', 'female', 'male', 'female_prob')

    for img_gen, df in age_dict.items():

        correlation_scatterplot(df, img_gen, 'age', 'old', 'young', 'old_prob')

if __name__ == "__main__":

    main()


