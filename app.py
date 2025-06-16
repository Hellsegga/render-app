import dash
from dash import dcc, html, Output, Input
import plotly.graph_objects as go
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import requests
import io
import os
import pandas as pd
import pickle
from morphomics_exp.utils_analysis import get_2d, inverse_function

def download_pickle_from_gdrive(file_id):
    """
    Download a pickle file from Google Drive by file ID and return the unpickled object.
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

# Display modes:
# 'all' - show all samples
# 'mean' - show mean coordinates
DISPLAY_MODE = 'all'

# Image display modes:
# 'reconstruction' - show only VAE reconstruction
# 'both' - show both VAE reconstruction and actual image
IMAGE_DISPLAY_MODE = 'reconstruction'

# Create a Dash app
app = dash.Dash(__name__)

# Load VAE data and models
current_dir = os.path.dirname(__file__)

# Load VAE pipeline
vae_pip = download_pickle_from_gdrive("1tI38ff0p3EAs2eKXKctIkmGJIGSDh07D")


standardizer = vae_pip['standardizer']
pca, vae = vae_pip['fitted_pca_vae']

# Load reduced data
mf = download_pickle_from_gdrive("1-uqINacNBm89kS29-Kw2iK9rhh04RfLE")
mf = mf.reset_index()
mf.rename(columns={'index': 'old_idcs'}, inplace=True)

mf.loc[mf["Model"].isin(["Cx3cr1_het", "Cx3cr1_hom"]), "Condition"] = "Development"

# Filter data for visualization
cond = (mf["Region"]=="IPL") & (mf["Condition"]=="Development") & (mf["Model"]=="Cx3cr1_het")
mf_vae_kxa = mf[cond]

# Extract and convert time values
def extract_and_convert(input_string):
    if input_string == "Adult":
        return 30
    if pd.isna(input_string) or not any(char.isdigit() for char in str(input_string)):
        return np.nan
    try:
        underscore_pos = input_string.find("_")
        if underscore_pos == -1:
            underscore_pos = len(input_string)
        extracted_part = input_string[1:underscore_pos]
        return int(extracted_part)
    except ValueError:
        return None

mf_vae_kxa["Time"] = mf_vae_kxa["Time"].apply(extract_and_convert)

# Filter outliers
points = np.array(mf_vae_kxa['pca_vae'].tolist())
centroid = np.mean(points, axis=0)
distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
std_dev = np.std(distances)
threshold = 4 * std_dev
mf_vae_kxa = mf_vae_kxa[distances <= threshold].copy()

# Create scatter plot
if DISPLAY_MODE == 'all':
    # Use all points
    x, y = zip(*mf_vae_kxa['pca_vae'])
    time_values = mf_vae_kxa['Time']
    hover_text = [f"Region: {r}<br>Model: {m}<br>Time: {t}" 
                 for r, m, t in zip(mf_vae_kxa['Region'], mf_vae_kxa['Model'], mf_vae_kxa['Time'])]
else:
    # Calculate mean coordinates for each unique combination of Region, Model, Time
    # Define timepoints to keep for each group

    ## For OPL use these
    #group_timepoints = [10, 15, 29, 44, 65]  # Customize these timepoints
    #group_timepoints = [9, 15, 20, 25, 29]  # Customize these timepoints

    ## For IPL use these
    #group_timepoints = [10, 15, 20, 29, 40, 44, 55, 65, 90]  # Customize these timepoints
    group_timepoints = [1, 2, 4, 5, 8, 11, 13, 15, 17, 20, 25, 30]  # Customize these timepoints

    # All timepoints
    #group_timepoints = mf_vae_kxa['Time'].unique()

    # Filter dataframe for each group with their respective timepoints
    group_df = mf_vae_kxa[mf_vae_kxa['Time'].isin(group_timepoints)]

    # Combine filtered groups and calculate means
    means_per_timepoint = group_df.groupby(['Region', 'Condition', 'Model', 'Time'])["pca_vae"].mean().reset_index()
    
    # Sort by Time to ensure correct arrow direction
    means_per_timepoint = means_per_timepoint.sort_values('Time')
        
    # Extract x, y coordinates from mean_coords
    x, y = zip(*means_per_timepoint['pca_vae'])
    time_values = means_per_timepoint['Time']
    hover_text = [f"Region: {r}<br>Model: {m}<br>Time: {t}" 
                 for r, m, t in zip(means_per_timepoint['Region'], means_per_timepoint['Model'], means_per_timepoint['Time'])]

fig = go.Figure()

# Add arrows between points for each Region/Model combination
if DISPLAY_MODE == 'mean':
    # Get unique combinations of Region and Model
    unique_combinations = means_per_timepoint[['Region', 'Model']].drop_duplicates()
    
    for _, row in unique_combinations.iterrows():
        # Filter data for this Region/Model combination
        group_data = means_per_timepoint[
            (means_per_timepoint['Region'] == row['Region']) & 
            (means_per_timepoint['Model'] == row['Model'])
        ].sort_values('Time')
        
        # Create arrows between consecutive points
        for i in range(len(group_data) - 1):
            x_start, y_start = group_data.iloc[i]['pca_vae']
            x_end, y_end = group_data.iloc[i + 1]['pca_vae']
            
            # Calculate vector between points
            dx = x_end - x_start
            dy = y_end - y_start
            
            # Scale vector to make arrow shorter (95% of distance)
            scale = 0.95
            x_arrow = x_start + dx * scale
            y_arrow = y_start + dy * scale
            
            # Add arrow
            fig.add_annotation(
                x=x_arrow, y=y_arrow,
                ax=x_start, ay=y_start,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="gray"
            )

# Add scatter points
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=10,
        color=time_values,
        showscale=True,
        colorbar=dict(title='Time')
    ),
    name='data',
    text=hover_text,
    hoverinfo='text'
))

fig.update_layout(
    title='VAE Latent Space',
    hovermode='closest',
    dragmode='pan',
    xaxis_title='Latent Dimension 1',
    yaxis_title='Latent Dimension 2'
)

# App layout
if IMAGE_DISPLAY_MODE == 'reconstruction':
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter', figure=fig, style={'width': '45vw', 'height': '80vh'}),
            html.Img(id='image', style={'width': '45vw', 'height': '80vh', 'border': '1px solid black'})
        ], style={'display': 'flex'})
    ])
else:  # 'both' mode
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='scatter', figure=fig, style={'width': '45vw', 'height': '80vh'}),
            html.Div([
                html.Img(id='reconstruction', style={'width': '45vw', 'height': '40vh', 'border': '1px solid black'}),
                html.Img(id='actual', style={'width': '45vw', 'height': '40vh', 'border': '1px solid black'})
            ], style={'display': 'flex', 'flexDirection': 'column'})
        ], style={'display': 'flex'})
    ])

def generate_vae_image(hoverData):
    if hoverData is None:
        return dash.no_update
    
    # Get hover coordinates
    x, y = hoverData['points'][0]['x'], hoverData['points'][0]['y']
    mouse_pos = np.array([x, y])
    
    # Generate image using VAE
    img = inverse_function(mouse_pos, 
                          model=vae,
                          pca=pca,
                          scaler=standardizer,
                          filter=None)
    
    # Normalize and convert to 2D
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
    img_2d = get_2d(img_normalized)
    
    # Flip the image vertically (equivalent to invert_yaxis)
    img_2d = np.flipud(img_2d)
    
    # Convert to PIL Image and then to base64
    img_pil = Image.fromarray((img_2d * 255).astype(np.uint8))
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

def get_actual_image(hoverData):
    if hoverData is None:
        return dash.no_update
    
    # Get the point index from hoverData
    point_index = hoverData['points'][0]['pointIndex']
    
    # Get the actual image from the dataframe
    if DISPLAY_MODE == 'all':
        actual_img = mf_vae_kxa.iloc[point_index]['pi']
    else:
        # For mean mode, we need to find the corresponding point in the original data
        region = means_per_timepoint.iloc[point_index]['Region']
        model = means_per_timepoint.iloc[point_index]['Model']
        time = means_per_timepoint.iloc[point_index]['Time']
        
        # Find the corresponding point in the original data
        mask = (mf_vae_kxa['Region'] == region) & (mf_vae_kxa['Model'] == model) & (mf_vae_kxa['Time'] == time)
        actual_img = mf_vae_kxa[mask]['pi'].iloc[0]
    
    # Reshape the image to 100x100
    actual_img = actual_img.reshape(100, 100)
    
    # Normalize to 0-1 range
    actual_img = (actual_img - np.min(actual_img)) / (np.max(actual_img) - np.min(actual_img))
    
    # Convert to uint8 with proper scaling
    actual_img = (actual_img * 255).astype(np.uint8)
    
    # Flip the image vertically (equivalent to invert_yaxis)
    actual_img = np.flipud(actual_img)
    
    # Convert to PIL Image and then to base64
    img_pil = Image.fromarray(actual_img)
    buffer = BytesIO()
    img_pil.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

if IMAGE_DISPLAY_MODE == 'reconstruction':
    @app.callback(
        Output('image', 'src'),
        Input('scatter', 'hoverData')
    )
    def update_image_on_hover(hoverData):
        return generate_vae_image(hoverData)
else:  # 'both' mode
    @app.callback(
        [Output('reconstruction', 'src'),
         Output('actual', 'src')],
        Input('scatter', 'hoverData')
    )
    def update_images_on_hover(hoverData):
        return generate_vae_image(hoverData), get_actual_image(hoverData)

if __name__ == '__main__':
    #app.run(debug=True, host='127.0.0.1', port=8050, use_reloader=False)
    app.run_server(debug=False, host='0.0.0.0', port=8080)
