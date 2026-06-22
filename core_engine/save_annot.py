import sys
import os
import pandas as pd
from PIL import Image, ImageDraw

def map_detections_to_image(csv_file_path, image_path, output_path=None):
    """
    Map detected annotations from CSV onto the original image and save it directly (no plotting)
    Also saves the CSV file to an annotations folder in the main directory
    """
    # Read the CSV data
    df = pd.read_csv(csv_file_path)

    # Load the original image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Define colors for different groups/clusters
    colors = [
        (255, 0, 0),      # red
        (0, 0, 255),      # blue
        (0, 255, 0),      # green
        (255, 165, 0),    # orange
        (128, 0, 128),    # purple
        (255, 255, 0),    # yellow
        (0, 255, 255),    # cyan
        (255, 0, 255),    # magenta
    ]

    # Draw rectangles for each detection
    for idx, row in df.iterrows():
        if row['IsValid'] == 1:
            x, y = row['X'], row['Y']
            radius = row['Radius']
            color_group = int(row['Colour_group'])
            color = colors[color_group % len(colors)]
            side = 2 * radius
            left_up = (x - radius, y - radius)
            right_down = (x + radius, y + radius)
            draw.rectangle([left_up, right_down], outline=color, width=3)

    # Save the result
    csv_output_path = None
    if output_path:
        img.save(output_path)
        print(f"Annotated image saved to: {output_path}")
        
        # Create annotations folder in the main directory (two levels up from core_engine)
        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        annotations_dir = os.path.join(main_dir, "annotations")
        
        # Create annotations directory if it doesn't exist
        if not os.path.exists(annotations_dir):
            os.makedirs(annotations_dir)
            print(f"Created annotations directory: {annotations_dir}")
        
        # Generate CSV filename based on output image name
        output_image_name = os.path.splitext(os.path.basename(output_path))[0]
        csv_filename = f"{output_image_name}_annotations.csv"
        csv_output_path = os.path.join(annotations_dir, csv_filename)
        df.to_csv(csv_output_path, index=False)
        print(f"CSV file saved to annotations folder: {csv_output_path}")
        
    return img, csv_output_path

def create_detection_summary(csv_file_path):
    """
    Create a summary of the detections
    """
    df = pd.read_csv(csv_file_path)
    
    print("=== DETECTION SUMMARY ===")
    print(f"Total detections: {len(df)}")
    print(f"Valid detections: {df['IsValid'].sum()}")
    print(f"ROI distribution: {df['ROI'].value_counts().to_dict()}")
    print(f"Color groups: {sorted(df['Colour_group'].unique())}")
    print(f"Cluster sizes: {df['N_in_clust'].value_counts().to_dict()}")
    print(f"Area range: {df['Area'].min():.1f} - {df['Area'].max():.1f}")
    print(f"Radius range: {df['Radius'].min():.1f} - {df['Radius'].max():.1f}")
    
    return df.describe()

def create_interactive_plot(csv_file_path, image_path):
    """
    Create an interactive plot with hover information
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    
    df = pd.read_csv(csv_file_path)
    img = Image.open(image_path)
    img_array = np.array(img)
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_array)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    
    # Store detection info for hover
    detection_info = []
    
    for idx, row in df.iterrows():
        if row['IsValid'] == 1:
            x, y = row['X'], row['Y']
            radius = row['Radius']
            color_group = int(row['Colour_group'])
            color = colors[color_group % len(colors)]
            
            circle = patches.Circle((x, y), radius, linewidth=2, 
                                  edgecolor=color, facecolor='none', alpha=0.7)
            ax.add_patch(circle)
            ax.plot(x, y, 'o', color=color, markersize=6)
            
            # Store info for this detection
            info = {
                'x': x, 'y': y, 'radius': radius, 'roi': row['ROI'],
                'color_group': color_group, 'area': row['Area'],
                'hue': row['Hue'], 'saturation': row['Saturation'],
                'rgb_mean': (row['Rmean'], row['Gmean'], row['Bmean'])
            }
            detection_info.append(info)
    
    ax.set_title('Interactive Detection Viewer\n(Click on detections for details)', 
                fontsize=14, fontweight='bold')
    
    # Add click handler
    def on_click(event):
        if event.inaxes != ax:
            return
        
        # Find closest detection
        click_x, click_y = event.xdata, event.ydata
        min_dist = float('inf')
        closest_detection = None
        
        for detection in detection_info:
            dist = np.sqrt((detection['x'] - click_x)**2 + (detection['y'] - click_y)**2)
            if dist < detection['radius'] and dist < min_dist:
                min_dist = dist
                closest_detection = detection
        
        if closest_detection:
            info_text = f"""
Detection Details:
Position: ({closest_detection['x']:.1f}, {closest_detection['y']:.1f})
ROI: {closest_detection['roi']}
Color Group: {closest_detection['color_group']}
Area: {closest_detection['area']}
Radius: {closest_detection['radius']:.1f}
Hue: {closest_detection['hue']:.1f}
Saturation: {closest_detection['saturation']:.1f}
RGB Mean: ({closest_detection['rgb_mean'][0]:.1f}, {closest_detection['rgb_mean'][1]:.1f}, {closest_detection['rgb_mean'][2]:.1f})
            """
            print(info_text)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# Example usage
if __name__ == "__main__":
    if len(sys.argv) >= 4:
        csv_path = sys.argv[1]
        image_path = sys.argv[2]
        output_path = sys.argv[3]
        result_img, csv_output_path = map_detections_to_image(csv_path, image_path, output_path)
        if csv_output_path:
            print(f"Annotation CSV saved to: {csv_output_path}")
    else:
        print("Usage: python save_annot.py <csv_path> <image_path> <output_path>")
        sys.exit(1)

# Additional utility functions

def filter_detections_by_criteria(csv_file_path, min_area=None, max_area=None, 
                                roi_filter=None, color_group_filter=None):
    """
    Filter detections based on various criteria
    """
    df = pd.read_csv(csv_file_path)
    
    filtered_df = df[df['IsValid'] == 1].copy()
    
    if min_area is not None:
        filtered_df = filtered_df[filtered_df['Area'] >= min_area]
    if max_area is not None:
        filtered_df = filtered_df[filtered_df['Area'] <= max_area]
    if roi_filter is not None:
        filtered_df = filtered_df[filtered_df['ROI'] == roi_filter]
    if color_group_filter is not None:
        filtered_df = filtered_df[filtered_df['Colour_group'] == color_group_filter]
    
    return filtered_df

def export_detections_to_formats(csv_file_path, output_dir=None):
    """
    Export detections to various formats (JSON, XML, etc.)
    """
    import json
    
    df = pd.read_csv(csv_file_path)
    
    # Use annotations folder as default output directory
    if output_dir is None:
        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(main_dir, "annotations")
        
        # Create annotations directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created annotations directory: {output_dir}")
    
    # Export to JSON
    json_data = df.to_dict('records')
    with open(f"{output_dir}/detections.json", 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Export filtered data
    valid_detections = df[df['IsValid'] == 1]
    valid_detections.to_csv(f"{output_dir}/valid_detections_only.csv", index=False)
    
    print(f"Exported data to {output_dir}")