import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import re

# Define paths for the dataset you want to plot
# Uncomment the dataset you want to use

# MNIST Random
# model_folders = [
#     "data/MNIST-RANDOM/GI_random_experiment_outputs",
#     "data/MNIST-RANDOM-100/GI_random_experiment_outputs"]
# dataset = 'MNIST-RANDOM'
# model_names = ['RANDOM-10', 'RANDOM-100']
# CIFAR 10 folder
# model_folders = [
#     "experiments/results/test_to_train_gis/cifar_10_resnet_20/GI_cifar10_real_resnet20_experiment_outputs",
#     "experiments/results/test_to_train_gis/cifar10_vgg16_bn/GI_cifar10_real_vgg16_bn_experiment_outputs"
# ]
# dataset = 'CIFAR10'
# model_names = ['ResNet20', 'VGG16']



# # Imagenette folders
# model_folders = [
#     "experiments/results/test_to_train_gis/imagenette_vgg16_random/GI_imagenette_real_VGG16_experiment_outputs",
#     "experiments/results/test_to_train_gis/imagenette_resnet50/GI_imagenette_real_ResNet50_experiment_outputs",
#     "experiments/results/test_to_train_gis/imagenette_deit_small/GI_imagenette_real_DeiT-Small_experiment_outputs",
#     "experiments/results/test_to_train_gis/imagenette_vit_base/GI_imagenette_real_ViT-Base_experiment_outputs"
# ]
# dataset = 'Imagenette'
# model_names = ['VGG16', 'ResNet50', 'DeiT-S', 'ViT-B']

# MNIST Folders
model_folders = [
    "experiments/results/test_to_train_gis/mnist_original_mlp/GI_experiment_outputs_plainnn",
    "experiments/results/test_to_train_gis/mnist_original_cnn/GI_experiment_outputs_cnn",
    "experiments/results/test_to_train_gis/mnist_original_vit/GI_experiment_outputs_vit"
]
dataset = 'MNIST'
model_names = ['MLP', 'CNN', 'ViT']




# Data structures to store our results
all_models_correct_data = []
all_models_misclassified_data = []
positions = []

# Check if any model has misclassified data
has_any_misclassified = False
for folder in model_folders:
    misclassified_files = glob.glob(os.path.join(folder, "misclassified_from_class_*.csv"))
    if misclassified_files:
        has_any_misclassified = True
        break

# Create compact figure with minimal margins
num_models = len(model_names)
width_per_model = 1.2 if has_any_misclassified else 1.0
fig_width = max(6, num_models * width_per_model)
fig_height = 3  # shorter height
plt.figure(figsize=(fig_width, fig_height))

plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)


# Colors for correct and misclassified
# Colors for correct and misclassified
colors = ['#A8D5BA', '#F4A7A7']  # mild green for correct, mild red for misclassified

# Position counter for plotting
position_counter = 0

# Lists to store box plots for the legend
box_correct = None
box_misclassified = None

# Loop through each model's results folder
for i, (folder, model_name) in enumerate(zip(model_folders, model_names)):
    print(f"\nProcessing {model_name} from folder: {folder}")
    
    # Get all CSV files for correct classifications
    correct_files = glob.glob(os.path.join(folder, "from_train_to_test_GIs_class_*.csv"))
    # Get all CSV files for misclassifications
    misclassified_files = glob.glob(os.path.join(folder, "misclassified_from_class_*.csv"))
    
    print(f"Found {len(correct_files)} correct classification files")
    print(f"Found {len(misclassified_files)} misclassification files")
    
    # Process correct classification files
    correct_data = []
    for file in correct_files:
        try:
            df = pd.read_csv(file)
            if df.empty:
                continue
                
            # Extract percent_gi_above_1 if it exists
            if "percent_gi_above_1" in df.columns:
                correct_data.append(df["percent_gi_above_1"].iloc[0])
            # If the file has a GI in the first column with no header, use that
            elif df.shape[1] >= 1 and df.columns[0].startswith("Unnamed"):
                values = df.iloc[:, 0].dropna().astype(float).tolist()
                correct_data.extend(values)
            else:
                # Try to extract numeric data from the first column
                values = df.iloc[:, 0].dropna()
                if values.apply(lambda x: isinstance(x, (int, float))).all():
                    correct_data.extend(values.astype(float).tolist())
                
        except Exception as e:
            print(f"Error reading {os.path.basename(file)}: {e}")
    
    # Process misclassification files
    misclassified_data = []
    if has_any_misclassified:  # Only try to process misclassified data if any model has it
        for file in misclassified_files:
            try:
                df = pd.read_csv(file)
                if df.empty:
                    continue
                    
                # Extract percent_gi_above_1 if it exists
                if "percent_gi_above_1" in df.columns:
                    misclassified_data.append(df["percent_gi_above_1"].iloc[0])
                # If the file has a GI in the first column with no header, use that
                elif df.shape[1] >= 1 and df.columns[0].startswith("Unnamed"):
                    values = df.iloc[:, 0].dropna().astype(float).tolist()
                    misclassified_data.extend(values)
                else:
                    # Try to extract numeric data from the first column
                    values = df.iloc[:, 0].dropna()
                    if values.apply(lambda x: isinstance(x, (int, float))).all():
                        misclassified_data.extend(values.astype(float).tolist())
                    
            except Exception as e:
                print(f"Error reading {os.path.basename(file)}: {e}")
    
    # Skip this model if we didn't find any data
    if not correct_data and not misclassified_data:
        print(f"No valid data found for {model_name}, skipping...")
        continue
    
    # Store the data
    all_models_correct_data.append(correct_data)
    all_models_misclassified_data.append(misclassified_data)
    
    # Print some stats
    print(f"Model: {model_name}")
    print(f"Correct samples: {len(correct_data)}")
    print(f"Misclassified samples: {len(misclassified_data)}")
    if correct_data:
        print(f"Correct GI range: {min(correct_data)} to {max(correct_data)}")
    if misclassified_data:
        print(f"Misclassified GI range: {min(misclassified_data)} to {max(misclassified_data)}")

    # Calculate positions for the box plots
    if has_any_misclassified:
        # If we have misclassified data for any model, create positions for both correct and misclassified
        pos_correct = position_counter
        pos_misclassified = position_counter + 0.4
        positions.append((pos_correct, pos_misclassified))
        position_counter += 1.2
    else:
        # If no misclassified data, just use a single position per model
        pos = position_counter
        positions.append(pos)
        position_counter += 1.0

# Now plot all the data
for i, (model_name, correct_data, misclassified_data) in enumerate(zip(model_names, all_models_correct_data, all_models_misclassified_data)):
    if has_any_misclassified:
        # Plot pairs of box plots (correct and misclassified)
        pos_correct, pos_misclassified = positions[i]
        
        # Plot correct data
        if correct_data:
            box1 = plt.boxplot([correct_data], positions=[pos_correct], 
                    patch_artist=True, widths=0.35, showfliers=False)
            # Color the boxes
            for patch in box1['boxes']:
                patch.set_facecolor(colors[0])
            
            # Save the box objects for the legend
            if box_correct is None:
                box_correct = box1
        
        # Plot misclassified data if available
        if misclassified_data:
            box2 = plt.boxplot([misclassified_data], positions=[pos_misclassified], 
                    patch_artist=True, widths=0.35, showfliers=False)
            # Color the boxes
            for patch in box2['boxes']:
                patch.set_facecolor(colors[1])
                
            # Save the box objects for the legend
            if box_misclassified is None:
                box_misclassified = box2
    else:
        # Plot single box plots (correct only)
        pos = positions[i]
        
        # Plot correct data
        if correct_data:
            box = plt.boxplot([correct_data], positions=[pos], 
                    patch_artist=True, widths=0.6, showfliers=False)
            # Color the boxes
            for patch in box['boxes']:
                patch.set_facecolor(colors[0])
            
            # Save the box objects for the legend
            if box_correct is None:
                box_correct = box

# Set x-tick positions and labels
if positions:
    if has_any_misclassified:
        # For datasets with both correct and misclassified
        model_positions = [sum(pos)/2 for pos in positions]
        plt.xticks(model_positions, model_names)
    else:
        # For datasets with only correct classifications
        plt.xticks(positions, model_names)
    
    # Add line at GI=1 (threshold for generalization)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title with reduced font size for compactness
    plt.ylabel('Percentage of GI > 1', fontsize=12)
    plt.title(f'{dataset} - GI Distribution Across Models', fontsize=14)
    
    # Create custom legend based on available data
    legend_elements = []
    legend_labels = []
    
    if box_correct:
        legend_elements.append(box_correct["boxes"][0])
        legend_labels.append('Correctly Classified')
    
    if box_misclassified:
        legend_elements.append(box_misclassified["boxes"][0])
        legend_labels.append('Misclassified')
    
    if legend_elements:
        plt.legend(legend_elements, legend_labels, loc='lower right', 
                  frameon=True, framealpha=0.8, fontsize=10)
    
    # Enhance appearance
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add y-axis limits for cleaner appearance
    plt.ylim(70, 102)  # 0-100% with a small margin
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'{dataset}_model_comparison_GIs.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No data found for plotting. Check your file paths and CSV format.")
