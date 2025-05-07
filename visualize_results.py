import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from model import AlzheimerCNN
from baselines import AlexNet3D, ResNet3D, ResNet3D50
from sklearn.metrics import roc_curve, auc

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Model to count parameters for
    
    Returns:
    --------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_info():
    """
    Get information about each model architecture.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing model information
    """
    models = {
        'alzheimer_cnn': AlzheimerCNN(widening_factor=8, use_age=True),
        'alexnet3d': AlexNet3D(use_age=True),
        'resnet3d': ResNet3D(use_age=True),
        'resnet3d50': ResNet3D50(use_age=True)
    }
    
    model_info = []
    
    for name, model in models.items():
        # Count parameters
        num_params = count_parameters(model)
        
        # Get number of layers
        num_layers = len(list(model.modules()))
        
        # Calculate number of convolutional and fully connected layers
        num_conv = sum(1 for m in model.modules() if isinstance(m, torch.nn.Conv3d))
        num_fc = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        
        # Add to list
        model_info.append({
            'Model': name,
            'Parameters': num_params,
            'Layers': num_layers,
            'Conv Layers': num_conv,
            'FC Layers': num_fc,
            'Has Age Encoding': model.use_age
        })
    
    return pd.DataFrame(model_info)

def visualize_model_complexity(info_df, output_dir):
    """
    Visualize model complexity metrics.
    
    Parameters:
    -----------
    info_df : pd.DataFrame
        DataFrame containing model information
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot number of parameters
    plt.figure(figsize=(10, 6))
    bars = plt.bar(info_df['Model'], info_df['Parameters'] / 1e6, color='skyblue')
    plt.title('Number of Parameters by Model (Millions)')
    plt.xlabel('Model')
    plt.ylabel('Parameters (Millions)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add parameter count above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_parameters.png'), dpi=300)
    plt.close()
    
    # Plot number of layers
    plt.figure(figsize=(12, 6))
    
    # Data to plot
    models = info_df['Model']
    layer_data = {
        'Conv Layers': info_df['Conv Layers'],
        'FC Layers': info_df['FC Layers']
    }
    
    # Create stacked bar chart
    bottom = np.zeros(len(models))
    
    for layer_type, count in layer_data.items():
        plt.bar(models, count, bottom=bottom, label=layer_type)
        bottom += count
    
    plt.title('Number of Layers by Type')
    plt.xlabel('Model')
    plt.ylabel('Number of Layers')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_layers.png'), dpi=300)
    plt.close()

def compare_performance_vs_complexity(info_df, performance_df, output_dir):
    """
    Compare model performance vs. complexity.
    
    Parameters:
    -----------
    info_df : pd.DataFrame
        DataFrame containing model information
    performance_df : pd.DataFrame
        DataFrame containing model performance metrics
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge dataframes
    merged_df = pd.merge(info_df, performance_df, on='Model')
    
    # Plot parameters vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['Parameters'] / 1e6, merged_df['Test Accuracy (%)'], 
                s=100, alpha=0.7)
    
    # Add model names as labels
    for i, model in enumerate(merged_df['Model']):
        plt.annotate(model, 
                    (merged_df['Parameters'].iloc[i] / 1e6, 
                     merged_df['Test Accuracy (%)'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add trend line
    if len(merged_df) > 2:  # Need at least 3 points for a meaningful trend line
        x = merged_df['Parameters'] / 1e6
        y = merged_df['Test Accuracy (%)']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, intercept + slope * x, 'r--', 
                label=f'Trend Line (r²={r_value**2:.2f})')
        plt.legend()
    
    plt.title('Model Performance vs. Complexity')
    plt.xlabel('Number of Parameters (Millions)')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_complexity.png'), dpi=300)
    plt.close()
    
    # Plot layers vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['Layers'], merged_df['Test Accuracy (%)'], 
                s=100, alpha=0.7)
    
    # Add model names as labels
    for i, model in enumerate(merged_df['Model']):
        plt.annotate(model, 
                    (merged_df['Layers'].iloc[i], 
                     merged_df['Test Accuracy (%)'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Add trend line
    if len(merged_df) > 2:  # Need at least 3 points for a meaningful trend line
        x = merged_df['Layers']
        y = merged_df['Test Accuracy (%)']
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        plt.plot(x, intercept + slope * x, 'r--', 
                label=f'Trend Line (r²={r_value**2:.2f})')
        plt.legend()
    
    plt.title('Model Performance vs. Number of Layers')
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_layers.png'), dpi=300)
    plt.close()

def create_radar_chart(info_df, performance_df, output_dir):
    """
    Create a radar chart comparing different model aspects.
    
    Parameters:
    -----------
    info_df : pd.DataFrame
        DataFrame containing model information
    performance_df : pd.DataFrame
        DataFrame containing model performance metrics
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Merge dataframes
    merged_df = pd.merge(info_df, performance_df, on='Model')
    
    # Select metrics for radar chart
    metrics = ['Test Accuracy (%)', 'Average F1 Score', 'Average ROC AUC']
    
    # Normalize parameters and layers for radar chart
    merged_df['Normalized Parameters'] = 100 * (merged_df['Parameters'] / merged_df['Parameters'].max())
    merged_df['Normalized Layers'] = 100 * (merged_df['Layers'] / merged_df['Layers'].max())
    
    # Create radar chart
    categories = ['Accuracy', 'F1 Score', 'ROC AUC', 'Parameters*', 'Layers*']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Number of categories
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set the first axis at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw category labels
    plt.xticks(angles[:-1], categories)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
    plt.ylim(0, 100)
    
    # Plot each model
    colors = plt.cm.viridis(np.linspace(0, 1, len(merged_df)))
    
    for i, (_, row) in enumerate(merged_df.iterrows()):
        # Get values for this model
        values = [
            row['Test Accuracy (%)'],  # Already in percentage
            row['Average F1 Score'] * 100,  # Convert to percentage
            row['Average ROC AUC'] * 100,  # Convert to percentage
            row['Normalized Parameters'],  # Already normalized
            row['Normalized Layers']  # Already normalized
        ]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Model'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison Radar Chart')
    plt.figtext(0.5, 0.01, '* Parameters and Layers are normalized relative to the maximum value', 
               ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_radar_chart.png'), dpi=300)
    plt.close()

def load_roc_curve_data(base_dir, models):
    """
    Load ROC curve data from model directories.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing model results
    models : list
        List of model names
    
    Returns:
    --------
    dict
        Dictionary containing ROC curve data for each model and class
    """
    class_names = ["CN", "MCI", "AD"]
    roc_data = {class_name: {} for class_name in class_names}
    
    for model_name in models:
        # Path to model directory
        model_dir = os.path.join(base_dir, model_name, model_name)
        
        # Path to ROC curve data
        roc_path = os.path.join(model_dir, "roc_curves_data.csv")
        
        # Check if ROC curve data exists
        if not os.path.exists(roc_path):
            print(f"ROC curve data not found for {model_name}")
            continue
        
        # Load ROC curve data
        try:
            roc_df = pd.read_csv(roc_path)
            
            # Extract data for each class
            for class_name in class_names:
                class_data = roc_df[roc_df['class'] == class_name]
                
                if len(class_data) > 0:
                    roc_data[class_name][model_name] = {
                        'fpr': class_data['fpr'].values,
                        'tpr': class_data['tpr'].values,
                        'auc': class_data['auc'].values[0]
                    }
        except Exception as e:
            print(f"Error loading ROC curve data for {model_name}: {e}")
    
    return roc_data

def plot_roc_curves_comparison(roc_data, output_dir):
    """
    Plot ROC curves comparison for each class.
    
    Parameters:
    -----------
    roc_data : dict
        Dictionary containing ROC curve data for each model and class
    output_dir : str
        Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for models
    models = set()
    for class_data in roc_data.values():
        models.update(class_data.keys())
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    color_map = {model: colors[i] for i, model in enumerate(sorted(models))}
    
    # Plot ROC curves for each class
    for class_name, class_data in roc_data.items():
        if not class_data:  # Skip if no data for this class
            continue
        
        plt.figure(figsize=(10, 8))
        
        # Plot random guessing line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')
        
        # Plot ROC curve for each model
        for model_name, model_data in class_data.items():
            fpr = model_data['fpr']
            tpr = model_data['tpr']
            auc_score = model_data['auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {auc_score:.3f})',
                    color=color_map[model_name])
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves Comparison - {class_name} Class')
        plt.legend(loc='lower right')
        
        # Set axis limits
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_comparison_{class_name}.png'), dpi=300)
        plt.close()
    
    # Plot average ROC AUC bar chart
    plt.figure(figsize=(10, 6))
    
    # Collect average AUC for each model
    avg_auc_data = []
    
    for model_name in models:
        model_aucs = []
        
        for class_name, class_data in roc_data.items():
            if model_name in class_data:
                model_aucs.append(class_data[model_name]['auc'])
        
        if model_aucs:
            avg_auc_data.append({
                'Model': model_name,
                'Average AUC': np.mean(model_aucs)
            })
    
    if avg_auc_data:
        avg_auc_df = pd.DataFrame(avg_auc_data)
        avg_auc_df = avg_auc_df.sort_values('Average AUC', ascending=False)
        
        # Plot bar chart
        bars = plt.bar(avg_auc_df['Model'], avg_auc_df['Average AUC'], 
                      color=[color_map[model] for model in avg_auc_df['Model']])
        
        # Add AUC value above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('Average ROC AUC Across Classes by Model')
        plt.xlabel('Model')
        plt.ylabel('Average AUC')
        plt.ylim([0.5, 1.05])  # AUC is typically between 0.5 and 1.0
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'average_auc_comparison.png'), dpi=300)
        plt.close()

def main():
    # Directory containing model results
    base_dir = 'widening_test_comparison_results_epochs-10_num_samples-300'
    results_dir = os.path.join(base_dir, "comparison_analysis")
    summary_file = os.path.join(results_dir, "model_comparison_summary.csv")
    
    if not os.path.exists(summary_file):
        print(f"Summary file not found at {summary_file}")
        return
    
    # Output directory
    output_dir = os.path.join(results_dir, "visualization")
    
    # Load performance data
    performance_df = pd.read_csv(summary_file)
    
    # Get model names from the summary file
    models = performance_df['Model'].tolist()
    
    # Load ROC curve data for all models
    print("Loading ROC curve data...")
    roc_data = load_roc_curve_data(base_dir, models)
    
    # Plot ROC curves comparison
    print("Plotting ROC curves comparison...")
    plot_roc_curves_comparison(roc_data, output_dir)
    
    # Get model architecture information
    print("Analyzing model architectures...")
    info_df = get_model_info()
    
    # Save model info
    # Check if output directory exists, create if not
    os.makedirs(output_dir, exist_ok=True)
    info_df.to_csv(os.path.join(output_dir, "model_architecture_info.csv"), index=False)
    
    # Visualize model complexity
    print("Visualizing model complexity...")
    visualize_model_complexity(info_df, output_dir)
    
    # Compare performance vs. complexity
    print("Comparing performance vs. complexity...")
    compare_performance_vs_complexity(info_df, performance_df, output_dir)
    
    # Create radar chart
    print("Creating radar chart...")
    create_radar_chart(info_df, performance_df, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()