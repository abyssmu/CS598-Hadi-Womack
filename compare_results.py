import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import glob
import sys

def load_results(base_dir):
    """
    Load results from all models in the base directory.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing model subdirectories
    
    Returns:
    --------
    dict
        Dictionary containing results for each model
    """
    print(f"Looking for results in: {base_dir}")
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(base_dir) 
                 if os.path.isdir(os.path.join(base_dir, d)) 
                 and d != "comparison_analysis"]
    
    print(f"Found model directories: {model_dirs}")
    
    results = {}
    for model_name in model_dirs:
        model_dir = os.path.join(base_dir, model_name)
        metrics_dir = os.path.join(model_dir + '/' + model_name, "metrics")
        
        # Skip if metrics directory doesn't exist
        if not os.path.exists(metrics_dir):
            print(f"Metrics directory not found for {model_name}, skipping...")
            continue
        
        print(f"Loading metrics for {model_name} from {metrics_dir}")
        
        # List all files in metrics directory
        metric_files = os.listdir(metrics_dir)
        print(f"  Found metric files: {metric_files}")
        
        # Load training metrics
        train_path = os.path.join(metrics_dir, "training_metrics.csv")
        if os.path.exists(train_path):
            train_metrics = pd.read_csv(train_path)
            print(f"  Loaded training metrics with {len(train_metrics)} epochs")
        else:
            print(f"  No training metrics found for {model_name}")
            train_metrics = None
        
        # Load test summary
        test_path = os.path.join(metrics_dir, "test_summary.csv")
        if os.path.exists(test_path):
            test_summary = pd.read_csv(test_path)
            # Convert to dictionary
            test_metrics = {row['metric']: row['value'] for _, row in test_summary.iterrows()}
            print(f"  Loaded test metrics: {test_metrics}")
        else:
            print(f"  No test summary found for {model_name}")
            test_metrics = None
        
        # Load classification report
        report_path = os.path.join(metrics_dir, "classification_report.csv")
        if os.path.exists(report_path):
            classification_report_df = pd.read_csv(report_path)
            print(f"  Loaded classification report with {len(classification_report_df)} rows")
        else:
            print(f"  No classification report found for {model_name}")
            classification_report_df = None
        
        # Load confusion matrix
        cm_path = os.path.join(metrics_dir, "confusion_matrix.csv")
        if os.path.exists(cm_path):
            confusion_matrix_df = pd.read_csv(cm_path)
            print(f"  Loaded confusion matrix with shape {confusion_matrix_df.shape}")
        else:
            print(f"  No confusion matrix found for {model_name}")
            confusion_matrix_df = None
        
        # Load ROC AUC
        roc_path = os.path.join(metrics_dir, "roc_auc.csv")
        if os.path.exists(roc_path):
            roc_auc = pd.read_csv(roc_path)
            print(f"  Loaded ROC AUC with {len(roc_auc)} classes")
        else:
            print(f"  No ROC AUC found for {model_name}")
            roc_auc = None
        
        # Store results
        results[model_name] = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'classification_report': classification_report_df,
            'confusion_matrix': confusion_matrix_df,
            'roc_auc': roc_auc
        }
    
    return results

def compare_training_curves(results, output_dir):
    """
    Compare training curves for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each model
    output_dir : str
        Directory to save comparison plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter models with training metrics
    models_with_training = {
        model: data for model, data in results.items() 
        if data['train_metrics'] is not None
    }
    
    if not models_with_training:
        print("No models with training metrics found, skipping training curve comparison...")
        return
    
    print(f"Comparing training curves for {len(models_with_training)} models")
    
    # Create figure for loss curves
    plt.figure(figsize=(12, 6))
    
    # Plot training loss for each model
    for model, data in models_with_training.items():
        train_df = data['train_metrics']
        if 'epoch' not in train_df.columns:
            print(f"  Warning: 'epoch' column missing in training metrics for {model}")
            train_df['epoch'] = range(1, len(train_df) + 1)
            
        plt.plot(train_df['epoch'], train_df['train_loss'], '-', label=f"{model} (Train)")
        plt.plot(train_df['epoch'], train_df['val_loss'], '--', label=f"{model} (Val)")
    
    plt.title('Loss Curves Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_comparison.png"), dpi=300)
    plt.close()
    
    # Create figure for accuracy curves
    plt.figure(figsize=(12, 6))
    
    # Plot training accuracy for each model
    for model, data in models_with_training.items():
        train_df = data['train_metrics']
        if 'epoch' not in train_df.columns:
            train_df['epoch'] = range(1, len(train_df) + 1)
            
        plt.plot(train_df['epoch'], train_df['train_acc'], '-', label=f"{model} (Train)")
        plt.plot(train_df['epoch'], train_df['val_acc'], '--', label=f"{model} (Val)")
    
    plt.title('Accuracy Curves Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    print("Training curve comparison completed")

def compare_test_metrics(results, output_dir):
    """
    Compare test metrics for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each model
    output_dir : str
        Directory to save comparison plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter models with test metrics
    models_with_test = {
        model: data for model, data in results.items() 
        if data['test_metrics'] is not None
    }
    
    if not models_with_test:
        print("No models with test metrics found, skipping test metrics comparison...")
        return
    
    print(f"Comparing test metrics for {len(models_with_test)} models")
    
    # Extract test metrics
    test_loss = {}
    test_acc = {}
    
    for model, data in models_with_test.items():
        test_loss[model] = data['test_metrics'].get('test_loss', np.nan)
        # Check both 'test_acc' and 'Test Accuracy (%)' keys
        if 'test_acc' in data['test_metrics']:
            test_acc[model] = data['test_metrics']['test_acc']
        elif 'Test Accuracy (%)' in data['test_metrics']:
            test_acc[model] = data['test_metrics']['Test Accuracy (%)']
        else:
            test_acc[model] = np.nan
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'Model': list(test_loss.keys()),
        'Test Loss': list(test_loss.values()),
        'Test Accuracy (%)': list(test_acc.values())
    })
    
    # Sort by accuracy (descending)
    test_df = test_df.sort_values('Test Accuracy (%)', ascending=False)
    
    # Save DataFrame
    test_df.to_csv(os.path.join(output_dir, "test_metrics_comparison.csv"), index=False)
    
    # Create figure for test metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot test loss
    sns.barplot(x='Model', y='Test Loss', data=test_df, ax=ax1, palette='Blues_d')
    ax1.set_title('Test Loss Comparison')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot test accuracy
    sns.barplot(x='Model', y='Test Accuracy (%)', data=test_df, ax=ax2, palette='Greens_d')
    ax2.set_title('Test Accuracy Comparison')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "test_metrics_comparison.png"), dpi=300)
    plt.close()
    
    print("Test metrics comparison completed")

def compare_class_metrics(results, output_dir):
    """
    Compare per-class metrics for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each model
    output_dir : str
        Directory to save comparison plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter models with classification reports
    models_with_reports = {}
    for model, data in results.items():
        if data['classification_report'] is not None:
            # Handle different formats of classification report
            report_df = data['classification_report']
            
            # Check if the first column is the index
            if report_df.columns[0] == 'Unnamed: 0':
                report_df = report_df.set_index('Unnamed: 0')
            
            models_with_reports[model] = {'classification_report': report_df}
    
    if not models_with_reports:
        print("No models with classification reports found, skipping class metrics comparison...")
        return
    
    print(f"Comparing class metrics for {len(models_with_reports)} models")
    
    # Extract per-class F1 scores
    class_names = ['CN', 'MCI', 'AD']
    class_f1 = {}
    
    for model, data in models_with_reports.items():
        report = data['classification_report']
        model_f1 = {}
        
        for class_name in class_names:
            try:
                # Try to get F1 score from different formats
                if class_name in report.index:
                    # Format where classes are in the index
                    if 'f1-score' in report.columns:
                        model_f1[class_name] = report.loc[class_name, 'f1-score']
                    else:
                        model_f1[class_name] = np.nan
                elif class_name in report.values:
                    # Format where classes are in a column
                    cls_row = report[report.iloc[:, 0] == class_name]
                    if 'f1-score' in report.columns:
                        model_f1[class_name] = cls_row['f1-score'].values[0]
                    else:
                        model_f1[class_name] = np.nan
                else:
                    model_f1[class_name] = np.nan
            except Exception as e:
                print(f"  Error extracting F1 score for {class_name} in {model}: {e}")
                model_f1[class_name] = np.nan
        
        class_f1[model] = model_f1
    
    # Create DataFrame
    f1_data = []
    for model, f1_scores in class_f1.items():
        for class_name, f1 in f1_scores.items():
            f1_data.append({
                'Model': model,
                'Class': class_name,
                'F1 Score': f1
            })
    
    f1_df = pd.DataFrame(f1_data)
    
    # Save DataFrame
    f1_df.to_csv(os.path.join(output_dir, "class_f1_comparison.csv"), index=False)
    
    # Create figure for F1 scores
    plt.figure(figsize=(12, 6))
    
    # Plot F1 scores
    sns.barplot(x='Class', y='F1 Score', hue='Model', data=f1_df)
    plt.title('F1 Score Comparison by Class')
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.legend(title='Model')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_f1_comparison.png"), dpi=300)
    plt.close()
    
    # Also compare precision and recall
    metrics = ['precision', 'recall']
    
    for metric in metrics:
        # Extract per-class metric scores
        class_metric = {}
        
        for model, data in models_with_reports.items():
            report = data['classification_report']
            model_metric = {}
            
            for class_name in class_names:
                try:
                    # Try to get metric from different formats
                    if class_name in report.index:
                        # Format where classes are in the index
                        if metric in report.columns:
                            model_metric[class_name] = report.loc[class_name, metric]
                        else:
                            model_metric[class_name] = np.nan
                    elif class_name in report.values:
                        # Format where classes are in a column
                        cls_row = report[report.iloc[:, 0] == class_name]
                        if metric in report.columns:
                            model_metric[class_name] = cls_row[metric].values[0]
                        else:
                            model_metric[class_name] = np.nan
                    else:
                        model_metric[class_name] = np.nan
                except Exception as e:
                    print(f"  Error extracting {metric} for {class_name} in {model}: {e}")
                    model_metric[class_name] = np.nan
            
            class_metric[model] = model_metric
        
        # Create DataFrame
        metric_data = []
        for model, scores in class_metric.items():
            for class_name, score in scores.items():
                metric_data.append({
                    'Model': model,
                    'Class': class_name,
                    f'{metric.capitalize()}': score
                })
        
        metric_df = pd.DataFrame(metric_data)
        
        # Save DataFrame
        metric_df.to_csv(os.path.join(output_dir, f"class_{metric}_comparison.csv"), index=False)
        
        # Create figure for metric scores
        plt.figure(figsize=(12, 6))
        
        # Plot metric scores
        sns.barplot(x='Class', y=f'{metric.capitalize()}', hue='Model', data=metric_df)
        plt.title(f'{metric.capitalize()} Comparison by Class')
        plt.xlabel('Class')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend(title='Model')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"class_{metric}_comparison.png"), dpi=300)
        plt.close()
    
    print("Class metrics comparison completed")

def compare_roc_curves(results, output_dir):
    """
    Compare ROC AUC scores for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each model
    output_dir : str
        Directory to save comparison plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter models with ROC AUC
    models_with_roc = {
        model: data for model, data in results.items() 
        if data['roc_auc'] is not None
    }
    
    if not models_with_roc:
        print("No models with ROC AUC found, skipping ROC comparison...")
        return
    
    print(f"Comparing ROC curves for {len(models_with_roc)} models")
    
    # Extract ROC AUC scores
    roc_data = []
    
    for model, data in models_with_roc.items():
        roc_df = data['roc_auc']
        
        if 'class' not in roc_df.columns or 'auc' not in roc_df.columns:
            print(f"  Warning: Expected columns 'class' and 'auc' missing in ROC data for {model}")
            continue
            
        for _, row in roc_df.iterrows():
            roc_data.append({
                'Model': model,
                'Class': row['class'],
                'AUC': row['auc']
            })
    
    if not roc_data:
        print("No valid ROC AUC data found after parsing")
        return
        
    # Create DataFrame
    roc_df = pd.DataFrame(roc_data)
    
    # Save DataFrame
    roc_df.to_csv(os.path.join(output_dir, "roc_auc_comparison.csv"), index=False)
    
    # Create figure for ROC AUC scores
    plt.figure(figsize=(12, 6))
    
    # Plot ROC AUC scores
    sns.barplot(x='Class', y='AUC', hue='Model', data=roc_df)
    plt.title('ROC AUC Comparison by Class')
    plt.xlabel('Class')
    plt.ylabel('AUC')
    plt.legend(title='Model')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_comparison.png"), dpi=300)
    plt.close()
    
    # Calculate average AUC for each model
    avg_auc = roc_df.groupby('Model')['AUC'].mean().reset_index()
    avg_auc = avg_auc.sort_values('AUC', ascending=False)
    
    # Create figure for average AUC
    plt.figure(figsize=(10, 6))
    
    # Plot average AUC
    sns.barplot(x='Model', y='AUC', data=avg_auc, palette='viridis')
    plt.title('Average ROC AUC Across Classes')
    plt.xlabel('Model')
    plt.ylabel('Average AUC')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_roc_auc_comparison.png"), dpi=300)
    plt.close()
    
    print("ROC curve comparison completed")

def create_summary_table(results, output_dir):
    """
    Create a summary table of key metrics for all models.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing results for each model
    output_dir : str
        Directory to save summary table
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key metrics
    summary_data = []
    
    for model, data in results.items():
        # Prepare a default row with model name
        row = {'Model': model}
        
        # Test metrics
        if data['test_metrics']:
            # Test loss
            if 'test_loss' in data['test_metrics']:
                row['Test Loss'] = data['test_metrics']['test_loss']
            
            # Test accuracy (check both possible keys)
            if 'test_acc' in data['test_metrics']:
                row['Test Accuracy (%)'] = data['test_metrics']['test_acc']
            elif 'Test Accuracy (%)' in data['test_metrics']:
                row['Test Accuracy (%)'] = data['test_metrics']['Test Accuracy (%)']
        
        # Class F1 scores from classification report
        if data['classification_report'] is not None:
            report = data['classification_report']
            
            for class_name in ['CN', 'MCI', 'AD']:
                # Try to find the class in the report
                try:
                    if class_name in report.index:
                        # Format where classes are in the index
                        if 'f1-score' in report.columns:
                            row[f'F1 Score - {class_name}'] = report.loc[class_name, 'f1-score']
                    elif class_name in report.values:
                        # Format where classes are in a column
                        cls_row = report[report.iloc[:, 0] == class_name]
                        if 'f1-score' in report.columns:
                            row[f'F1 Score - {class_name}'] = cls_row['f1-score'].values[0]
                except:
                    # If anything goes wrong, just leave it as NaN
                    pass
        
        # ROC AUC
        if data['roc_auc'] is not None:
            roc_df = data['roc_auc']
            
            if 'class' in roc_df.columns and 'auc' in roc_df.columns:
                for _, roc_row in roc_df.iterrows():
                    class_name = roc_row['class']
                    row[f'ROC AUC - {class_name}'] = roc_row['auc']
        
        # Add to summary data
        summary_data.append(row)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Calculate average F1 and AUC if possible
    if summary_df.shape[0] > 0:
        # Average F1 Score
        f1_cols = [col for col in summary_df.columns if col.startswith('F1 Score - ')]
        if f1_cols:
            summary_df['Average F1 Score'] = summary_df[f1_cols].mean(axis=1)
        
        # Average ROC AUC
        auc_cols = [col for col in summary_df.columns if col.startswith('ROC AUC - ')]
        if auc_cols:
            summary_df['Average ROC AUC'] = summary_df[auc_cols].mean(axis=1)
        
        # Sort by average F1 score (descending) if available
        if 'Average F1 Score' in summary_df.columns:
            summary_df = summary_df.sort_values('Average F1 Score', ascending=False)
    
    # Save DataFrame
    summary_df.to_csv(os.path.join(output_dir, "model_comparison_summary.csv"), index=False)
    
    # Print summary to console
    if len(summary_df) > 0:
        print("\nSummary of model comparison:")
        
        # Determine which columns to display in summary
        display_cols = ['Model']
        if 'Test Accuracy (%)' in summary_df.columns:
            display_cols.append('Test Accuracy (%)')
        if 'Average F1 Score' in summary_df.columns:
            display_cols.append('Average F1 Score')
        if 'Average ROC AUC' in summary_df.columns:
            display_cols.append('Average ROC AUC')
        
        # Display summary
        display_df = summary_df[display_cols].copy()
        
        # Format numeric columns
        for col in display_cols[1:]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
        print(display_df.to_string(index=False))
    
    return summary_df

def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default directory
        base_dir = 'comparison_results_epochs-10_num_samples-300'
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Base directory '{base_dir}' not found!")
        print("Please specify a valid results directory or check your path.")
        print("Usage: python compare_results_fixed.py [base_directory]")
        return
    
    # Output directory for comparison results
    output_dir = os.path.join(base_dir, "comparison_analysis")
    
    # Load results
    print("\nLoading results...")
    results = load_results(base_dir)
    
    if not results:
        print("No results found in the specified directory.")
        return
    
    print(f"\nFound results for {len(results)} models: {list(results.keys())}")
    
    # Compare training curves
    print("\nComparing training curves...")
    compare_training_curves(results, output_dir)
    
    # Compare test metrics
    print("\nComparing test metrics...")
    compare_test_metrics(results, output_dir)
    
    # Compare class metrics
    print("\nComparing class metrics...")
    compare_class_metrics(results, output_dir)
    
    # Compare ROC curves
    print("\nComparing ROC curves...")
    compare_roc_curves(results, output_dir)
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(results, output_dir)
    
    print(f"\nComparison analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()