import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pickle 
def plot_multiclass_roc(y_true, y_scores, n_classes, save_path=None):
    """
    Plot ROC curve and calculate AUC for multi-class classification and save the plot.
    
    Parameters:
    - y_true: True labels (1D array of integer class labels)
    - y_scores: Predicted probabilities for each class (2D array, shape: [n_samples, n_classes])
    - n_classes: Number of classes in the classification problem
    - save_path: Path to save the ROC curve plot (optional)
    
    Returns:
    - Dictionary of AUC scores for each class
    - Macro-average ROC AUC
    """
    # Binarize the output (one-hot encode the labels)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    
    # Compute ROC curve and AUC for each class
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curve for each class
        plt.plot(fpr[i], tpr[i], color=color, 
                 label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
    
    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    
    # Compute macro-average ROC AUC
    macro_roc_auc = np.mean(list(roc_auc.values()))
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve\nMacro-average ROC AUC: {macro_roc_auc:.4f}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    # Print individual class AUC scores
    for cls, auc_score in roc_auc.items():
        print(f"Class {cls} AUC: {auc_score:.4f}")
    
    return roc_auc, macro_roc_auc

# Example usage
def main():
    file_path = "results/Classification/ImagenetModels--data=imagenet_3channel/evaluations/submodel=resnet18__predictions.pkl"
    n_classes = 3
    
    with open(file_path, 'rb') as file:
      data = pickle.load(file)
    # True labels
    y_true , y_scores = data["y_true"], data["y_hat"]    
    y_true, y_scores = np.concatenate(y_true), np.concatenate(y_scores)
    # Example save path (modify as needed)
    save_path = 'pics/common_multiclass_roc_curve.png'
    # Calculate and save ROC curves
    class_auc_scores, macro_auc = plot_multiclass_roc(
        y_true, y_scores, n_classes, save_path
    )
    
    print(f"Macro-average ROC AUC: {macro_auc:.4f}")

if __name__=="__main__":
  main()
