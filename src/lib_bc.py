from numpy import ndarray
import numpy as np
from numba import njit
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


def printf(f, *args):
    print(f % args, end='')


@njit()
def score_map(score: ndarray, threshold: float) -> ndarray:
    return (score > threshold).astype(np.int8)


@njit()
def optimal_cutoff(y_true: ndarray, pred_score: ndarray) -> float:
    threshold = np.arange(0, 1, 0.01)
    KS_list = np.empty(len(threshold))
    for j, i in enumerate(threshold):
        pred_result = score_map(pred_score, i)
        TP = np.count_nonzero((pred_result == 1) & (y_true == 1))
        FP = np.count_nonzero((pred_result == 1) & (y_true == 0))
        bad = (y_true == 1).sum()
        good = (y_true == 0).sum()
        tpr = round(TP / bad, 3)
        fpr = round(FP / good, 3)
        KS = abs(tpr - fpr)
        KS_list[j] = KS

    # max_KS_ind = KS_list.index(max(KS_list))
    max_KS_ind = np.argmax(KS_list)
    optimal_threshold = threshold[max_KS_ind]

    # printf("Optimal threshold that maximizes KS to %.4f is: threshold = %.4f\n", max(KS_list), optimal_threshold)

    return optimal_threshold


@njit()
def metric_f1(y_true: ndarray, pred_result: ndarray) -> float:
    tp = np.count_nonzero((pred_result == 1) & (y_true == 1))
    fn = np.count_nonzero((pred_result == 0) & (y_true == 1))
    fp = np.count_nonzero((pred_result == 1) & (y_true == 0))
    f1_value = 2 * tp / (2 * tp + fp + fn)
    return f1_value


@njit()
def f1_cutoff(y_true: ndarray, pred_score: ndarray) -> float:
    threshold = np.arange(0.1, 0.9, 0.01)
    f1_list = np.empty(len(threshold))
    for j, i in enumerate(threshold):
        pred_result = score_map(pred_score, i)
        f1_list[j] = metric_f1(y_true, pred_result)
    max_KS_ind = np.argmax(f1_list)
    optimal_threshold = threshold[max_KS_ind]
    return optimal_threshold


@njit()
def acc_cutoff(y_true: ndarray, pred_score: ndarray) -> float:
    threshold = np.arange(0, 1, 0.01)
    f1_list = np.empty(len(threshold))
    for j, i in enumerate(threshold):
        pred_result = score_map(pred_score, i)
        f1_list[j] = (y_true == pred_result).mean()
    max_KS_ind = np.argmax(f1_list)
    optimal_threshold = threshold[max_KS_ind]
    return optimal_threshold


def plot_roc(test_y_true: ndarray, test_y_proba: ndarray, name: str) -> None:
    fpr, tpr, _ = roc_curve(test_y_true, test_y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    pname = "Receiver operating characteristic of " + name
    plt.title(pname)
    plt.legend(loc="lower right")
    plt.show()


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See https://matplotlib.org/examples/color/colormaps_reference.html

    title:         Title for the heatmap. Default is None.
    """

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1 = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar, xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)
    plt.show()
    
def plot_fi_xgb(model, imp_type, model_nm):
  xgb_fi_dict = model.get_score(importance_type=imp_type)
  fi_df = pd.DataFrame(xgb_fi_dict.items(), columns=[
                       'feature_names', 'feature_importance'])
  fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

  plt.figure(figsize=(10, 8))
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  plt.title(model_nm + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')

def plot_feature_importance(imp, feat, model):

  feature_importance = np.array(imp)
  feature_names = np.array(feat)

  fi_dict = {'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(fi_dict)

  fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

  plt.figure(figsize=(10,8))
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

  plt.title(model + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')
