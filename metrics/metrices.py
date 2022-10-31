import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.text import Text

def log_table(rich_table):
    """Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    reference: https://github.com/Textualize/rich/discussions/1799
    """
    console = Console()
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())

def plot_confusion(actual, pred, class_name, save_path, save=True):

    actual = np.array(actual)
    pred = np.array(pred)

    # generate confusin matrix
    num_class = len(class_name)
    cm = np.zeros((num_class, num_class),dtype=int)
    for p,a in zip(pred, actual):
        cm[p][a] += 1

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    ax.matshow(cm_normalized, cmap=plt.cm.Blues)
    
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(x=i, y=j, s=f"{cm_normalized[i,j]:.2f}", va='center', ha='center', size='large')

    ticks = [i for i in range(num_class)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_name, ha='right')

    ax.set_yticks(ticks)
    ax.set_yticklabels(class_name, rotation=45,ha='right')

    plt.ylabel('Predictions', fontsize=12)
    plt.xlabel('Actuals', fontsize=12)

    if save:
        plt.savefig(save_path)
    
    return cm

def classification_report(cm, labels):
    
    Pre, Rc, F1 = [],[],[]
    # calculate tp,tn,fp,fn
    total = np.sum(cm)
    accuracy = np.sum(np.diag(cm)) / total

    for i in range(cm.shape[0]):
        fp,fn,tp = 0,0,0
        for j in range(cm.shape[1]):
            if i==j:
                tp = cm[i][j]
            else:
                fp += cm[i][j]
                fn += cm[j][i]

        tn = total-(tp+fp+fn)
        precision = round(tp/(tp+tn), 3)
        recall = round(tp/(tp+fn),3)
        f1 = round(2*((precision* recall)/(precision+recall)),3)

        Pre.append(precision)
        Rc.append(recall)
        F1.append(f1)
    
    # print results
    table = Table(title="Classification report")
    table.add_column("Class", style="magenta")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="magenta")
    table.add_column("F1-Score", style="magenta")

    for i in range(len(labels)):
        table.add_row(f"{labels[i]}",f"{Pre[i]:.2f}", f"{Rc[i]:.2f}", f"{F1[0]:.2f}")
    table.add_row("Accuracy", f"{accuracy}", style="red")

    return(log_table(table))

