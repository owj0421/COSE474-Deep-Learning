import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def plot_loss(best_stats):
    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(best_stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(best_stats['train_acc_history'], label='train')
    plt.plot(best_stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()

def plot_class_dist(data):
    dataset_info = pd.DataFrame(pd.Series(data).value_counts())
    dataset_info = dataset_info.reset_index().sort_values(by='index')
    dataset_info.plot.bar(x='index', y='count')


def plot_accuracy_by_class(y_true, y_pred):
    result_df = pd.DataFrame({'index': y_true, 'y_pred': y_pred})
    result_df['is_correct'] = (result_df['index'] == result_df['y_pred'])
    accuracy = pd.DataFrame(result_df.groupby('index').agg({'is_correct': sum})['is_correct'] / result_df['index'].value_counts(), columns=['accuracy']).reset_index()
    accuracy.plot.bar(x='index', y='accuracy')