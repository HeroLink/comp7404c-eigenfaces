import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def plot_gallery(images, titles, n_row=3, n_col=4):
    """ Helper function to plot a gallery of portraits """
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def report(y_true, y_pred, target_names):
    """ Print classification report and plot confusion matrix """
    print(classification_report(y_true, y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                            display_labels=target_names,
                                            xticks_rotation='vertical')
    plt.tight_layout()
    plt.show()