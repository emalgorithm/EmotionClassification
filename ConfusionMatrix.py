import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    def __init__(self):
        self.matrix = []
        self.index_map = {}
        self.classes = []

    def create_matrix(self, expected, predicted, classes):
        self.classes = classes

        matrix = [[0] * len(classes) for i in range(len(classes))]
        self.index_map = {c: i for (i, c) in enumerate(classes)}

        for pred, exp in zip(predicted, expected):
            matrix[self.index_map[pred]][self.index_map[exp]] += 1

        self.matrix = np.array(matrix).T

        return self

    def format_float(self, number):
        return round(number, 4)

    def calc_accuracy(self):
        total = sum(sum(l) for l in self.matrix)

        accuracy = sum(self.matrix[i][i] for i in range(len(self.matrix))) / total

        return self.format_float(accuracy)

    def calc_precision_for_classes(self):
        precisions = {c: self.calc_class_precision(c) for c in self.classes}
        return precisions

    def calc_class_precision(self, c):
        precision = 0.0

        class_index = self.index_map[c]
        no_correct = self.matrix[class_index][class_index]

        # class column sum
        row_sum = np.sum(np.array(self.matrix), axis=0)[class_index]

        if row_sum > 0:
            precision = float(no_correct) / float(row_sum)

        return self.format_float(precision)

    def calc_recall_for_classes(self):
        recall = {c: self.get_recall_for_class(c) for c in self.classes}

        return recall

    def get_recall_for_class(self, c):
        recall = 0.0

        class_index = self.index_map[c]
        no_correct = self.matrix[class_index][class_index]

        # class row sum
        row_sum = np.sum(np.array(self.matrix), axis=1)[class_index]

        if row_sum > 0:
            recall = float(no_correct) / float(row_sum)

        return self.format_float(recall)

    def calc_f1_score(self):
        f_measure = {}

        precision_for_classes = self.calc_precision_for_classes()
        recall_for_classes = self.calc_recall_for_classes()

        for c in self.classes:
            p = precision_for_classes[c]
            r = recall_for_classes[c]

            f_score = 0.0

            if (p + r) > 0:
                f_score = (2 * (p * r)) / (p + r)

            f_measure[c] = self.format_float(f_score)

        return f_measure

    def plot_confusion_matrix(self, normalize=False title='Confusion matrix'):
        fig, ax  = plt.subplots()
        ax.set_xticks([x for x in range(len(self.classes))])
        ax.set_yticks([y for y in range(len(self.classes))])

        # Place classes on minor ticks
        ax.set_xticks([x + 0.5 for x in range(len(self.classes))], minor=True)
        ax.set_xticklabels(self.classes, rotation='90', fontsize=10, minor=True)
        ax.set_yticks([y + 0.5 for y in range(len(self.classes))], minor=True)
        ax.set_yticklabels(self.classes[::-1], fontsize=10, minor=True)

        ax.tick_params(which='major', labelbottom='off', labelleft='off')
        ax.tick_params(which='minor', width=0)

        # Plot heat map
        proportions = [row / np.sum(row) for row in np.array(self.matrix)]
        ax.pcolor(np.array(proportions[::-1]), cmap=plt.cm.Blues)

        # row normalize
        row_sums = self.matrix.sum(axis=1)
        row_normalied_matrix = self.matrix / row_sums[:, np.newaxis]

        # Plot counts as text
        for row in range(len(self.matrix)):
            for col in range(len(self.matrix[row])):
                if (normalize):
                    confusion = round(row_normalied_matrix[::-1][row][col], 2)
                else:
                    confusion = self.matrix[::-1][row][col]
                # if confusion != 0:
                ax.text(col + 0.5, row + 0.5, confusion, fontsize=9,
                    horizontalalignment='center',
                    verticalalignment='center')

        ax.grid(True, linestyle=':')
        ax.set_title(title)
        fig.tight_layout()
        plt.show()


## EXAMPLE ##
# X_train, X_test = X_clean[train_index], X_clean[test_index]
# y_train, y_test = y_clean[train_index], y_clean[test_index]
#
# predicted = [predict(trees, x) for x in X_test]
#
## create confusion matrix with parameters (actual_y, predicted_y, classes)
# conf_matrix = ConfusionMatrix().create_matrix(y_test, predicted, list(set(y_test)))
#
# accuracy = conf_matrix.calc_accuracy()
# print("{}% accuracy".format(accuracy))
#
# print("Precision for each class")
# print(conf_matrix.calc_precision_for_classes())
#
# print("Recall for each class")
# print(conf_matrix.calc_recall_for_classes())
#
# print("F1 score for each class")
# print(conf_matrix.calc_f1_score())
#
# # Plot confusion matrix
# conf_matrix.plot_confusion_matrix()

