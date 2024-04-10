import os
import csv
import random
from sklearn.model_selection import train_test_split

Labeled_fraction = 0.1

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def get_classes(data):
    """Extract the classes from our data. If we want to limit them,
    only return the classes we need."""
    classes = []
    for item in data:
        if item[1] not in classes:
            classes.append(item[1])
    # Sort them.
    classes = sorted(classes)
    return classes

def choose_data(data, labeled_fraction):
    unlabeled, labeled  = train_test_split(data, test_size=labeled_fraction, random_state=42)
    return labeled, unlabeled


def split_data(data, labeled_fraction):
    labeled_data = []
    unlabeled_data = []
    start_idx = 0
    while start_idx < len(data):
        current_class = data[start_idx][1]
        end_idx = start_idx + 1
        while end_idx < len(data) and data[end_idx][1] == current_class:
            end_idx+=1

        one_class_data = data[start_idx:end_idx]
        print(current_class,' from ', start_idx,' to ', end_idx)
        labeled, unlabeled = choose_data(one_class_data, labeled_fraction)
        labeled_data = labeled_data + labeled
        unlabeled_data = unlabeled_data + unlabeled
        start_idx = end_idx + 1
    return labeled_data, unlabeled_data


train_data = get_data('train.csv')
classes_list = get_classes(train_data)

print("Num classes: ", len(classes_list))

labeled_data, unlabeled_data = split_data(train_data, Labeled_fraction)

file_name = "train_labeled_" + str(int(Labeled_fraction * 100)) + "percent.csv"
with open(file_name, "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(labeled_data)

file_name = "train_unlabeled_" + str(int(Labeled_fraction * 100)) + "percent.csv"
with open(file_name, "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(unlabeled_data)





