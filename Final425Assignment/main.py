#Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
#based on a MATLAB code by James Hays and Sam Birch 

import numpy as np
from util import sample_images, build_vocabulary, get_bags_of_sifts, sub_sample_images
from classifiers import nearest_neighbor_classify, svm_classify
from sklearn.metrics import confusion_matrix

#For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

#For simplicity you can define a "num_train_per_cat" vairable, limiting the number of
#examples per category. num_train_per_cat = 100 for intance.

#Sample images from the training/testing dataset. 
#You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = sample_images("data/sift/train", n_sample=300)
test_image_paths, test_labels = sample_images("data/sift/test", n_sample=100)


''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

        
print('Extracting SIFT features\n')
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)

# A compact function for producing the histograms. Left it here for a record of work.
# Not intended to be elegant code.
def produce_15_hist():

    # These modules are imported locally as Matplotlib is a headache and takes time to import
    import os
    import matplotlib.pyplot as plt
    all_paths = [os.path.join("data/sift/train",i) for i in os.listdir("data/sift/train")]

    for pu in all_paths:
        train_imagepaths, train_labels = sub_sample_images(pu, n_sample=50)

        print(pu)
        label_pu = os.path.basename(pu)
        train_image_feats = get_bags_of_sifts(train_imagepaths, kmeans)

        sm = np.sum(train_image_feats,axis=0)

        plt.hist(sm,50)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.title(label_pu)
        plt.savefig(os.path.join(r"C:\Users\User\Desktop\figs",label_pu))
        plt.clf()

test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)
        
#If you want to avoid recomputing the features while debugging the
#classifiers, you can either 'save' and 'load' the extracted features
#to/from a file.
#
# ''' Step 2: Classify each test image by training and using the appropriate classifier
#  Each function to classify test features will return an N x l cell array,
#  where N is the number of test cases and each entry is a string indicating
#  the predicted one-hot vector for each test image. See the starter code for each function
#  for more details. '''
#
print('Using nearest neighbor classifier to predict test set categories\n')
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
print(pred_labels_knn)
print('compared to')
print(test_labels)
all = [1 for i in range(0,len(pred_labels_knn)) if (pred_labels_knn[i]==test_labels[i])]
print('score: ' + str(sum(all)/float(len(pred_labels_knn))))

#
#
print('Using support vector machine to predict test set categories\n')
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)
print(pred_labels_svm)
print('compared to')
print(test_labels)
all = [1 for i in range(0,len(pred_labels_svm)) if (pred_labels_svm[i]==test_labels[i])]
print('score: ' + str(sum(all)/float(len(pred_labels_svm))))

#
#
#
print('---Evaluation---\n')
# # Step 3: Build a confusion matrix and score the recognition system for
# #         each of the classifiers.

# Wrapped this in a function to avoid unintentional saving of figures
def confuse():
    # Build the confusion matricies for both classifiers
    cm_knn = confusion_matrix(test_labels, pred_labels_knn)
    cm_svm = confusion_matrix(test_labels, pred_labels_svm)

    import matplotlib.pyplot as plt


    plt.imshow(cm_knn)
    plt.colorbar()
    plt.title("kNN")
    plt.savefig(r"C:\Users\User\Desktop\knn.png")
    plt.clf()
    plt.imshow(cm_svm)
    plt.colorbar()
    plt.title("SVM")
    plt.savefig(r"C:\Users\User\Desktop\svm.png")

# # 1) Calculate the total accuracy of your model by counting number
# #   of true positives and true negatives over all.
# # 2) Build a Confusion matrix and visualize it.
# #   You will need to convert the one-hot format labels back
# #   to their category name format.
#
#
# # Interpreting your performance with 100 training examples per category:
# #  accuracy  =   0 -> Your code is broken (probably not the classifier's
# #                     fault! A classifier would have to be amazing to
# #                     perform this badly).
# #  accuracy ~= .10 -> Your performance is chance. Something is broken or
# #                     you ran the starter code unchanged.
# #  accuracy ~= .50 -> Rough performance with bag of SIFT and nearest
# #                     neighbor classifier. Can reach .60 with K-NN and
# #                     different distance metrics.
# #  accuracy ~= .60 -> You've gotten things roughly correct with bag of
# #                     SIFT and a linear SVM classifier.
# #  accuracy >= .70 -> You've also tuned your parameters well. E.g. number
# #                     of clusters, SVM regularization, number of patches
# #                     sampled when building vocabulary, size and step for
# #                     dense SIFT features.
# #  accuracy >= .80 -> You've added in spatial information somehow or you've
# #                     added additional, complementary image features. This
# #                     represents state of the art in Lazebnik et al 2006.
# #  accuracy >= .85 -> You've done extremely well. This is the state of the
# #                     art in the 2010 SUN database paper from fusing many
# #                     features. Don't trust this number unless you actually
# #                     measure many random splits.
# #  accuracy >= .90 -> You used modern deep features trained on much larger
# #                     image databases.
# #  accuracy >= .96 -> You can beat a human at this task. This isn't a
# #                     realistic number. Some accuracy calculation is broken
# #                     or your classifier is cheating and seeing the test
# #                     labels.