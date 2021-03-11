''' functions used for training and evaluating the trained models '''

# load libraries
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, recall_score, plot_roc_curve, roc_curve, auc, classification_report, precision_recall_curve, brier_score_loss, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import statistics
import numpy as np
from keras_model_lib import keras_models
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, CSVLogger, ModelCheckpoint
from tensorflow.keras.models import load_model
from num2words import num2words 
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from tensorflow.keras.metrics import SpecificityAtSensitivity, SensitivityAtSpecificity, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
import matplotlib.pyplot as plt

# load local code
from img_loading import stratified_shuffle, augment, normalize

# global variables for the learning rate scheduler
global total_epochs
global LR_Specs


def plot_curve(fpr, tpr, testing): 
    # refered to the sklearn library (https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py)

    # to plot five fold cross-validation ROC curve
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    colourArr = ["orange","blue","green","yellow","magenta"]
    labelArr = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    fpr_interp = np.linspace(0,1,20) # fpr is horizontal axis, tpr is vertical axis
    tpr_interp = []
    AUC_arr = []
    for i in range(len(fpr)):
        fpr_curr = fpr[i]
        tpr_curr = tpr[i]
        plt.plot(fpr_curr, tpr_curr, color=colourArr[i], label=labelArr[i], alpha=0.2)
        temp = np.interp(fpr_interp, fpr_curr, tpr_curr)
        temp[0] = 0.0
        tpr_interp.append(temp) # interpolate into the shape to have same dimensions for averaging & stdv
        AUC_arr.append(auc(fpr_interp, temp))

    tpr_avg = np.mean(tpr_interp, axis=0)
    tpr_std = np.std(tpr_interp, axis=0)
    tpr_ub = np.minimum(tpr_avg + tpr_std, 1)
    tpr_lb = np.maximum(tpr_avg - tpr_std, 0)
    AUC_mean = auc(fpr_interp,tpr_avg)
    STDV_mean = np.std(AUC_arr)


    plt.fill_between(fpr_interp, tpr_lb, tpr_ub, color = 'grey', alpha=.2, label = "Mean ± 1 STDV")
    plt.plot(fpr_interp, tpr_avg, color='black', label='Mean ROC (AUC = {0:5.3f} ± {1:5.3f})'.format(AUC_mean, STDV_mean), alpha=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve: ' + testing)
    plt.legend()

    fig = plt.gcf()
    fig.savefig(os.path.join(testing + '_ROC.jpg'))
    plt.close("all")
    return os.path.join(testing + '_ROC.jpg')


def plot_curve_scatter(fpr, tpr, testing): # closely followed sklearn library (https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py)
    ''' similar to the plot_curve function above, except this adds a scatter plot for better visualization of how thresholding will affect classification '''
    
    plt.plot([0, 1], [0, 1], color='r', linestyle='--')
    colourArr = ["orange","blue","green","yellow","magenta"]
    labelArr = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    fpr_interp = np.linspace(0,1,20) # fpr is horizontal axis, tpr is vertical axis
    tpr_interp = []
    AUC_arr = []

    for i in range(len(fpr)):
        fpr_curr = fpr[i]
        tpr_curr = tpr[i]
        plt.scatter(fpr_curr, tpr_curr, c=colourArr[i], label=labelArr[i], alpha=0.8)
        temp = np.interp(fpr_interp, fpr_curr, tpr_curr)
        temp[0] = 0.0
        tpr_interp.append(temp) # interpolate into the shape to have same dimensions for averaging & stdv
        AUC_arr.append(auc(fpr_interp, temp))

    tpr_avg = np.mean(tpr_interp, axis=0)
    tpr_std = np.std(tpr_interp, axis=0)
    tpr_ub = np.minimum(tpr_avg + tpr_std, 1)
    tpr_lb = np.maximum(tpr_avg - tpr_std, 0)
    AUC_mean = auc(fpr_interp,tpr_avg)
    STDV_mean = np.std(AUC_arr)

    plt.fill_between(fpr_interp, tpr_lb, tpr_ub, color = 'grey', alpha=.1, label = "Mean ± 1 STDV")
    plt.plot(fpr_interp, tpr_avg, color='black', label='Mean ROC (AUC = {0:5.3f} ± {1:5.3f})'.format(AUC_mean, STDV_mean), alpha=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve: ' + testing)
    plt.legend()
    # plt.show()
    fig = plt.gcf()
    fig.savefig(os.path.join(testing + '_ROC.jpg'))
    plt.close("all")
    return os.path.join(testing + '_ROC.jpg')


def lr_scheduler(epoch, lr): # triangular2 (halved every full cycle)
    ''' learning rate scheduler with 4 options
        - steady - constant learning rate (returns the max LR defined in LR_Specs constantly)
        - step_decay - decays with the number of steps defined in LR_Specs
        - tri1 - cyclic learning rate which cycles from max to min for the number of cycles defiend in LR_Specs
        - tri2 - cyclic learning rate which cycles like tri1 by decays exponentially after each cycle
    '''
    global total_epochs
    global LR_Specs

    # obtain values from LR_Specs
    # minimum learning rate
    lr_floor = LR_Specs[1]

    # maximum learning rate, step size, and cycle count (if cyclic)
    lr_max = LR_Specs[0]
    step_size = total_epochs/(LR_Specs[3]*2) # into 100/10 -> step size of 10 for cyclic learning rate (multiplied by two because each 'step' is defined half of a cycle)
    cycle_count = np.floor((epoch+0.0001)/step_size) # if odd, we increase to max

    if LR_Specs[2] == 'steady':
        # steady unchanging learning rate
        return lr_max
    if LR_Specs[2] == 'step_decay':
        # decaying constantly by step_size
        step_size = total_epochs/(LR_Specs[3]) # into 100/10 -> step size of 10
        delta_lr_per_step = (lr_max-lr_floor)/step_size
        return lr_max - (delta_lr_per_step * ((epoch) % step_size))
    if LR_Specs[2] == 'tri1':
        # cycle from max to min with no decay
        delta_lr_per_step = (lr_max-lr_floor)/step_size
        if ((cycle_count + 1) % 2) == 0: # Even
            outputLR = lr_floor + (delta_lr_per_step * ((epoch) % step_size))
        else: # Odd
            # decreasing from max to floor
            outputLR = lr_max - (delta_lr_per_step * ((epoch) % step_size))
        return outputLR 
    if LR_Specs[2] == 'tri2':
        # cycle from max to min with decay by half each cycle
        curr_lr_max  = lr_max * (0.5 ** np.floor(cycle_count//2))
        delta_lr_per_step = (curr_lr_max-lr_floor)/step_size
        if ((cycle_count + 1) % 2) == 0: # Even
            outputLR = lr_floor + (delta_lr_per_step * ((epoch) % step_size))
        else: # Odd
            # decreasing from max to floor
            outputLR = curr_lr_max - (delta_lr_per_step * ((epoch) % step_size))
        return outputLR 


def evaluate_classification_stratified(lbl_test, predicted_classes):
    ''' evaluates each label vs predicted class for accuracy and returns the accuracy.. can be easily changed to calculate other metrics '''
    # https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/
    lbl_test = (np.array(lbl_test) > 1)
    correct = (np.array(lbl_test) == np.array(predicted_classes))
    acc = correct.sum()/correct.size

    return acc


def testFold_stratified(img_test, lbl_test, model, class_num, validation_threshold):
    ''' evaluates the model on the testing data '''
    # use label to create a balanced and shuffled split
    unique_set = list(set(lbl_test))

    # initialize lists
    ioc_arr = []
    distribution_of_classes = []
    class_idx_in_list = []
    Acc_t = []
    class_lst = []
    pred_lst = []
    correct_count = []
    incorrect_count = []

    # loop through each unique value and find the indicies 
    for i, unique_val in enumerate(unique_set):
        # ioc are the indices of occurences
        ioc = np.where(lbl_test == unique_val)[0]

        # make prediction on the test data using the best model
        predictions = model.predict(
            img_test[ioc],
            batch_size=1,
            verbose=1)

        if class_num > 1:
            predictions = predictions[:,1]
            lbl_test = np.squeeze(lbl_test[:,1])

        predicted_classes = np.squeeze(predictions > validation_threshold)

        # error when predicted_Classes is of length 1, there will no list. turn it back to list to stay
        if len(lbl_test[ioc]) == 1:
            predicted_classes = [predicted_classes]

        # calling a function used to evaluate the performance
        acc = evaluate_classification_stratified(lbl_test[ioc], predicted_classes)

        # epsilon value to avoid division by zero for PPV and NPV (if it never guesses either positive or negative [i.e. greatly overfitting or unbalanced dataset])
        epsilon = 0.000001

        # calculate evaluation metrics (Accuracy, Sepcificity, Sensitivity, PPV, NPV)
        Acc_t.append(acc)
        class_lst.append(unique_val)
        pred_lst.append(predictions)
        correct_count.append('N/A')
        incorrect_count.append('N/A')

    return Acc_t, class_lst, pred_lst, correct_count, incorrect_count


def evaluate_classification(lbl_test, predicted_classes, class_list):
    ''' evaluates the model on the testing data and returns the following: FN, FP, TN, TP, specAtSens_val, sensAtSpec_val, classification_matrix, bal_acc '''
    # https://www.tensorflow.org/versions/r2.3/api_docs/python/tf/keras/metrics/
    # binary case ONLY
    TN, FP, FN, TP = confusion_matrix(lbl_test, predicted_classes).ravel()

    print('class_metrics:', TN, FP, TN, TP)

    # obtain FN, FP, TN, TP using tf.keras.metrics
    FN = FalseNegatives()
    FN.update_state(lbl_test, predicted_classes)
    FN = FN.result().numpy()

    FP = FalsePositives()
    FP.update_state(lbl_test, predicted_classes)
    FP = FP.result().numpy()

    TN = TrueNegatives()
    TN.update_state(lbl_test, predicted_classes)
    TN = TN.result().numpy()

    TP = TruePositives()
    TP.update_state(lbl_test, predicted_classes)
    TP = TP.result().numpy()

    # calculate specificity at sensitivity and sensitivity at specificity (hold at 80%)
    specAtSens_val = SpecificityAtSensitivity(0.80)
    specAtSens_val.update_state(lbl_test, predicted_classes)
    specAtSens_val = specAtSens_val.result().numpy()

    sensAtSpec_val = SensitivityAtSpecificity(0.80)
    sensAtSpec_val.update_state(lbl_test, predicted_classes)
    sensAtSpec_val = sensAtSpec_val.result().numpy()

    # obtain the classification matrix
    classification_matrix = classification_report(lbl_test, predicted_classes, target_names=class_list)

    # calculate balanced accuracy
    bal_acc = balanced_accuracy_score(lbl_test, predicted_classes)

    return FN, FP, TN, TP, specAtSens_val, sensAtSpec_val, classification_matrix, bal_acc


def testFold(img_test, lbl_test, img_val, lbl_val, threshold_val, model, class_num, class_list):
    # calculate ROC values
    '''https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/'''

    # make prediction on the test data using the best model for the validation set to acquire the best threshold 
    if threshold_val is None:
        predictions_val = model.predict(
            img_val,
            batch_size=1,
            verbose=1)

        if class_num > 1:
            predictions_val = np.squeeze(predictions_val[:,1])
        fpr_v, tpr_v, thresholds_v = roc_curve(lbl_val, predictions_val)

        # gmean is maximized to find the largest G-Mean value
        gmeans = np.sqrt(tpr_v * (1-fpr_v))
        ix = np.argmax(gmeans)
        validation_threshold = thresholds_v[ix]

    else:
        # otherwise, use the value that was read in from the model name
        validation_threshold = threshold_val

    # These lines are required for the code to run successfully on TF 2.1
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # make prediction on the test data using the best model
    predictions = model.predict(
        img_test,
        batch_size=1,
        verbose=1)

    if class_num > 1:
        predictions = np.squeeze(predictions[:,1])
        lbl_test = np.squeeze(lbl_test[:,1])

    # predicted_classes = np.squeeze(predictions > 0.5)
    predicted_classes = np.squeeze(predictions > validation_threshold)

    # calling a function used to evaluate the performance
    FN, FP, TN, TP, specAtSens_val, sensAtSpec_val, classification_matrix, bal_acc = evaluate_classification(lbl_test, predicted_classes, class_list)

    # epsilon value to avoid division by zero for PPV and NPV (if it never guesses either positive or negative [i.e. greatly overfitting or unbalanced dataset])
    epsilon = 0.000001

    # calculate evaluation metrics (Accuracy, Sepcificity, Sensitivity, PPV, NPV)
    Acc_t, Sens_t, Spec_t = (TN+TP)/(TN+TP+FN+FP), TP/(TP + FN), TN/(TN + FP)
    PPV_t, NPV_t = np.round(TP/(TP + FP + epsilon),6), np.round(TN/(FN + TN + epsilon),6)

    F1_score = TP/(TP+(0.5*(FP+FN)))

    # calculate ROC values
    '''https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/'''
    fpr, tpr, _ = roc_curve(lbl_test, predictions)
    roc_auc = auc(fpr, tpr)

    # calculate precision recall
    precision, recall, _ = precision_recall_curve(lbl_test, predictions)
    pr_auc = auc(recall, precision)

    # brier score 
    brier_score = brier_score_loss(lbl_test, predictions)

    # average precision and recall
    avg_precision = average_precision_score(lbl_test, predictions)
    avg_recall = recall_score(lbl_test, predicted_classes, average='weighted')

    return Acc_t, Sens_t, Spec_t, tpr, fpr, roc_auc, recall, precision, pr_auc, PPV_t, NPV_t, specAtSens_val, sensAtSpec_val, F1_score, bal_acc, brier_score, validation_threshold, avg_precision, avg_recall, lbl_test, predictions

