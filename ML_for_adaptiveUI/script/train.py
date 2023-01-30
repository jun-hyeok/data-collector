from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Conv1D
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier  # Ridge regression
from sklearn.linear_model import LogisticRegression  # Logistic regression
from sklearn.linear_model import SGDClassifier  # Stochastic gradient descent
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)  # Linear discriminant analysis
from sklearn.neighbors import KNeighborsClassifier  # K-nearest neighbors
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.tree import DecisionTreeClassifier  # Decision tree
from sklearn.ensemble import RandomForestClassifier  # Random forest
from sklearn.ensemble import GradientBoostingClassifier  # Gradient boosting
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost
from sklearn.svm import SVC  # Support vector machine
from sklearn.neural_network import MLPClassifier  # Multi-layer perceptron
from sklearn.gaussian_process import GaussianProcessClassifier  # Gaussian process
from sklearn.model_selection import cross_val_score
import eli5
from eli5.sklearn import PermutationImportance
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

train_pkl = 'Data\\data_merged2.pkl'
test_pkl = 'Data\\data_merged_test.pkl'

models = [
    RidgeClassifier(alpha=1.0, solver="auto", random_state=42),
    LogisticRegression(C=1.0, solver="lbfgs", multi_class="auto", random_state=42),
    # SGDClassifier(loss="hinge", penalty="l2", alpha=0.001, random_state=42),
    # LinearDiscriminantAnalysis(solver="svd", tol=0.0001),
    # KNeighborsClassifier(n_neighbors=5, weights="uniform", leaf_size=30),
    # GaussianNB(priors=None, var_smoothing=1e-09),
    # DecisionTreeClassifier(
    #     criterion="gini",
    #     max_depth=None,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     random_state=42,
    #     max_leaf_nodes=None,
    # ),
    RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        max_leaf_nodes=None,
    ),
    # GradientBoostingClassifier(
    #     loss="deviance",
    #     learning_rate=0.1,
    #     n_estimators=100,
    #     subsample=1.0,
    #     criterion="friedman_mse",
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_depth=3,
    #     random_state=42,
    #     max_leaf_nodes=None,
    # ),
    # AdaBoostClassifier(
    #     n_estimators=50,
    #     learning_rate=1.0,
    #     algorithm="SAMME.R",
    #     random_state=42,
    # ),
    SVC(C=1.0, kernel="rbf", random_state=42),
    # MLPClassifier(
    #     hidden_layer_sizes=(100,),
    #     activation="relu",
    #     solver="adam",
    #     alpha=0.0001,
    #     max_iter=200,
    #     shuffle=True,
    #     random_state=42,
    # ),
    # GaussianProcessClassifier(random_state=42),
]

def dataload(fun='nn', mode='train', filepath = 'C:\\Users\\uvrlab\\Downloads\\Data for training and validation\\Training\\data_merged(pid1to6).pkl'):
    file = filepath
    df = pd.read_pickle(file)
    y_ = df['condition'].to_numpy(dtype=np.int32)
    arousal_ = df["arousal"].to_numpy(dtype=np.int32)
    valence_ = df["valence"].to_numpy(dtype=np.int32)
    df.drop(["pid", "condition", "acc.x", "acc.y", "acc.z", "arousal", "valence"], axis=1, inplace=True)
    # df.drop(["pid", "condition", "acc.x", "acc.y", "acc.z", "arousal", "valence", 'AccelX', 'AccelY', 'AccelZ', 'Face', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Posture', 'desktop', 'laptop', 'mfcc1', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc2', 'mfcc20', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'phone'], axis=1, inplace=True)
    col = df.columns.tolist()
    X = df.to_numpy(dtype=np.float32)    
    X = StandardScaler().fit_transform(X)
    y_ -= 1
    n_classes = 6

    if fun == 'nn':
        y = np_utils.to_categorical(y_, num_classes=n_classes)
    elif fun == 'arousal':
        y = np_utils.to_categorical(arousal_, num_classes=3)
    elif fun == 'valence':
        y = np_utils.to_categorical(valence_, num_classes=3)
    elif fun == 'arousal_ml':
        y = arousal_
    elif fun == 'valence_ml':
        y = valence_
    else:
        y = y_

    if mode == 'train':
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=True, random_state=77, test_size=0.2)
        return X_train, X_valid, y_train, y_valid, col
    if mode == 'test':
        return X, y, col

def build_1dnn(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(30, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def y_to_multi_binary(y):
    y_social = np.where(y < 3, 0, 1)
    y_social = np.expand_dims(y_social, axis = 1)
    y_mobile = np.where((y+1) % 3 == 0, 1, 0)
    y_mobile = np.expand_dims(y_mobile, axis = 1)
    y_work = np.where(y % 3 == 0, 1, 0)
    y_work = np.expand_dims(y_work, axis = 1)
    print(y_social.shape)

    return [y_social, y_mobile, y_work]

def binary_model(n_inputs):
    model = Sequential()
    model.add(Dense(50, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_binary():
    X_train, X_valid, y_train, y_valid, col = dataload(fun='binary', mode='train', filepath=train_pkl)
    y_train_list = y_to_multi_binary(y_train)
    y_valid_list = y_to_multi_binary(y_valid)
    for idx, y_condition in enumerate(y_train_list):
        print("%d model!"%idx)
        model = binary_model(X_train.shape[1])
        history = model.fit(X_train, y_condition, verbose=1, epochs=100)
        model.save('binary_model_1128_%d.h5'%idx)
        plt.plot(history.history['loss'])
        plt.title('Model %d loss'%idx)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("loss_%d_1128.png"%idx)
        plt.close()
        yhat = model.predict(X_valid)
        yhat = yhat.round()
        acc = accuracy_score(y_valid_list[idx], yhat)
        print('%d model acc> %3f'%(idx, acc))

def test_binary():
    X_test, y_test, col = dataload(fun='binary', mode='test', filepath=test_pkl)
    y_test_list = y_to_multi_binary(y_test)
    for idx, y_condition in enumerate(y_test_list):
        print("%d model!"%idx)
        model = binary_model(X_test.shape[1])
        model = load_model('binary_model_%d.h5'%idx)
        # layer_outputs = [layer.output for layer in model.layers]
        layer = model.layers[0]
        weight, bias = layer.get_weights()
        weight = np.sum(weight, axis=1)
        # plt.pcolor(weight)
        print(weight.shape)
        df =pd.DataFrame(weight)
        df.to_csv('feature_device_%d.csv'%(idx))
        ind = np.argpartition(weight, -5)[-5:][::-1]
        print(ind)
        plt.bar(range(38), weight)
        plt.savefig('layers\\feature_device_%d.png'%(idx))
        plt.close()
        for i, co in enumerate(col):
            print(i, co)
        # extract_model = Model(inputs=model.input, outputs=layer_outputs)
        # extract_features_by_layer = extract_model.predict(np.expand_dims(X_test[0], axis=0))
        # for layer_num, layer in enumerate(extract_features_by_layer):
        #     weigts, bias = layer.get_weights()
        #     print(weigts.shape)
        #     plt.pcolor(layer)
        #     # plt.imshow(layer)
        #     plt.savefig('layers\\feature_%d_%d.png'%(idx, layer_num))
        #     plt.close

        results = model.evaluate(X_test, y_condition, verbose=1)
        yhat = model.predict(X_test)
        yhat = yhat.round()
        acc = accuracy_score(y_condition, yhat)
        print('%d model acc> %3f'%(idx, acc))
        cm = confusion_matrix(y_condition, yhat, normalize='true')
        plt.matshow(cm)
        plt.savefig('test_confusion_matrix_%d.png'%(idx))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('norm_test_confusion_matrix_%d.png'%(idx))


def train():
    X_train, X_valid, y_train, y_valid, col = dataload(fun = 'nn',mode='train', filepath=train_pkl)
    X_test, y_test, col = dataload(fun = 'nn', mode='test', filepath=train_pkl)
    print(X_train.shape)
    model = build_1dnn(X_train.shape[1], y_train.shape[1])
    # model = load_model('train_model.h5')
    history = model.fit(X_train, y_train, verbose=1, epochs=100)
    yhat = model.predict(X_valid)
    yhat = yhat.round()
    acc = accuracy_score(y_valid, yhat)
    model.save('.\\model\\train_model_1128_nn.h5')
    print('test >%3f'%acc)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results_kfold = cross_val_score(model, X_test, y_test, cv=kfold)
    print(results_kfold)

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(".\\IMG\\loss_1128_nn.png")
    plt.close()

    labels = ["c1", "c2", "c3", "c4", "c5", "c6"]
    cm = confusion_matrix(y_valid.argmax(axis=1), yhat.argmax(axis=1))
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('.\\IMG\\cm_train_1128_nn.png')
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('.\\IMG\\norm_cm_train_1128_nn.png')

def test():
    X_test, y_test = dataload(mode='test', filepath=train_pkl)
    model = build_1dnn(X_test.shape[1], y_test.shape[1])
    model = load_model('.\\model\\train_model_1128_nn.h5')
    results = model.evaluate(X_test, y_test, verbose=1)  
    yhat = model.predict(X_test)
    yhat = yhat.round()
    acc = accuracy_score(y_test, yhat)
    print('loss> %3f'%results)
    print('acc>%3f'%acc)
    # print('>loss : %3f, >acc'%(results[0], results[1])) 

    labels = ["c1", "c2", "c3", "c4", "c5", "c6"]
    cm = confusion_matrix(y_test.argmax(axis=1), yhat.argmax(axis=1))
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('.\\IMG\\cm_test_1128_nn.png')
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1,2], yticklabels=[0,1,2])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('.\\IMG\\norm_cm_test_1128_nn.png')

def train_and_test(model):
    print(model.__class__.__name__)
    # train the model
    X_train, X_valid, y_train, y_valid, col = dataload(fun='valence_ml', mode='train', filepath=train_pkl)
    model.fit(X_train, y_train)
    # # get importance
    # if model.__class__ in [
    #     DecisionTreeClassifier,
    #     RandomForestClassifier,
    #     GradientBoostingClassifier,
    #     AdaBoostClassifier,
    # ]:
    #     importances = model.feature_importances_
    #     importance = model.feature_importances_
    #     # summarize feature importance
    #     for i, v in enumerate(importance):
    #         print("  Feature: %0d, Score: %.5f" % (i, v))
    #     # plot feature importance
    #     plt.bar([x for x in range(len(importance))], importance)
    #     plt.show()
    # make predictions
    y_pred = model.predict(X_valid)
    # evaluate predictions
    print("Train  Accuracy: ", accuracy_score(y_valid, y_pred))

    X_test, y_test, col = dataload(fun='valence_ml', mode='test', filepath=train_pkl)
    if model.__class__.__name__ == "RandomForestClassifier":
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        results = cross_val_score(model, X_test, y_test, cv=kfold)  
        print(results)
        importance = model.feature_importances_
        df = pd.DataFrame(importance)
        df.to_csv("rf_importance_result_valence.csv")
        # summarize feature importance
        # for i,v in enumerate(importance):
        #     print('Feature: %0d, Score: %.5f' % (i,v))
        # plot feature importance
        plt.bar([x for x in range(len(importance))], importance)
        plt.savefig(".\\IMG\\importance_result_%s_valence.png"%(model.__class__.__name__))
        plt.close()
        cm = confusion_matrix(y_valid, y_pred)
        plt.matshow(cm)
        plt.colorbar()
        plt.savefig('test_confusion_matrix_result_%s_valence.png'%(model.__class__.__name__))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('norm_test_confusion_matrix_result_%s_valence.png'%(model.__class__.__name__))
    # # results = model.evaluate(X_test, y_test)
    # yhat = model.predict(X_test)
    # acc = accuracy_score(y_test, yhat)
    # # print('loss> %3f'%results)
    # print('TEST acc>%3f'%acc)
    # # evaluate predictions
    # cm = confusion_matrix(y_test, yhat)
    # plt.matshow(cm)
    # plt.colorbar()
    # plt.savefig('test_confusion_matrix_1128_arousal_%s.png'%model.__class__.__name__)
    # print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Classification report:\n", classification_report(y_test, y_pred))

def test_ml(model):
    print(model.__class__.__name__)
    # train the model
    X_test, y_test = dataload(fun='svm', mode='test', filepath=test_pkl)
    model = load_model('train_%s.h5'%model.__class__.__name__)
    results = model.evaluate(X_test, y_test)
    yhat = model.predict(X_test)
    acc = accuracy_score(y_test, yhat)
    print('loss> %3f'%results)
    print('acc>%3f'%acc)
    y_pred = model.predict(X_test)
    # evaluate predictions
    print("  Accuracy: ", accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, yhat)
    plt.matshow(cm)
    plt.colorbar()
    plt.savefig('test_confusion_matrix_%s.png'%model.__class__.__name_)
    # print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Classification report:\n", classification_report(y_test, y_pred))

def train_and_test_binary(model):
    print(model.__class__.__name__)
    # train the model
    X_train, X_valid, y_train, y_valid, col = dataload(fun='ml', mode='train', filepath=train_pkl)
    print(col)
    y_train_list = y_to_multi_binary(y_train)
    y_valid_list = y_to_multi_binary(y_valid)
    X_test, y_test, col = dataload(fun='ml', mode='test', filepath=train_pkl)
    y_test_list = y_to_multi_binary(y_test)
    for idx, y_condition, y_condition_test in zip(range(0, X_train.shape[0]),y_train_list, y_test_list):
        print("%d model!"%idx)
        model.fit(X_train, y_condition)
        yhat = model.predict(X_valid)
        yhat = yhat.round()
        acc = accuracy_score(y_valid_list[idx], yhat)
        print('[train]%d model acc> %3f'%(idx, acc))

        # yhat = model.predict(X_test)
        # yhat = yhat.round()
        # acc = accuracy_score(y_condition_test, yhat)
        # print('[test]%d model test acc> %3f'%(idx, acc))
        if model.__class__.__name__ == "RandomForestClassifier":
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            results = cross_val_score(model, X_test, y_test, cv=kfold)  
            print(results)
            importance = model.feature_importances_
            df = pd.DataFrame(importance)
            df.to_csv("rf_importance_results_%d.csv"%idx)
            # summarize feature importance
            # for i,v in enumerate(importance):
            #     print('Feature: %0d, Score: %.5f' % (i,v))
            # plot feature importance
            plt.bar([x for x in range(len(importance))], importance)
            plt.savefig(".\\IMG\\importance_results_%s_%d.png"%(model.__class__.__name__, idx))
            plt.close()
            cm = confusion_matrix(y_valid_list[idx], yhat)
            plt.matshow(cm)
            plt.colorbar()
            plt.savefig('test_confusion_matrix_results_%s_%d.png'%(model.__class__.__name__, idx))
            cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots(figsize=(10,10))
            sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[0,1], yticklabels=[0,1])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig('norm_test_confusion_matrix_results_%s_%d.png'%(model.__class__.__name__, idx))



if __name__ == '__main__':
    for model in models:
        train_and_test(model)