import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from main01 import Experiment01
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns


class Vis:




    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        X_train, X_test, y_train, y_test = Experiment01.load_data()
        #support_vectors =  Experiment01.run()

        #pca = PCA(n_components=1 )
        #X_train = pca.fit_transform(X_train)
        #X_test = pca.transform(X_test)

        print(X_train.shape)
        #print(support_vectors.shape)
        #print(X_train)

        X_train = X_train.toarray()
        print(X_train[:3])


        plt.plot(X_train[:3])
        #plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red')
        #plt.title('Linearly separable data with support vectors')
        #plt.xlabel('X1')
        #plt.ylabel('X2')
        plt.show()





    @staticmethod
    def train_load_data(file_path_prefix="/Users/koradumpert/Desktop/DSCI372/project_4_ML/"):

        """
        Load the data and partition it into testing and training data.
        :param filename: The location of the data to load from file.
        :return: train_X, train_y, test_X, test_y; each as an iterable object
        (like a list or a numpy array).
        """

        df = pd.read_csv(file_path_prefix + "cleaned_FNews_data.csv", header=0)
        df = df.sample(n=150, random_state=42) #make it smaller so runs quicker

        from sklearn.model_selection import train_test_split
        text = df['text']
        labels = df['label']

        vect = CountVectorizer() #encodes text to vectors
        X = vect.fit_transform(text)
        y = labels








        return X, y





if __name__ == "__main__":
    # Run experiment

    Vis.run()


