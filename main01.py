import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC




class Experiment01:




    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        X_train, X_test, y_train, y_test = Experiment01.load_data()

        # Train classifer
        classify = SVC(kernel='linear')
        classify.fit(X_train, y_train)


        score = classify.score(X_test, y_test)
        print(f"The accuracy of this SVM in labeling news as Fake or Real was: {score}")



    @staticmethod
    def load_data(file_path_prefix="/Users/koradumpert/Desktop/DSCI372/project_4_ML/"):

        """
        Load the data and partition it into testing and training data.
        :param filename: The location of the data to load from file.
        :return: train_X, train_y, test_X, test_y; each as an iterable object
        (like a list or a numpy array).
        """



        df = pd.read_csv(file_path_prefix + "cleaned_FNews_data.csv", header=0)
        df = df.sample(n=15000, random_state=42) #make it smaller so runs quicker (OG set was 70,000 +)

        from sklearn.model_selection import train_test_split
        text = df['text']
        labels = df['label']

        vect = CountVectorizer() #encodes text to vectors to allow for features
        X = vect.fit_transform(text)

        # Splits data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)




        return X_train, X_test, y_train, y_test




if __name__ == "__main__":
    # Run experiment

    Experiment01.run()
