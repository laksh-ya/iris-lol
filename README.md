# Iris Flower Classifier

a simple machine learning app that predicts iris flower species based on their measurements.
Live demo: https://lakshyairis.streamlit.app/

Built this as a quick weekend project to practice deploying ML models.

The app uses a Random Forest classifier trained on the classic iris dataset - you know, the one with setosa, versicolor, and virginica flowers.

Just input the sepal length, sepal width, petal length, and petal width, and it'll predict which species you've got.

The preprocessing pipeline uses StandardScaler to normalize the features before feeding them to the model.

Training was done with a 70/30 train-test split, and the model pulls around 97% accuracy on the test set, which is pretty solid for such a straightforward setup.

