# Importing required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class KMeansAlternative:
    def __init__(self, data=None):
        self.plot = True  # True if you want to plot final result else False
        self.data = data if data is not None else np.random.rand(100, 2)

    def operation(self):
        k = 3  # the number of clusters to be made
        iterations = 10  # the number of iterations updating centroids

        # Using scikit-learn's KMeans for initialization and comparison
        kmeans_sklearn = KMeans(n_clusters=k, max_iter=iterations, random_state=0)
        clusters_sklearn = kmeans_sklearn.fit_predict(self.data)

        # Convert data to TensorFlow tensors
        X_tensor = tf.constant(self.data, dtype=tf.float32)

        # K-means algorithm using TensorFlow and Keras
        kmeans_model = keras.Sequential([
            keras.layers.Input(shape=(self.data.shape[1],)),
            keras.layers.Dense(k, activation='softmax')
        ])

        kmeans_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        kmeans_model.fit(X_tensor, np.zeros(len(self.data)), epochs=iterations, verbose=0)
        
        # Get cluster assignments from the Keras model
        clusters_keras = np.argmax(kmeans_model.predict(X_tensor), axis=1)

        # Plot the results
        if self.plot:
            self.__plot__(self.data, clusters_sklearn, kmeans_sklearn.cluster_centers_,
                           clusters_keras, kmeans_model.get_layer(index=0).get_weights()[0].T)

        return True

    def __plot__(self, X, clusters_sklearn, centroids_sklearn, clusters_keras, centroids_keras):
        '''
        Plotting the final cluster using matplotlib
        '''
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot using scikit-learn results
        axs[0].scatter(X[:, 0], X[:, 1], c=clusters_sklearn, cmap='viridis', alpha=0.7, edgecolors='k')
        axs[0].scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], c='red', marker='X', s=200, label='Centroids')
        axs[0].set_title('Scikit-learn KMeans')

        # Plot using TensorFlow and Keras results
        axs[1].scatter(X[:, 0], X[:, 1], c=clusters_keras, cmap='viridis', alpha=0.7, edgecolors='k')
        axs[1].scatter(centroids_keras[:, 0], centroids_keras[:, 1], c='red', marker='X', s=200, label='Centroids')
        axs[1].set_title('TensorFlow and Keras')

        plt.show()


if __name__ == "__main__":
    # Generate sample data
    data, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)

    kobj = KMeansAlternative(data)
    result = kobj.operation()
