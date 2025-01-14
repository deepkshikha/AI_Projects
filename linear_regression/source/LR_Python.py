import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, learning_rate=0.001, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.random.randn(self.n) * 0.01  # Small random weights
        self.b = 0.0
        self.X = X
        self.Y = Y

        # List to store loss values for each iteration
        loss_history = []

        for i in range(self.iterations):
            self.update_weights()
            current_loss = np.mean((self.Y - self.predict(self.X)) ** 2)
            loss_history.append(current_loss)

            # Print loss every 1000 iterations for monitoring
            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {current_loss}")

        # Plot the loss vs. iteration graph
        plt.plot(range(self.iterations), loss_history, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs. Iteration')
        plt.legend()
        plt.show()

        return self


    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = -(2 / self.m) * self.X.T.dot(self.Y - Y_pred)
        db = -(2 / self.m) * np.sum(self.Y - Y_pred)
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.W) + self.b

def main():
    # Load dataset
    pdf = pd.read_csv("../../../Downloads/BostonHousing.csv")
    
    # Handle missing values in the 'rm' column by filling with the column mean
    pdf['rm'].fillna(pdf['rm'].mean(), inplace=True)
    
    # Define features (X) and target (Y)
    X = pdf.drop('medv', axis=1)  # Drop the target column from features
    Y = pdf['medv']  
    
    
    # Check for NaN or infinite values
    assert not np.any(np.isnan(X)), "X contains NaN"
    assert not np.any(np.isnan(Y)), "Y contains NaN"

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Train the model
    model = LinearRegression(learning_rate=0.16, iterations=10000)
    model.fit(X_train, Y_train)

    # Predictions
    Y_pred = model.predict(X_test)
    print(Y_pred)

    # Visualization
    plt.scatter(Y_test, Y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')
    plt.show()

if __name__ == "__main__":
    main()