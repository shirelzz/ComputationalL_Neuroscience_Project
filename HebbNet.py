import numpy as np
import pandas as pd


class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.zeros((input_size, output_size))
        self.learning_rate = learning_rate

    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            input_pixels_vector = inputs[i]
            output_category_vector = outputs[i]

            # Update weights
            self.weights += self.learning_rate * np.outer(input_pixels_vector, output_category_vector)

    def predict(self, input_pixels_vector):
        return np.dot(input_pixels_vector, self.weights)


file_path = 'letters.csv'
letters_df = pd.read_csv(file_path)

X = letters_df.iloc[:, 1:].values  # Pixels
y = letters_df['letter'].values  # Labels (letters)
# print(X)
# print(y)


# Create output groups (A-I, J-R, S-Z)
def categorize_letter(letter):
    if 'A' <= letter <= 'I':
        return [1, 0, 0]
    elif 'J' <= letter <= 'R':
        return [0, 1, 0]
    elif 'S' <= letter <= 'Z':
        return [0, 0, 1]


# Check if the actual label y[i] is within the predicted category range
def is_in_range(letter, category_range):
    start, end = category_range.split('-')  # Split the range like 'A-I' into start ('A') and end ('I')
    return start <= letter <= end  # Check if the letter falls within the range


def add_noise(X, noise_factor):
    noisy_X = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    noisy_X = np.clip(noisy_X, 0.0, 1.0)  # Keep values between 0 and 1
    return noisy_X


def test_with_noise(X, y, noise_levels):
    results = []
    for noise_factor in noise_levels:
        noisy_X = add_noise(X, noise_factor)
        correct = 0
        for i in range(len(noisy_X)):
            input_vector = noisy_X[i]
            predicted_output = network.predict(input_vector)
            recognized_class = np.argmax(predicted_output)
            category = ['A-I', 'J-R', 'S-Z'][recognized_class]
            if is_in_range(y[i], category):
                correct += 1
        accuracy = correct / len(X) * 100
        results.append((noise_factor, accuracy))
    return results


output_vectors = np.array([categorize_letter(letter) for letter in y])
# print(output_vectors)
network = HebbianNetwork(input_size=64, output_size=3, learning_rate=0.5)
network.train(X, output_vectors)

categories = ['A-I', 'J-R', 'S-Z']
results = []
correct = 0
for i in range(len(X)):
    input_vector = X[i]
    predicted_output = network.predict(input_vector)
    recognized_class = np.argmax(predicted_output)
    results.append((y[i], categories[recognized_class]))

    # Get category range (e.g., 'A-I', 'J-R', 'S-Z')
    category_range = categories[recognized_class]

    # Check if the actual label y[i] is within the range of the predicted category
    if is_in_range(y[i], category_range):
        correct += 1

# Show results
print(f"Predictions and their actual labels: {results[:26]}")
print(f"Accuracy: {correct / len(X) * 100}%")

noise_levels = [0.05, 0.1, 0.2]  # Ranges for noise (5%, 10%, 20%)
results = test_with_noise(X, y, noise_levels)

print("Accuracy under different noise levels:")
for noise_factor, accuracy in results:
    print(f"Noise level {noise_factor*100}%: Accuracy = {accuracy}%")