import numpy as np
import pandas as pd


class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.1):
        self.weights = np.zeros((input_size, output_size))
        self.learning_rate = learning_rate

    def train(self, inputs, outputs):
        for i in range(len(inputs)):
            input_pixels_vector = inputs[i]
            output_letter_vector = outputs[i]

            # Update weights
            self.weights += self.learning_rate * np.outer(input_pixels_vector, output_letter_vector)

        # Add weight decay to prevent overfitting
        self.weights *= 0.99  # Slightly decay weights

        # Normalize weights safely
        norms = np.linalg.norm(self.weights, axis=0, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.weights /= norms
        # print(self.weights)
        print(self.weights.shape)

    def predict(self, input_pixels_vector):
        output = np.dot(input_pixels_vector, self.weights)
        # print(output.shape)
        return np.maximum(output, 0)  # Apply ReLU to output


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


def create_50_samples_per_letter(df, pixel_columns, num_pixels_to_change):
    """
    Creates augmented data by flipping the values of randomly selected pixels
    for each letter's sample in the dataset.
    """
    augmented_data = []
    letters = df['letter'].unique()

    for letter in letters:
        letter_data = df[df['letter'] == letter]
        for _ in range(50 - len(letter_data)):
            # Duplicate an existing row and modify specific pixels
            sample = letter_data.iloc[0].copy()
            random_pixels = np.random.choice(pixel_columns, size=num_pixels_to_change, replace=False)

            for pixel in random_pixels:
                # Flip the current pixel value (0 -> 1, 1 -> 0)
                sample[pixel] = 1 - sample[pixel]

            augmented_data.append(sample)

    return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)


def split_train_test(df, pixel_columns):
    train_data = []
    test_data = []

    letters = df['letter'].unique()
    for letter in letters:
        letter_data = df[df['letter'] == letter]
        train_data.append(letter_data.iloc[:35])  # 35 samples for training
        test_data.append(letter_data.iloc[35:])   # 15 samples for testing

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    X_train = train_df[pixel_columns].values
    y_train = train_df['letter'].values

    X_test = test_df[pixel_columns].values
    y_test = test_df['letter'].values

    return X_train, X_test, y_train, y_test


# One-Hot coding
def letter_to_one_hot(letter):
    index = ord(letter) - ord('A')  # Map 'A' to 0, 'B' to 1, ..., 'Z' to 25
    one_hot = np.zeros(26)  # Create a zero vector of length 26
    one_hot[index] = 1  # Set the corresponding index to 1
    return one_hot


def evaluate_network(X, y, network):
    correct = 0
    for i in range(len(X)):
        input_vector = X[i]
        predicted_output = network.predict(input_vector)

        predicted_letter_index = np.argmax(predicted_output)  # Find the index of the highest score
        actual_letter_index = ord(y[i]) - ord('A')

        if predicted_letter_index == actual_letter_index:
            correct += 1

    accuracy = correct / len(X) * 100
    return accuracy


def evaluate_network_3(X, y, network):
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

        accuracy = correct / len(X) * 100
        return accuracy


# Load the dataset
file_path = 'letters.csv'
letters_df = pd.read_csv(file_path)

# Pixel columns
pixel_columns = [f'pixel{i}' for i in range(1, 65)]  # pixel1, pixel2, ..., pixel64

# Create data with noise (5%, 10%, 20%, 50%)
letters_4_pixels = create_50_samples_per_letter(letters_df, pixel_columns, num_pixels_to_change=4)
letters_7_pixels = create_50_samples_per_letter(letters_df, pixel_columns, num_pixels_to_change=7)
letters_13_pixels = create_50_samples_per_letter(letters_df, pixel_columns, num_pixels_to_change=13)
letters_32_pixels = create_50_samples_per_letter(letters_df, pixel_columns, num_pixels_to_change=32)

X_train_4, X_test_4, y_train_4, y_test_4 = split_train_test(letters_4_pixels, pixel_columns)
X_train_7, X_test_7, y_train_7, y_test_7 = split_train_test(letters_7_pixels, pixel_columns)
X_train_13, X_test_13, y_train_13, y_test_13 = split_train_test(letters_13_pixels, pixel_columns)
X_train_32, X_test_32, y_train_32, y_test_32 = split_train_test(letters_32_pixels, pixel_columns)

# Normalize the data
letters_4_pixels[pixel_columns] = letters_4_pixels[pixel_columns] / 255.0
letters_7_pixels[pixel_columns] = letters_7_pixels[pixel_columns] / 255.0
letters_13_pixels[pixel_columns] = letters_13_pixels[pixel_columns] / 255.0
letters_32_pixels[pixel_columns] = letters_32_pixels[pixel_columns] / 255.0

output_vectors_train_4 = np.array([categorize_letter(letter) for letter in y_train_4])
output_vectors_train_7 = np.array([categorize_letter(letter) for letter in y_train_7])
output_vectors_train_13 = np.array([categorize_letter(letter) for letter in y_train_13])
output_vectors_train_32 = np.array([categorize_letter(letter) for letter in y_train_32])

# Initialize and train the network for each variation
network_4 = HebbianNetwork(input_size=len(pixel_columns), output_size=3, learning_rate=0.1)
network_4.train(X_train_4, output_vectors_train_4)

network_7 = HebbianNetwork(input_size=len(pixel_columns), output_size=3, learning_rate=0.1)
network_7.train(X_train_7, output_vectors_train_7)

network_13 = HebbianNetwork(input_size=len(pixel_columns), output_size=3, learning_rate=0.1)
network_13.train(X_train_13, output_vectors_train_13)

network_32 = HebbianNetwork(input_size=len(pixel_columns), output_size=3, learning_rate=0.1)
network_32.train(X_train_32, output_vectors_train_32)

# Evaluate for each variation
print("3 categories")

print("4 Pixels Variation - 5% noise")
print(f"Accuracy on training data: {evaluate_network_3(X_train_4, y_train_4, network_4):.2f}%")
print(f"Accuracy on testing data: {evaluate_network_3(X_test_4, y_test_4, network_4):.2f}%")

print("\n7 Pixels Variation - 10% noise")
print(f"Accuracy on training data: {evaluate_network_3(X_train_7, y_train_7, network_7):.2f}%")
print(f"Accuracy on testing data: {evaluate_network_3(X_test_7, y_test_7, network_7):.2f}%")

print("\n13 Pixels Variation - 20% noise")
print(f"Accuracy on training data: {evaluate_network_3(X_train_13, y_train_13, network_13):.2f}%")
print(f"Accuracy on testing data: {evaluate_network_3(X_test_13, y_test_13, network_13):.2f}%")

print("\n32 Pixels Variation - 50% noise")
print(f"Accuracy on training data: {evaluate_network_3(X_train_32, y_train_32, network_32):.2f}%")
print(f"Accuracy on testing data: {evaluate_network_3(X_test_32, y_test_32, network_32):.2f}%")


output_vectors_train_4 = np.array([letter_to_one_hot(letter) for letter in y_train_4])
output_vectors_train_7 = np.array([letter_to_one_hot(letter) for letter in y_train_7])
output_vectors_train_13 = np.array([letter_to_one_hot(letter) for letter in y_train_13])
output_vectors_train_32 = np.array([letter_to_one_hot(letter) for letter in y_train_32])


# Initialize and train the network for each variation
network_4 = HebbianNetwork(input_size=len(pixel_columns), output_size=26, learning_rate=0.1)
network_4.train(X_train_4, output_vectors_train_4)

network_7 = HebbianNetwork(input_size=len(pixel_columns), output_size=26, learning_rate=0.1)
network_7.train(X_train_7, output_vectors_train_7)

network_13 = HebbianNetwork(input_size=len(pixel_columns), output_size=26, learning_rate=0.1)
network_13.train(X_train_13, output_vectors_train_13)

network_32 = HebbianNetwork(input_size=len(pixel_columns), output_size=26, learning_rate=0.1)
network_32.train(X_train_32, output_vectors_train_32)

# Evaluate for each variation
print("26 categories")

print("4 Pixels Variation - 5% noise")
print(f"Accuracy on training data: {evaluate_network(X_train_4, y_train_4, network_4):.2f}%")
print(f"Accuracy on testing data: {evaluate_network(X_test_4, y_test_4, network_4):.2f}%")

print("\n7 Pixels Variation - 10% noise")
print(f"Accuracy on training data: {evaluate_network(X_train_7, y_train_7, network_7):.2f}%")
print(f"Accuracy on testing data: {evaluate_network(X_test_7, y_test_7, network_7):.2f}%")

print("\n13 Pixels Variation - 20% noise")
print(f"Accuracy on training data: {evaluate_network(X_train_13, y_train_13, network_13):.2f}%")
print(f"Accuracy on testing data: {evaluate_network(X_test_13, y_test_13, network_13):.2f}%")

print("\n32 Pixels Variation - 50% noise")
print(f"Accuracy on training data: {evaluate_network(X_train_32, y_train_32, network_32):.2f}%")
print(f"Accuracy on testing data: {evaluate_network(X_test_32, y_test_32, network_32):.2f}%")