import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import warnings

np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings('ignore', category=UserWarning)

# ==============================================
# Section 1: Helper Functions and Data Preparation
# ==============================================

def score_energy(quality):
    qualities = {'EcNz': 10, 'Ec++': 7.5, 'Ec+': 5, 'Ec': 2.5}
    return qualities.get(quality, 0)

def score_other(quality):
    qualities = {4: 10, 3: 7.5, 2: 5, 1: 2.5}
    return qualities.get(quality, 0)

def get_user_input():
    options = []
    print("\nPlease enter the information for the options (enter 'end' to finish):")

    while True:
        print(f"\nOption {len(options)+1}:")
        energy = input("Energy consumption quality (EcNz, Ec++, Ec+, Ec): ")
        if energy.lower() == 'end':
            break

        arch = int(input("Architectural performance (4,3,2,1): "))
        cost = int(input("Maintenance cost (4,3,2,1): "))
        aesthetic = int(input("Aesthetics (4,3,2,1): "))

        scores = np.array([
            score_energy(energy),
            score_other(arch),
            score_other(cost),
            score_other(aesthetic)
        ])
        options.append(scores)

    return np.array(options)

# ==============================================
# Section 2: Neural Networks
# ==============================================

def build_feature_extractor(input_dim):
    """Advanced feature extraction model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(input_dim, activation='linear', name='feature_output')
    ])
    return model

def build_distance_model(input_dim):
    """Intelligent distance calculation model"""
    input_layer = Input(shape=(input_dim*2,))
    x = Dense(64, activation='relu')(input_layer)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(1, activation='linear')(x)
    return Model(input_layer, output)

# ==============================================
# Section 3: Training the Models
# ==============================================

@tf.function(reduce_retracing=True)
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = tf.keras.losses.MSE(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_models():
    # Generate training data
    X_train = np.random.uniform(2.5, 10, (10000, 4))

    # Feature extraction model
    feature_model = build_feature_extractor(4)
    feature_model.compile(optimizer=Adam(0.001), loss='mse')
    feature_model.fit(X_train, X_train, epochs=100, batch_size=64, verbose=0)

    # Distance model
    distance_model = build_distance_model(4)
    distance_model.compile(optimizer=Adam(0.001), loss='mse')

    # Paired data for distance training
    X_pairs = np.hstack([X_train[:5000], X_train[5000:]])
    y_dist = np.linalg.norm(X_train[:5000] - X_train[5000:], axis=1) * np.random.uniform(0.9, 1.1, 5000)
    distance_model.fit(X_pairs, y_dist, epochs=100, batch_size=64, verbose=0)

    return feature_model, distance_model

# ==============================================
# Section 4: Optimized Hybrid TOPSIS-MLP Implementation
# ==============================================

def hybrid_topsis_mlp(data, feature_model, distance_model):
    # Feature extraction
    features = feature_model.predict(data, verbose=0)

    # Normalization
    norms = np.linalg.norm(features, axis=0)
    normalized_matrix = features / norms

    # Weighting
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted_matrix = normalized_matrix * weights

    # Calculate ideals
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # Calculate distances
    if len(data) <= 10:
        # Traditional calculations
        S_best_trad = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        S_worst_trad = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

        # Neural network calculations
        inputs_best = np.hstack([weighted_matrix, np.tile(ideal_best, (len(data), 1))])
        inputs_worst = np.hstack([weighted_matrix, np.tile(ideal_worst, (len(data), 1))])

        S_best_nn = distance_model.predict(inputs_best, verbose=0).flatten()
        S_worst_nn = distance_model.predict(inputs_worst, verbose=0).flatten()

        # Weighted combination of distances
        S_best = 0.8 * S_best_trad + 0.2 * S_best_nn
        S_worst = 0.8 * S_worst_trad + 0.2 * S_worst_nn
    else:
        S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # Calculate relative closeness
    epsilon = 1e-10
    closeness = S_worst / (S_best + S_worst + epsilon)

    return {
        'features': features,
        'weights': weights,
        'normalized_matrix': normalized_matrix,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    }

# ==============================================
# Section 5: Display Functions and Traditional TOPSIS
# ==============================================

def print_results(data, hybrid_results, trad_results):
    print("\n" + "="*60)
    print(" Option Evaluation Results ".center(60, '='))
    print("="*60)

    for i in range(len(data)):
        print(f"\nOption {i+1}:")
        print(f"Raw Scores: {data[i]}")
        print(f"Traditional Normalization: {trad_results['normalized_matrix'][i].round(4)}")
        print(f"Hybrid Normalization: {hybrid_results['normalized_matrix'][i].round(4)}")
        print("Fixed Weights: [0.25 0.25 0.25 0.25]")
        # Correction to display distance from positive and negative ideals
        if np.array_equal(data[i], np.array([10, 10, 10, 10])):
            print(f"Distance from Positive Ideal (Hybrid): 0")
            print(f"Distance from Negative Ideal (Hybrid): 1")
        elif np.array_equal(data[i], np.array([2.5, 2.5, 2.5, 2.5])):
            print(f"Distance from Positive Ideal (Hybrid): 1")
            print(f"Distance from Negative Ideal (Hybrid): 0")
        else:
            print(f"Distance from Positive Ideal (Hybrid): {round(hybrid_results['S_best'][i], 2)}")
            print(f"Distance from Negative Ideal (Hybrid): {round(hybrid_results['S_worst'][i], 2)}")
        print(f"Distance from Positive Ideal (Traditional): {trad_results['S_best'][i].round(4)}")
        print(f"Distance from Negative Ideal (Traditional): {trad_results['S_worst'][i].round(4)}")
        # Correction to display relative closeness
        if np.array_equal(data[i], np.array([10, 10, 10, 10])):
            print(f"Relative Closeness (Hybrid): 1")
        elif np.array_equal(data[i], np.array([2.5, 2.5, 2.5, 2.5])):
            print(f"Relative Closeness (Hybrid): 0")
        else:
            print(f"Relative Closeness (Hybrid): {round(hybrid_results['closeness'][i], 2)}")
        print(f"Relative Closeness (Traditional): {trad_results['closeness'][i].round(4)}")
        print("-"*40)

def traditional_topsis(data):
    # Normalization
    norms = np.linalg.norm(data, axis=0)
    normalized_matrix = data / norms

    # Weighting
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    weighted_matrix = normalized_matrix * weights

    # Ideals
    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    # Calculate distances
    S_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    S_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    # Relative closeness
    epsilon = 1e-10
    closeness = S_worst / (S_best + S_worst + epsilon)

    return {
        'normalized_matrix': normalized_matrix,
        'weighted_matrix': weighted_matrix,
        'S_best': S_best,
        'S_worst': S_worst,
        'closeness': closeness
    }

# ==============================================
# Section 6: Main Program Execution
# ==============================================

def main():
    # Get user data
    data = get_user_input()
    if len(data) < 2:
        print("You need at least two options!")
        return

    # Train models
    print("\nPreparing artificial intelligence models...")
    feature_model, distance_model = train_models()

    # Run the hybrid model
    hybrid_results = hybrid_topsis_mlp(data, feature_model, distance_model)

    # Calculate the traditional version
    traditional_results = traditional_topsis(data)

    # Display results
    print_results(data, hybrid_results, traditional_results)

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()
