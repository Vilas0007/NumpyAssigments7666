"""
NUMPY FOR AI/ML — BIG PRACTICAL ASSIGNMENTS
===========================================

Purpose:
    Practice NumPy skills commonly used in data science, machine learning,
    neural networks, preprocessing, and evaluation.

How to run:
    1. Install NumPy if needed: python -m pip install numpy
    2. Run: python numpy_aiml_big_assignments.py
    3. Choose an assignment number from the menu.

Notes:
    - This file uses synthetic/sample data so it runs without downloading datasets.
    - Each assignment includes comments and a suggested extension.
    - Try solving each TODO section yourself before checking the solution.
"""

import numpy as np


def heading(number, title):
    print("\n" + "=" * 78)
    print(f"ASSIGNMENT {number}: {title}")
    print("=" * 78)


def assignment_1_array_foundations():
    heading(1, "NumPy Array Foundations and Data Types")
    # Create arrays from lists and inspect shape, dimensions, size, and data type.
    a = np.array([10, 20, 30, 40, 50])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print("1-D array:", a)
    print("2-D array:\n", b)
    print("a shape / ndim / size / dtype:", a.shape, a.ndim, a.size, a.dtype)
    print("b shape / ndim / size / dtype:", b.shape, b.ndim, b.size, b.dtype)

    # Generate arrays useful for indexing and experiments.
    print("Zeros:\n", np.zeros((2, 3)))
    print("Ones:\n", np.ones((2, 3)))
    print("Range:", np.arange(0, 12, 2))
    print("Evenly spaced:", np.linspace(0, 1, 5))
    print("Reshaped:\n", np.arange(1, 13).reshape(3, 4))
    print("\nPractice: create a 4x4 identity matrix and a 3x2 array of random integers.")


def assignment_2_indexing_slicing():
    heading(2, "Indexing, Slicing, Boolean Masks, and Fancy Indexing")
    data = np.array([[12, 80, 65], [17, 45, 90], [22, 72, 55], [19, 88, 77]])
    print("Rows are students; columns are Maths, Science, English:\n", data)
    print("First student:", data[0])
    print("Science scores:", data[:, 1])
    print("Last two students:\n", data[-2:, :])
    print("Maths scores above 18:", data[data[:, 0] > 18])
    print("Scores >= 80:", data[data >= 80])
    print("Selected rows 0 and 2:\n", data[[0, 2], :])
    print("\nPractice: select students whose Maths score is at least 18 and English >= 70.")


def assignment_3_vectorization_broadcasting():
    heading(3, "Vectorization and Broadcasting")
    # Vectorized operations apply to whole arrays without explicit Python loops.
    celsius = np.array([-5, 0, 10, 20, 30, 40], dtype=float)
    fahrenheit = celsius * (9 / 5) + 32
    print("Celsius:", celsius)
    print("Fahrenheit:", fahrenheit)

    # Broadcasting: one feature-mean vector is subtracted from every row.
    features = np.array([[10., 100.], [20., 200.], [30., 300.]])
    mean_by_column = features.mean(axis=0)
    centered = features - mean_by_column
    print("Original features:\n", features)
    print("Column means:", mean_by_column)
    print("Centered features:\n", centered)
    print("\nPractice: convert an array of kilometers to miles and normalize a 4x3 matrix by column.")


def assignment_4_statistics_eda():
    heading(4, "Statistical Analysis and Mini EDA")
    rng = np.random.default_rng(42)
    study_hours = rng.normal(loc=4.0, scale=1.5, size=100)
    scores = 35 + study_hours * 10 + rng.normal(0, 7, size=100)
    scores = np.clip(scores, 0, 100)
    print("Study hours (first 10):", np.round(study_hours[:10], 2))
    print("Scores (first 10):", np.round(scores[:10], 2))
    print("Mean score:", round(float(np.mean(scores)), 2))
    print("Median score:", round(float(np.median(scores)), 2))
    print("Standard deviation:", round(float(np.std(scores)), 2))
    print("Variance:", round(float(np.var(scores)), 2))
    print("25th, 50th, 75th percentiles:", np.percentile(scores, [25, 50, 75]))
    print("Correlation matrix:\n", np.corrcoef(study_hours, scores))
    print("Count of scores >= 75:", int(np.sum(scores >= 75)))
    print("\nPractice: report min, max, range, quartiles, and correlation for a new dataset.")


def assignment_5_missing_values():
    heading(5, "Missing Values and Data Cleaning")
    data = np.array([
        [25., 50000., 3.],
        [np.nan, 62000., 5.],
        [31., np.nan, 4.],
        [28., 58000., np.nan],
        [40., 90000., 8.],
    ])
    print("Data (age, salary, experience):\n", data)
    missing_mask = np.isnan(data)
    print("Missing-value mask:\n", missing_mask)
    print("Missing count by column:", missing_mask.sum(axis=0))
    col_means = np.nanmean(data, axis=0)
    cleaned = data.copy()
    row_indices, col_indices = np.where(np.isnan(cleaned))
    cleaned[row_indices, col_indices] = col_means[col_indices]
    print("Column means:", col_means)
    print("Imputed data:\n", cleaned)
    print("\nPractice: compare mean imputation with median imputation and explain when each is useful.")


def assignment_6_scaling_normalization():
    heading(6, "Feature Scaling: Standardization and Min-Max Normalization")
    X = np.array([
        [20., 40000., 2.],
        [25., 55000., 4.],
        [30., 70000., 6.],
        [35., 85000., 8.],
    ])
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    # Guard against division by zero in constant columns.
    std_safe = np.where(std == 0, 1, std)
    standardized = (X - mean) / std_safe
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    span = np.where((x_max - x_min) == 0, 1, x_max - x_min)
    minmax = (X - x_min) / span
    print("Original X:\n", X)
    print("Standardized X:\n", np.round(standardized, 3))
    print("Min-max scaled X:\n", np.round(minmax, 3))
    print("Standardized column means:", standardized.mean(axis=0))
    print("\nPractice: implement a reusable scaling function that can transform train and test data safely.")


def assignment_7_outliers():
    heading(7, "Outlier Detection with IQR and Z-Scores")
    salaries = np.array([28000, 30000, 32000, 31000, 35000, 36000, 37000, 39000, 42000, 250000])
    q1, q3 = np.percentile(salaries, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    iqr_outliers = salaries[(salaries < lower) | (salaries > upper)]
    mean, std = salaries.mean(), salaries.std()
    z = (salaries - mean) / (std if std else 1)
    print("Salaries:", salaries)
    print("Q1, Q3, IQR:", q1, q3, iqr)
    print("IQR bounds:", lower, upper)
    print("IQR outliers:", iqr_outliers)
    print("Z-scores:", np.round(z, 2))
    print("Absolute z-score > 3:", salaries[np.abs(z) > 3])
    print("\nPractice: compare both methods and explain why outlier removal should not be automatic.")


def assignment_8_correlation():
    heading(8, "Correlation Matrix and Redundant Features")
    rng = np.random.default_rng(10)
    income = rng.normal(60000, 12000, 200)
    experience = np.clip((income - 25000) / 8000 + rng.normal(0, 1, 200), 0, None)
    spending = 0.35 * income + rng.normal(0, 7000, 200)
    X = np.column_stack([income, experience, spending])
    corr = np.corrcoef(X, rowvar=False)
    print("Columns: income, experience, spending")
    print("Correlation matrix:\n", np.round(corr, 3))
    print("Upper-triangle correlations:\n", corr[np.triu_indices(3, k=1)])
    print("\nPractice: find pairs with absolute correlation > 0.8 and discuss the limitations of correlation.")


def assignment_9_linear_regression_from_scratch():
    heading(9, "Simple Linear Regression from Scratch")
    # Model: y = slope*x + intercept. Use least-squares formulas.
    x = np.array([1., 2., 3., 4., 5., 6.])
    y = np.array([3., 4., 6., 8., 9., 11.])
    x_mean, y_mean = x.mean(), y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    predictions = slope * x + intercept
    mse = np.mean((y - predictions) ** 2)
    ss_res = np.sum((y - predictions) ** 2)
    ss_total = np.sum((y - y_mean) ** 2)
    r2 = 1 - ss_res / ss_total
    print("x:", x)
    print("y:", y)
    print("Slope:", round(float(slope), 4))
    print("Intercept:", round(float(intercept), 4))
    print("Predictions:", np.round(predictions, 2))
    print("MSE:", round(float(mse), 4))
    print("R2:", round(float(r2), 4))
    print("\nPractice: make a prediction for x=7 and compare your model with np.polyfit(x, y, 1).")


def assignment_10_gradient_descent():
    heading(10, "Gradient Descent for Linear Regression")
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([2., 4., 5., 4., 5.])
    # Standardize x to make gradient descent easier to tune.
    x_mean, x_std = x.mean(), x.std()
    xs = (x - x_mean) / x_std
    w, b = 0.0, 0.0
    learning_rate = 0.05
    epochs = 1000
    n = len(xs)
    losses = []
    for epoch in range(epochs):
        pred = w * xs + b
        error = pred - y
        dw = (2 / n) * np.sum(error * xs)
        db = (2 / n) * np.sum(error)
        w -= learning_rate * dw
        b -= learning_rate * db
        if epoch % 100 == 0:
            losses.append(float(np.mean((w * xs + b - y) ** 2)))
    print("Learned weight:", round(float(w), 4))
    print("Learned bias:", round(float(b), 4))
    print("Loss snapshots every 100 epochs:", np.round(losses, 4))
    print("Predictions:", np.round(w * xs + b, 2))
    print("\nPractice: test different learning rates and explain divergence versus slow convergence.")


def assignment_11_train_test_split():
    heading(11, "Train/Test Split and Data Leakage Awareness")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 3))
    y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.2, 20)
    indices = rng.permutation(len(X))
    test_size = int(0.25 * len(X))
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print("Train X shape:", X_train.shape, "| Train y shape:", y_train.shape)
    print("Test X shape:", X_test.shape, "| Test y shape:", y_test.shape)
    # Correct practice: fit preprocessing values only on training data.
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    train_std = np.where(train_std == 0, 1, train_std)
    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    print("Training means after scaling:", np.round(X_train_scaled.mean(axis=0), 6))
    print("Test sample after train-based scaling:", np.round(X_test_scaled[0], 3))
    print("\nPractice: demonstrate how using the full dataset mean before splitting causes leakage.")


def assignment_12_confusion_matrix_metrics():
    heading(12, "Classification Metrics with NumPy")
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 score:", round(f1, 3))
    print("\nPractice: calculate metrics for an imbalanced dataset and explain why accuracy may mislead.")


def assignment_13_sigmoid_softmax():
    heading(13, "Sigmoid, Softmax, and Binary Classification")
    logits = np.array([-3., -1., 0., 1., 3.])
    sigmoid = 1 / (1 + np.exp(-logits))
    print("Logits:", logits)
    print("Sigmoid probabilities:", np.round(sigmoid, 4))
    print("Predicted class at threshold 0.5:", (sigmoid >= 0.5).astype(int))

    class_logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 1.0], [1.2, 0.8, 2.2]])
    # Subtract each row maximum for numerical stability.
    shifted = class_logits - np.max(class_logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    probabilities = exps / np.sum(exps, axis=1, keepdims=True)
    print("Class logits:\n", class_logits)
    print("Softmax probabilities:\n", np.round(probabilities, 4))
    print("Predicted classes:", np.argmax(probabilities, axis=1))
    print("\nPractice: explain why softmax probabilities in each row sum to one.")


def assignment_14_neural_network_forward_pass():
    heading(14, "Neural Network Forward Pass with NumPy")
    rng = np.random.default_rng(7)
    X = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]])  # 2 samples, 3 features
    W1 = rng.normal(0, 0.5, (3, 4))  # 3 inputs -> 4 hidden units
    b1 = np.zeros((1, 4))
    W2 = rng.normal(0, 0.5, (4, 2))  # 4 hidden units -> 2 output units
    b2 = np.zeros((1, 2))
    hidden_linear = X @ W1 + b1
    hidden = np.maximum(0, hidden_linear)  # ReLU
    output_logits = hidden @ W2 + b2
    shifted = output_logits - output_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    output_probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    print("Input shape:", X.shape)
    print("W1 shape:", W1.shape, "| hidden shape:", hidden.shape)
    print("W2 shape:", W2.shape, "| output shape:", output_probabilities.shape)
    print("Output probabilities:\n", np.round(output_probabilities, 4))
    print("Row sums:", output_probabilities.sum(axis=1))
    print("\nPractice: add a second hidden layer and verify every matrix multiplication shape.")


def assignment_15_loss_functions():
    heading(15, "Loss Functions: MSE, Binary Cross-Entropy, Categorical Cross-Entropy")
    y = np.array([3.0, 5.0, 2.0, 8.0])
    y_hat = np.array([2.5, 5.5, 2.0, 7.0])
    mse = np.mean((y - y_hat) ** 2)
    mae = np.mean(np.abs(y - y_hat))
    print("Regression MSE:", round(float(mse), 4))
    print("Regression MAE:", round(float(mae), 4))

    true_binary = np.array([1., 0., 1., 1.])
    p = np.array([0.9, 0.2, 0.7, 0.6])
    eps = 1e-12
    p_safe = np.clip(p, eps, 1 - eps)
    bce = -np.mean(true_binary * np.log(p_safe) + (1 - true_binary) * np.log(1 - p_safe))
    print("Binary cross-entropy:", round(float(bce), 4))

    true_class = np.array([0, 2, 1])
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2]])
    cce = -np.mean(np.log(np.clip(probs[np.arange(len(true_class)), true_class], eps, 1)))
    print("Categorical cross-entropy:", round(float(cce), 4))
    print("\nPractice: change one prediction at a time and observe how the loss changes.")


def assignment_16_pca_from_scratch():
    heading(16, "Principal Component Analysis (PCA) from Scratch")
    rng = np.random.default_rng(12)
    latent = rng.normal(size=(100, 2))
    transform = np.array([[3.0, 1.0, 0.2], [0.5, 2.0, 1.0]])
    X = latent @ transform + rng.normal(0, 0.15, (100, 3))
    X_centered = X - X.mean(axis=0)
    covariance = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    components = eigenvectors[:, :2]
    X_reduced = X_centered @ components
    explained_ratio = eigenvalues / eigenvalues.sum()
    print("Original shape:", X.shape)
    print("Reduced shape:", X_reduced.shape)
    print("Eigenvalues:", np.round(eigenvalues, 4))
    print("Explained variance ratio:", np.round(explained_ratio, 4))
    print("Variance retained by 2 components:", round(float(explained_ratio[:2].sum()), 4))
    print("\nPractice: reconstruct X from two components and calculate reconstruction MSE.")


def assignment_17_kmeans_from_scratch():
    heading(17, "K-Means Clustering from Scratch")
    rng = np.random.default_rng(21)
    cluster_a = rng.normal([0, 0], 0.7, size=(40, 2))
    cluster_b = rng.normal([4, 4], 0.7, size=(40, 2))
    cluster_c = rng.normal([0, 5], 0.7, size=(40, 2))
    X = np.vstack([cluster_a, cluster_b, cluster_c])
    k = 3
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(50):
        # Distances matrix has shape (samples, clusters).
        distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            points = X[new_labels == cluster_id]
            if len(points):
                new_centers[cluster_id] = points.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels, centers = new_labels, new_centers
    inertia = sum(np.sum((X[labels == i] - centers[i]) ** 2) for i in range(k))
    print("Data shape:", X.shape)
    print("Cluster centers:\n", np.round(centers, 2))
    print("Cluster counts:", np.bincount(labels, minlength=k))
    print("Inertia:", round(float(inertia), 3))
    print("\nPractice: try k=2, 3, 4, and 5 and compare inertia. Why does inertia always tend to fall as k grows?")


def assignment_18_cosine_similarity_recommender():
    heading(18, "Cosine Similarity and a Tiny Recommendation Engine")
    # Rows represent users; columns represent movies/items. Zero means unrated.
    ratings = np.array([
        [5., 4., 0., 1., 0.],
        [4., 5., 0., 1., 0.],
        [1., 0., 5., 4., 4.],
        [0., 1., 4., 5., 5.],
    ])
    norms = np.linalg.norm(ratings, axis=1, keepdims=True)
    normalized = ratings / np.where(norms == 0, 1, norms)
    similarity = normalized @ normalized.T
    print("User-item ratings:\n", ratings)
    print("User-user cosine similarity:\n", np.round(similarity, 3))
    target_user = 0
    similar_users = np.argsort(similarity[target_user])[::-1]
    print("Users most similar to user 0 (including self):", similar_users)
    print("\nPractice: recommend unrated items using ratings from the most similar users; exclude the target user.")


def assignment_19_time_series_features():
    heading(19, "Time-Series Features and Moving Averages")
    rng = np.random.default_rng(33)
    days = np.arange(1, 31)
    sales = 100 + 1.8 * days + 10 * np.sin(days / 3) + rng.normal(0, 4, len(days))
    window = 5
    moving_average = np.convolve(sales, np.ones(window) / window, mode="valid")
    # First difference can highlight changes between consecutive observations.
    difference = np.diff(sales)
    print("Days:", days)
    print("Sales (rounded):", np.round(sales, 1))
    print(f"{window}-day moving average (first 10):", np.round(moving_average[:10], 1))
    print("Day-to-day differences (first 10):", np.round(difference[:10], 1))
    print("\nPractice: calculate rolling mean for windows 3, 7, and 14. Avoid using future data in forecasting features.")


def assignment_20_capstone():
    heading(20, "CAPSTONE: End-to-End NumPy ML Data Preparation and Evaluation")
    rng = np.random.default_rng(101)
    n = 300
    age = rng.integers(18, 66, n).astype(float)
    income = rng.normal(55000, 15000, n)
    visits = rng.poisson(5, n).astype(float)
    # Generate a synthetic binary outcome from a simple score.
    score = -2.2 + 0.025 * (age - 40) + 0.000035 * (income - 50000) + 0.22 * visits
    probability = 1 / (1 + np.exp(-score))
    y = (rng.random(n) < probability).astype(int)
    X = np.column_stack([age, income, visits])

    # 1. Split first to prevent information leakage.
    order = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = order[:split], order[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 2. Fit standardization parameters using training data only.
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma == 0, 1, sigma)
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # 3. Fit logistic regression with gradient descent.
    weights = np.zeros(X_train_s.shape[1])
    bias = 0.0
    lr = 0.1
    epochs = 1500
    eps = 1e-12
    for _ in range(epochs):
        logits = X_train_s @ weights + bias
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        dw = X_train_s.T @ (probs - y_train) / len(y_train)
        db = np.mean(probs - y_train)
        weights -= lr * dw
        bias -= lr * db

    # 4. Evaluate using held-out test data.
    test_logits = X_test_s @ weights + bias
    test_probs = 1 / (1 + np.exp(-np.clip(test_logits, -50, 50)))
    y_pred = (test_probs >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    loss = -np.mean(y_test * np.log(np.clip(test_probs, eps, 1)) +
                    (1 - y_test) * np.log(np.clip(1 - test_probs, eps, 1)))
    print("Feature names: age, income, visits")
    print("Train/test shapes:", X_train_s.shape, X_test_s.shape)
    print("Learned weights:", np.round(weights, 4), "| bias:", round(float(bias), 4))
    print("Test accuracy:", round(float(accuracy), 3))
    print("Test precision:", round(float(precision), 3))
    print("Test recall:", round(float(recall), 3))
    print("Test log loss:", round(float(loss), 4))
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print("\nCAPSTONE extensions:")
    print("  a) Add missing values and implement imputation.")
    print("  b) Compare threshold 0.3, 0.5, and 0.7.")
    print("  c) Add L2 regularization to the weight update.")
    print("  d) Run the experiment with 5 random seeds and report mean/std metrics.")
    print("  e) Explain why synthetic results do not prove real-world model quality.")


ASSIGNMENTS = {
    "1": assignment_1_array_foundations,
    "2": assignment_2_indexing_slicing,
    "3": assignment_3_vectorization_broadcasting,
    "4": assignment_4_statistics_eda,
    "5": assignment_5_missing_values,
    "6": assignment_6_scaling_normalization,
    "7": assignment_7_outliers,
    "8": assignment_8_correlation,
    "9": assignment_9_linear_regression_from_scratch,
    "10": assignment_10_gradient_descent,
    "11": assignment_11_train_test_split,
    "12": assignment_12_confusion_matrix_metrics,
    "13": assignment_13_sigmoid_softmax,
    "14": assignment_14_neural_network_forward_pass,
    "15": assignment_15_loss_functions,
    "16": assignment_16_pca_from_scratch,
    "17": assignment_17_kmeans_from_scratch,
    "18": assignment_18_cosine_similarity_recommender,
    "19": assignment_19_time_series_features,
    "20": assignment_20_capstone,
}


def main():
    while True:
        print("\n" + "#" * 78)
        print("NUMPY FOR AI/ML — ASSIGNMENT MENU")
        print("#" * 78)
        for key, function in ASSIGNMENTS.items():
            print(f"{key:>2}. {function.__doc__ or function.__name__.replace('assignment_', '').replace('_', ' ')}")
        print(" 0. Exit")
        choice = input("\nChoose an assignment (0-20): ").strip()
        if choice == "0":
            print("Good work. Keep practicing NumPy!")
            break
        function = ASSIGNMENTS.get(choice)
        if function is None:
            print("Invalid choice. Enter a number from 0 to 20.")
            continue
        try:
            function()
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            print(f"This assignment encountered a numerical/input issue: {error}")
        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()
