from math import log, exp

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.classes_ = set()

    def fit(self, X, y):
        total_docs = len(y)
        class_counts = {}
        feature_counts = {}

        # Initialize dictionaries for class counts and feature counts
        for label in y:
            if label not in class_counts:
                class_counts[label] = 0
                feature_counts[label] = [0] * len(X[0])
            class_counts[label] += 1

        # Sum feature counts for each class
        for i, label in enumerate(y):
            for j, feature in enumerate(X[i]):
                feature_counts[label][j] += feature

        # Calculate log prior probabilities for each class
        for label, count in class_counts.items():
            self.class_log_prior_[label] = log(count / total_docs)
            self.classes_.add(label)

        # Calculate log likelihoods (feature probabilities)
        for label in self.classes_:
            total_features = sum(feature_counts[label]) + len(X[0]) * self.alpha  # Total features count with smoothing
            self.feature_log_prob_[label] = [
                log((feature + self.alpha) / total_features) for feature in feature_counts[label]
            ]

        return self

    def predict(self, X):
        predictions = []
        for doc in X:
            class_scores = self._compute_log_scores(doc)
            # Select the class with the highest score
            predictions.append(max(class_scores, key=class_scores.get))

        return predictions

    def predict_proba(self, X):
        probabilities = []
        for doc in X:
            class_scores = self._compute_log_scores(doc)

            # Convert log scores to probabilities using the softmax function
            max_log_score = max(class_scores.values())
            exp_scores = {label: exp(score - max_log_score) for label, score in class_scores.items()}  # Stability trick
            total_exp_scores = sum(exp_scores.values())
            probabilities.append({label: score / total_exp_scores for label, score in exp_scores.items()})

        return probabilities

    def _compute_log_scores(self, doc):
        class_scores = {}
        for label in self.classes_:
            # Start with the log prior for each class
            class_scores[label] = self.class_log_prior_[label]

            # Add the log likelihoods for each feature
            for i, feature in enumerate(doc):
                if feature > 0:  # Only consider non-zero features
                    class_scores[label] += feature * self.feature_log_prob_[label][i]

        return class_scores

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
