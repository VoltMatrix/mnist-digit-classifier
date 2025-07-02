# 🌟 Step 1: Importing Libraries
# What: Brings in tools (like a toolbox) needed for the project, such as data loaders, plot makers, and model builders.
# Why: Without these tools, we can’t load the data, see it, or train a model to recognize digits. Each tool has a job, like a chef using different knives.
# How it fits: Sets up the foundation so we can start working with the MNIST dataset and build our digit classifier.
from sklearn.datasets import fetch_openml  # What: Gets the MNIST data from the internet. Why: Provides the dataset we need to learn from. How it fits: Starts the process by fetching 70,000 digit images.
import matplotlib.pyplot as plt  # What: Lets us draw pictures of digits or charts to understand them. Why: Helps us see the data and results visually. How it fits: Builds intuition for designing the model.
import numpy as np  # What: Helps with math and organizing data like a calculator and file organizer. Why: Essential for handling numbers and arrays in the dataset. How it fits: Supports all numerical tasks in classification.
from sklearn.linear_model import SGDClassifier  # What: Gives us a simple model to learn digit patterns. Why: Starts with an easy, fast model to understand classification basics. How it fits: Trains the model to spot digits like 5.
from sklearn.model_selection import cross_val_predict, cross_val_score  # What: Checks how good our model is by testing it in different ways. Why: Ensures fair testing to see if the model works on new data. How it fits: Improves reliability of digit recognition.
from sklearn.base import BaseEstimator  # What: Allows us to make custom tools if needed (not used here but ready). Why: Prepares us for future customizations. How it fits: Keeps options open for advanced tasks.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve  # What: Provides ways to measure if our model is right (e.g., precision, recall). Why: Helps us judge how well the model finds digits. How it fits: Ensures we can evaluate and improve the classifier.
from sklearn.metrics import roc_curve, roc_auc_score  # What: Tools to plot ROC curves and compute Area Under Curve (AUC) for model comparison. Why: Shows how well the model separates digits. How it fits: Helps pick the best model for digit classification.
from sklearn.ensemble import RandomForestClassifier  # What: Offers a stronger model by combining many smaller models. Why: Improves accuracy by using multiple decision trees. How it fits: Tests a better way to recognize digits.
from sklearn.svm import SVC  # What: Brings in a powerful model for separating digits. Why: Handles complex patterns in digit shapes. How it fits: Expands the model to classify all digits.
from sklearn.multiclass import OneVsRestClassifier  # What: Helps the model handle multiple digits at once. Why: Turns a binary model into one that can handle 0-9. How it fits: Prepares for full digit recognition.
from sklearn.preprocessing import StandardScaler  # What: Adjusts data to make the model work better. Why: Ensures all pixel values are on the same scale. How it fits: Boosts model performance for digit classification.
from sklearn.neighbors import KNeighborsClassifier  # What: Uses nearby examples to guess digits with multiple labels. Why: Good for tasks with multiple outputs like odd/even. How it fits: Adds flexibility for complex classification tasks.

# 🌟 Step 2: Loading the MNIST Dataset
# What: Downloads 70,000 digit images and splits them into pictures (X) and their labels (y, like answers).
# Why: We need data to teach the model what digits look like. MNIST is a famous set everyone uses to practice.
# How it fits: Gives us the raw material to start training our digit-recognizing system.
mnist = fetch_openml('mnist_784', version=1, as_frame=False)  # What: Grabs the MNIST data as simple number lists, not fancy tables. Why: Makes data easier and faster to process. How it fits: Loads the dataset for analysis.
X, y = mnist["data"], mnist["target"]  # What: Separates the images (X) from the digit labels (y). Why: Organizes the data into features (pixels) and targets (digits 0-9). How it fits: Prepares data for training.

# 🌟 Step 3: Converting Labels to Integers
# What: Turns the labels (e.g., "5") into numbers (e.g., 5) that the computer can use.
# Why: Computers work better with numbers than text, and this saves memory by using a small number type.
# How it fits: Prepares the labels so the model can learn to match images to digits.
y = y.astype(np.uint8)  # What: Changes the labels to a compact number format (0-255, enough for 0-9). Why: Saves space and ensures compatibility with models. How it fits: Sets up labels for learning.

# 🌟 Step 4: Visualizing a Sample Image
# What: Shows the first image (a 5) on the screen to see what we’re working with.
# Why: Seeing the image helps us check if the data is correct and understand what a digit looks like in pixels.
# How it fits: Builds our intuition about the problem, making it easier to design a good model.
some_digit = X[0]  # What: Picks the first image from the dataset. Why: Gives us a sample to look at. How it fits: Starts our exploration of the data.
some_digit_image = some_digit.reshape(28, 28)  # What: Rearranges the 784 pixels into a 28x28 square picture. Why: Matches the image’s original shape for viewing. How it fits: Prepares the image for display.
plt.imshow(some_digit_image, cmap="binary")  # What: Draws the picture in black and white. Why: Shows the digit clearly as it was written. How it fits: Lets us see the data visually.
plt.axis("off")  # What: Removes the axes to focus on the digit. Why: Keeps the view clean and focused on the image. How it fits: Enhances the visual understanding.
plt.show()  # What: Displays the image for us to see. Why: Allows us to confirm the digit (should look like a 5). How it fits: Validates the data loading process.

# 🌟 Step 5: Splitting Data into Training and Test Sets
# What: Divides the 70,000 images into 60,000 for learning (training) and 10,000 for testing.
# Why: We need a separate test set to check if the model works on new images, not just the ones it learned from.
# How it fits: Ensures our model can handle digits it hasn’t seen before, a key part of making it reliable.
X_train, X_test = X[:60000], X[60000:]  # What: Takes the first 60,000 for training and the last 10,000 for testing. Why: Splits data to train and then test the model. How it fits: Sets up the learning and evaluation phases.
y_train, y_test = y[:60000], y[60000:]  # What: Matches the labels to the training and test images. Why: Ensures the model knows the answers for both sets. How it fits: Links images to their correct digits.

# 🌟 Step 6: Creating Binary Labels for Digit 5
# What: Labels each image as True (if it’s a 5) or False (if it’s not) for a simpler task.
# Why: Starting with one digit (5) makes it easier to learn how classification works before tackling all digits.
# How it fits: Builds a foundation for understanding how to train a model to spot specific digits.
y_train_5 = (y_train == 5)  # What: Marks True for 5s and False for others in the training set. Why: Creates a yes/no question (is it a 5?). How it fits: Simplifies the first classification task.
y_test_5 = (y_test == 5)    # What: Does the same for the test set. Why: Applies the same rule to test data. How it fits: Ensures consistent evaluation.

# 🌟 Step 7: Training an SGDClassifier for Binary Classification
# What: Teaches a simple model (SGDClassifier) to recognize 5s using the training data.
# Why: SGD is a fast, basic model that helps us start learning how to classify digits without overcomplicating things.
# How it fits: Trains the model to detect 5s, a first step toward recognizing all digits.
sgd_clf = SGDClassifier(random_state=42)  # What: Sets up the model with a fixed random seed for consistent results. Why: Ensures the same results every time we run it. How it fits: Makes experiments repeatable.
sgd_clf.fit(X_train, y_train_5)  # What: Lets the model learn from the training images and their 5/not-5 labels. Why: Teaches the model to spot 5s by adjusting its settings. How it fits: Builds the core ability to classify.

# 🌟 Step 8: Getting Cross-Validated Predictions
# What: Uses a method to predict 5s on the training data without letting the model peek at the answers during prediction.
# Why: This gives us a fair test of the model’s ability, showing how it might do with new data.
# How it fits: Helps us measure the model’s performance reliably, ensuring it can classify 5s well.
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)  # What: Predicts 5s using three different data splits. Why: Tests the model on parts it hasn’t seen, reducing bias. How it fits: Provides a solid performance check.

# 🌟 Step 9: Computing the Confusion Matrix
# What: Creates a table showing how many predictions were correct or wrong for 5s vs. non-5s.
# Why: Tells us exactly where the model succeeds (e.g., spotting 5s) or fails (e.g., mistaking 3s for 5s).
# How it fits: Gives us a clear picture of the model’s strengths and weaknesses for improving digit classification.
cm = confusion_matrix(y_train_5, y_train_pred)  # What: Builds the table comparing actual and predicted 5s. Why: Breaks down successes and errors. How it fits: Shows the model’s accuracy details.
print("Confusion Matrix:\n", cm)  # What: Shows the table (e.g., [[correct non-5s, wrong 5s], [missed 5s, correct 5s]]). Why: Lets us see the numbers. How it fits: Helps us analyze performance.

# 🌟 Step 10: Calculating Precision
# What: Computes the precision score for the binary classification task.
# Why: Precision measures how often the model is correct when it predicts a 5, helping us evaluate its reliability.
# How it fits: Ensures the model’s predictions are trustworthy, a key aspect of digit classification.
precision = precision_score(y_train_5, y_train_pred)  # What: Compute precision: TP / (TP + FP). Why: Focuses on accuracy of positive predictions. How it fits: Checks how reliable the 5 predictions are.
print("Precision:", precision)  # What: Display precision (e.g., 0.729 means 72.9% of predicted 5s are correct). Why: Shows the result. How it fits: Gives us a performance metric.

# 🌟 Step 11: Calculating Recall
# What: Computes the recall score for the binary classification task.
# Why: Recall measures how many actual 5s the model detects, showing its ability to find all 5s.
# How it fits: Ensures the model doesn’t miss many 5s, which is important for balanced performance.
recall = recall_score(y_train_5, y_train_pred)  # What: Compute recall: TP / (TP + FN). Why: Measures how complete the detection is. How it fits: Ensures we catch most 5s.
print("Recall:", recall)  # What: Display recall (e.g., 0.756 means 75.6% of actual 5s are detected). Why: Shows the result. How it fits: Provides another performance metric.

# 🌟 Step 12: Calculating F1 Score
# What: Computes the F1 score, which is the harmonic mean of precision and recall.
# Why: F1 score balances precision and recall, providing a single metric to evaluate overall performance.
# How it fits: Gives a comprehensive view of the model’s effectiveness in classifying 5s.
f1 = f1_score(y_train_5, y_train_pred)  # What: Compute F1 score: 2 * (precision * recall) / (precision + recall). Why: Balances the two metrics. How it fits: Offers a single performance number.
print("F1 Score:", f1)  # What: Display F1 score (e.g., 0.742). Why: Shows the result. How it fits: Simplifies performance assessment.

# 🌟 Step 13: Exploring Precision/Recall Trade-Off
# What: Gets decision scores from the SGDClassifier and computes precision/recall for different thresholds.
# Why: Allows us to adjust the threshold to balance precision and recall based on our needs (e.g., prioritizing precision).
# How it fits: Helps fine-tune the model for specific performance criteria, improving its utility for digit classification.
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")  # What: Get decision scores (how "5-like" each image is). Why: Shows confidence levels for predictions. How it fits: Enables threshold adjustment.
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)  # What: Compute precision/recall for various thresholds. Why: Maps out trade-offs. How it fits: Helps choose the best threshold.

# 🌟 Step 14: Plotting Precision and Recall vs. Threshold
# What: Defines a function to plot precision and recall against different thresholds and displays the plot.
# Why: Visualizes the trade-off between precision and recall, helping us choose an optimal threshold.
# How it fits: Enables us to customize the model’s behavior for better digit classification performance.
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):  # What: Define function to plot precision/recall vs. threshold. Why: Organizes the plotting logic. How it fits: Supports visualization.
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")  # What: Plot precision as a blue dashed line. Why: Shows precision trend. How it fits: Visualizes one side of the trade-off.
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")  # What: Plot recall as a green solid line. Why: Shows recall trend. How it fits: Visualizes the other side.
    plt.xlabel("Threshold")  # What: Label x-axis as "Threshold". Why: Clarifies what the x-axis represents. How it fits: Improves chart readability.
    plt.ylabel("Score")  # What: Label y-axis as "Score". Why: Clarifies what the y-axis represents. How it fits: Enhances understanding.
    plt.legend()  # What: Show legend to distinguish precision and recall lines. Why: Identifies each line. How it fits: Makes the chart clear.
    plt.grid(True)  # What: Add a grid for better readability. Why: Helps read values easily. How it fits: Improves usability.
    plt.title("Precision and Recall vs. Threshold")  # What: Set plot title. Why: Describes the chart. How it fits: Provides context.
plt.figure(figsize=(8, 4))  # What: Set plot size to 8x4 inches for clarity. Why: Ensures the chart is readable. How it fits: Prepares the display.
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)  # What: Call the function to create the plot. Why: Generates the visualization. How it fits: Shows the trade-off.
plt.show()  # What: Display the image for us to see. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 15: Plotting Precision vs. Recall Curve
# What: Plots precision against recall to visualize their relationship across thresholds.
# Why: Shows how precision decreases as recall increases, helping us understand the model’s trade-offs.
# How it fits: Provides insight into the model’s performance, aiding in optimization for digit classification.
plt.figure(figsize=(8, 4))  # What: Set plot size to 8x4 inches. Why: Ensures readability. How it fits: Prepares the display.
plt.plot(recalls, precisions, "b-", label="Precision vs. Recall")  # What: Plot precision vs. recall as a blue solid line. Why: Shows the trade-off curve. How it fits: Visualizes performance balance.
plt.xlabel("Recall")  # What: Label x-axis as "Recall". Why: Clarifies x-axis meaning. How it fits: Improves readability.
plt.ylabel("Precision")  # What: Label y-axis as "Precision". Why: Clarifies y-axis meaning. How it fits: Enhances understanding.
plt.legend()  # What: Show legend. Why: Identifies the line. How it fits: Clarifies the chart.
plt.grid(True)  # What: Add a grid for readability. Why: Helps read values. How it fits: Improves usability.
plt.title("Precision vs. Recall Curve")  # What: Set plot title. Why: Describes the chart. How it fits: Provides context.
plt.show()  # What: Display the plot. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 16: Finding a Threshold for 90% Precision
# What: Identifies the lowest threshold that achieves at least 90% precision.
# Why: Allows us to set a strict criterion (high precision) for predicting 5s, even if it lowers recall.
# How it fits: Customizes the model for scenarios where precision is critical, enhancing its utility for specific tasks.
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]  # What: Find the lowest threshold where precision is at least 90%. Why: Selects the strictest point meeting our goal. How it fits: Sets a precision target.
print("Threshold for 90% precision:", threshold_90_precision)  # What: Display the threshold. Why: Shows the value. How it fits: Confirms the choice.

# 🌟 Step 17: Making Predictions with the 90% Precision Threshold
# What: Uses the selected threshold to make new predictions on the training set.
# Why: Tests the model’s performance at the chosen threshold, balancing precision and recall.
# How it fits: Ensures the model meets our precision requirement, improving its reliability for digit classification.
y_train_pred_90 = (y_scores >= threshold_90_precision)  # What: Predict 5 if the score is above the threshold, otherwise not-5. Why: Applies the new rule. How it fits: Tests the adjusted model.

# 🌟 Step 18: Evaluating Precision and Recall at the 90% Threshold
# What: Computes precision and recall for the new predictions made with the 90% precision threshold.
# Why: Verifies that the threshold achieves the desired precision and shows the trade-off with recall.
# How it fits: Confirms the model’s performance under the new threshold, ensuring it meets our classification criteria.
precision_90 = precision_score(y_train_5, y_train_pred_90)  # What: Compute precision at the new threshold. Why: Checks if it’s 90%. How it fits: Validates precision goal.
recall_90 = recall_score(y_train_5, y_train_pred_90)  # What: Compute recall at the new threshold. Why: Shows the trade-off. How it fits: Assesses recall impact.
print("Precision at 90% threshold:", precision_90)  # What: Display precision (should be ~0.90). Why: Confirms result. How it fits: Verifies performance.
print("Recall at 90% threshold:", recall_90)  # What: Display recall (e.g., ~0.437). Why: Shows trade-off. How it fits: Highlights the cost.

# 🌟 Step 19: Computing the ROC Curve
# What: Calculates the False Positive Rate (FPR), True Positive Rate (TPR), and thresholds for the ROC curve using the test set labels.
# Why: The ROC curve evaluates the model’s ability to distinguish 5s from non-5s across different thresholds.
# How it fits: Provides a comprehensive view of the model’s performance, helping us compare it with other models.
fpr, tpr, thresholds = roc_curve(y_test_5, y_scores)  # What: Compute FPR, TPR, and thresholds for ROC curve. Why: Measures separation ability. How it fits: Supports model comparison.

# 🌟 Step 20: Defining and Plotting the ROC Curve
# What: Defines a function to plot the ROC curve and displays it, but the function is incomplete (missing labels and grid).
# Why: The ROC curve visualizes the trade-off between TPR and FPR, helping us assess the model’s overall performance.
# How it fits: Helps us compare models and ensure the classifier is effective for digit recognition.
def plot_roc_curve(fpr, tpr, label=None):  # What: Define function to plot ROC curve. Why: Organizes plotting logic. How it fits: Prepares visualization.
    plt.plot(fpr, tpr, linewidth=2, label=label)  # What: Plot FPR vs. TPR with a thick line. Why: Shows the curve. How it fits: Displays performance.
    plt.plot([0, 1], [0, 1], 'k--')  # What: Plot a dashed diagonal line (random guessing baseline). Why: Shows a comparison point. How it fits: Sets a reference.
    plt.xlabel("False Positive Rate")  # What: Label x-axis as FPR. Why: Clarifies meaning. How it fits: Improves readability.
    plt.ylabel("True Positive Rate (Recall)")  # What: Label y-axis as TPR. Why: Clarifies meaning. How it fits: Enhances understanding.
    plt.grid(True)  # What: Add a grid for readability. Why: Helps read values. How it fits: Improves usability.
    plt.title("ROC Curve")  # What: Set plot title. Why: Describes the chart. How it fits: Provides context.

plot_roc_curve(fpr, tpr)  # What: Call the function to plot the ROC curve for SGDClassifier. Why: Generates the visualization. How it fits: Shows SGD performance.
plt.show()  # What: Display the plot. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 21: Computing the ROC AUC Score
# What: Calculates the Area Under the ROC Curve (AUC) for the SGDClassifier.
# Why: AUC summarizes the ROC curve into a single number (0 to 1), with higher values indicating better performance.
# How it fits: Provides a metric to compare the SGDClassifier with other models, ensuring we choose the best for digit classification.
roc_auc_score(y_train_5, y_scores)  # What: Compute AUC. Why: Quantifies overall performance. How it fits: Supports model selection.

# 🌟 Step 22: Training a RandomForestClassifier for Binary Classification
# What: Trains a RandomForestClassifier to detect 5s and gets probability scores using cross-validation.
# Why: Random Forest is a more powerful model than SGD, often achieving better performance by combining multiple decision trees.
# How it fits: Compares different models to find the best one for classifying 5s, improving overall digit recognition.
forest_clf = RandomForestClassifier(random_state=42)  # What: Initialize Random Forest with a fixed random seed. Why: Ensures consistent results. How it fits: Sets up a stronger model.
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")  # What: Get probabilities via 3-fold cross-validation. Why: Provides confidence scores. How it fits: Tests Random Forest.
y_scores_forest = y_probas_forest[:, 1]  # What: Extract probability of being a 5. Why: Focuses on the positive class. How it fits: Prepares for ROC analysis.

# 🌟 Step 23: Computing the ROC Curve for Random Forest
# What: Calculates FPR, TPR, and thresholds for the Random Forest model’s ROC curve.
# Why: Allows us to plot the ROC curve for Random Forest and compare it with SGDClassifier.
# How it fits: Ensures we select the best model for digit classification by comparing performance metrics.
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)  # What: Compute ROC curve components. Why: Measures separation ability. How it fits: Supports comparison.

# 🌟 Step 24: Plotting ROC Curves for Comparison
# What: Plots the ROC curves of both SGDClassifier and RandomForestClassifier on the same graph.
# Why: Visual comparison helps us see which model performs better at distinguishing 5s from non-5s.
# How it fits: Choosing the best model improves the accuracy of digit classification.
plt.plot(fpr, tpr, "b:", label="SGD")  # What: Plot SGD’s ROC curve as a blue dotted line. Why: Shows SGD performance. How it fits: Provides a baseline.
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")  # What: Plot Random Forest’s ROC curve with label. Why: Shows Random Forest performance. How it fits: Enables comparison.
plt.legend(loc="lower right")  # What: Show legend in the lower right corner. Why: Identifies each curve. How it fits: Clarifies the chart.
plt.show()  # What: Display the plot. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 25: Computing the ROC AUC Score for Random Forest
# What: Calculates the AUC for the RandomForestClassifier.
# Why: Compares the AUC with SGDClassifier to quantify which model is better overall.
# How it fits: A higher AUC indicates a better model for digit classification, guiding our model selection.
roc_auc_score(y_train_5, y_scores_forest)  # What: Compute AUC for Random Forest. Why: Quantifies performance. How it fits: Supports decision-making.

# 🌟 Step 26: Training an SVM for Multiclass Classification
# What: Trains an SVC (Support Vector Classifier) on the full training set to classify all digits (0-9).
# Why: Moves from binary to multiclass classification, tackling the full MNIST problem.
# How it fits: Expands the model to classify any digit, not just 5s, aligning with the ultimate goal of digit recognition.
svm_clf = SVC()  # What: Initialize SVM classifier. Why: Sets up a powerful model. How it fits: Prepares for all-digit classification.
svm_clf.fit(X_train, y_train)  # What: Train on the full training set. Why: Teaches all digit patterns. How it fits: Builds a complete classifier.
svm_clf.predict([some_digit])  # What: Predict the class of the first digit. Why: Tests the model. How it fits: Verifies performance.

# 🌟 Step 27: Exploring SVM Decision Scores
# What: Gets the decision scores for the first digit and inspects the classes.
# Why: Understands how SVM makes decisions by examining scores for each class.
# How it fits: Provides insight into the model’s confidence, helping us trust its predictions for digit classification.
some_digit_scores = svm_clf.decision_function([some_digit])  # What: Get decision scores for each class. Why: Shows confidence levels. How it fits: Analyzes prediction basis.
print(some_digit_scores)  # What: Display scores. Why: Lets us see the numbers. How it fits: Confirms the process.
np.argmax(some_digit_scores)  # What: Find the index of the highest score. Why: Identifies the predicted class. How it fits: Validates the prediction.
svm_clf.classes_  # What: Show all classes the SVM was trained on. Why: Lists the digits (0-9). How it fits: Confirms the scope.
svm_clf.classes_[5]  # What: Access the class at index 5. Why: Checks the fifth digit. How it fits: Verifies the label.

# 🌟 Step 28: Using OneVsRestClassifier with SVM
# What: Trains an SVM using the OneVsRest strategy to handle multiclass classification.
# Why: OneVsRest trains one binary classifier per class, making it easier for SVM to handle 0-9.
# How it fits: Ensures the SVM can classify all digits, improving its applicability for digit recognition.
ovr_clf = OneVsRestClassifier(SVC())  # What: Wrap SVC in OneVsRestClassifier. Why: Enables multiclass handling. How it fits: Prepares for all digits.
ovr_clf.fit(X_train, y_train)  # What: Train on the full training set. Why: Teaches all digit patterns. How it fits: Builds the model.
ovr_clf.predict([some_digit])  # What: Predict the class of the first digit. Why: Tests the model. How it fits: Verifies performance.
len(ovr_clf.estimators_)  # What: Check the number of binary classifiers. Why: Confirms 10 classifiers (one per digit). How it fits: Validates the setup.

# 🌟 Step 29: Training SGDClassifier for Multiclass Classification
# What: Retrains the SGDClassifier on the full training set to classify all digits (0-9).
# Why: Tests SGD’s ability to handle multiclass classification, comparing it with SVM.
# How it fits: Expands the model to classify any digit, aligning with the full MNIST challenge.
sgd_clf.fit(X_train, y_train)  # What: Train SGD on the full training set. Why: Teaches all digit patterns. How it fits: Builds a complete classifier.
sgd_clf.predict([some_digit])  # What: Predict the class of the first digit. Why: Tests the model. How it fits: Verifies performance.
sgd_clf.decision_function([some_digit])  # What: Get decision scores for each class. Why: Shows confidence levels. How it fits: Analyzes prediction basis.

# 🌟 Step 30: Evaluating SGDClassifier with Cross-Validation
# What: Performs 3-fold cross-validation to compute the accuracy of the SGDClassifier on the full training set.
# Why: Measures how often the model correctly classifies digits, ensuring it generalizes well.
# How it fits: Confirms the model’s performance across all digits, a key step in digit recognition.
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")  # What: Compute accuracy for each fold. Why: Tests on different splits. How it fits: Ensures reliability.

# 🌟 Step 31: Scaling the Data and Re-evaluating SGDClassifier
# What: Scales the training data using StandardScaler and re-evaluates the SGDClassifier.
# Why: Scaling normalizes pixel values to improve SGD’s performance by ensuring all features are on the same scale.
# How it fits: Boosts the model’s accuracy, making it more effective for digit classification.
scalar = StandardScaler()  # What: Initialize StandardScaler to normalize features. Why: Prepares the tool. How it fits: Sets up scaling.
X_train_scaled = scalar.fit_transform(X_train.astype(np.float64))  # What: Scale training data (mean=0, std=1). Why: Improves model learning. How it fits: Enhances performance.
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")  # What: Recompute accuracy. Why: Tests scaled data. How it fits: Confirms improvement.

# 🌟 Step 32: Computing the Confusion Matrix for Multiclass Classification
# What: Gets predictions for all digits and computes a 10x10 confusion matrix.
# Why: Shows where the model confuses digits (e.g., 3 vs. 8), helping us identify areas for improvement.
# How it fits: Provides a detailed view of the model’s performance across all digits, essential for digit recognition.
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)  # What: Get predictions for all digits. Why: Tests on unseen splits. How it fits: Ensures fair evaluation.
conf_mx = confusion_matrix(y_train, y_train_pred)  # What: Compute 10x10 confusion matrix. Why: Breaks down errors. How it fits: Analyzes performance.
print(conf_mx)  # What: Display the matrix. Why: Shows the numbers. How it fits: Allows inspection.

# 🌟 Step 33: Visualizing the Confusion Matrix
# What: Plots the confusion matrix as a heatmap to visualize correct and incorrect predictions.
# Why: A heatmap makes it easier to see which digits are confused, highlighting patterns of errors.
# How it fits: Helps us understand the model’s weaknesses, guiding improvements in digit classification.
plt.matshow(conf_mx, cmap=plt.cm.gray)  # What: Plot the matrix using a grayscale colormap. Why: Shows values visually. How it fits: Highlights errors.
plt.title("Confusion Matrix for All Digits")  # What: Set plot title. Why: Describes the chart. How it fits: Provides context.
plt.show()  # What: Display the heatmap. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 34: Normalizing the Confusion Matrix to Compare Error Rates
# What: Normalizes the confusion matrix by dividing each row by its sum and sets the diagonal to 0 to focus on errors.
# Why: Shows the proportion of misclassifications, making it easier to identify which digits are most confused.
# How it fits: Pinpoints specific errors, helping us improve the model for digit recognition.
row_sums = conf_mx.sum(axis=1, keepdims=True)  # What: Compute the sum of each row. Why: Gets total instances per class. How it fits: Prepares for normalization.
norm_conf_mx = conf_mx / row_sums  # What: Normalize by dividing each element by its row sum. Why: Turns counts into proportions. How it fits: Focuses on error rates.
np.fill_diagonal(norm_conf_mx, 0)  # What: Set diagonal to 0. Why: Ignores correct predictions to focus on errors. How it fits: Highlights mistakes.

# 🌟 Step 35: Visualizing the Normalized Confusion Matrix
# What: Plots the normalized confusion matrix as a heatmap to highlight errors.
# Why: Visualizes error rates, making it clear which digits are most often mistaken for others.
# How it fits: Identifies specific areas for improvement, enhancing the model’s accuracy for digit classification.
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)  # What: Plot the normalized matrix. Why: Shows error proportions. How it fits: Visualizes mistakes.
plt.title("Normalized Confusion Matrix (Errors)")  # What: Set plot title. Why: Describes the chart. How it fits: Provides context.
plt.show()  # What: Display the heatmap. Why: Shows the result. How it fits: Allows analysis.

# 🌟 Step 36: Visualizing Specific Errors (3 vs. 5)
# What: Selects examples of correct and incorrect predictions for digits 3 and 5, preparing to visualize them.
# Why: Visualizing examples of errors helps us understand why the model confuses certain digits.
# How it fits: Provides insight into the model’s mistakes, guiding us to improve digit classification.
cl_a, cl_b = 3, 5  # What: Define two digits to compare: 3 and 5. Why: Focuses on a common confusion pair. How it fits: Targets specific errors.
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]  # What: Correct predictions: actual 3, predicted 3. Why: Shows successful cases. How it fits: Highlights correct classifications.
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]  # What: Errors: actual 3, predicted 5. Why: Shows one type of mistake. How it fits: Reveals misclassifications.
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]  # What: Errors: actual 5, predicted 3. Why: Shows the reverse mistake. How it fits: Completes the error analysis.
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]  # What: Correct predictions: actual 5, predicted 5. Why: Shows successful cases. How it fits: Balances the comparison.

# 🌟 Step 37: Plotting Specific Errors (3 vs. 5)
# What: Plots a 2x2 grid of images showing correct and incorrect predictions for digits 3 and 5.
# Why: Visual comparison of correct and incorrect predictions reveals why the model confuses 3s and 5s.
# How it fits: Understanding errors helps us improve the model, ensuring better digit recognition.
def plot_digits(instances, images_per_row=5):  # What: Define function to plot multiple digits in a grid. Why: Organizes the plotting. How it fits: Enables visualization.
    images_per_row = min(len(instances), images_per_row)  # What: Ensure we don’t exceed the number of instances. Why: Avoids errors. How it fits: Controls display.
    images = [instance.reshape(28, 28) for instance in instances]  # What: Reshape each instance to 28x28. Why: Matches image shape. How it fits: Prepares for plotting.
    n_rows = (len(instances) - 1) // images_per_row + 1  # What: Compute number of rows needed. Why: Determines grid size. How it fits: Structures the layout.
    row_images = []  # What: List to store rows of images. Why: Builds the grid step-by-step. How it fits: Organizes the display.
    n_empty = n_rows * images_per_row - len(instances)  # What: Number of empty slots in the grid. Why: Accounts for incomplete rows. How it fits: Fills the grid.
    images.append(np.zeros((28, 28 * n_empty)))  # What: Pad with empty images if needed. Why: Completes the grid. How it fits: Ensures proper sizing.
    for row in range(n_rows):  # What: Loop through rows. Why: Builds each row of images. How it fits: Constructs the grid.
        rimages = images[row * images_per_row: (row + 1) * images_per_row]  # What: Select images for this row. Why: Groups images. How it fits: Arranges the layout.
        row_images.append(np.concatenate(rimages, axis=1))  # What: Concatenate images horizontally. Why: Forms a row. How it fits: Builds the row image.
    image = np.concatenate(row_images, axis=0)  # What: Concatenate rows vertically. Why: Forms the full grid. How it fits: Completes the image.
    plt.imshow(image, cmap="binary")  # What: Display the grid of images in black-and-white. Why: Shows the digits clearly. How it fits: Visualizes the data.
    plt.axis("off")  # What: Hide axes for a cleaner plot. Why: Focuses on the images. How it fits: Enhances clarity.

plt.figure(figsize=(8, 8))  # What: Set figure size to 8x8 inches. Why: Ensures readability. How it fits: Prepares the display.
plt.subplot(221)  # What: Top-left: 3s predicted as 3s (correct). Why: Starts the grid. How it fits: Positions the first plot.
plot_digits(X_aa[:25], images_per_row=5)  # What: Plot first 25 correct 3s. Why: Shows successful cases. How it fits: Visualizes correct predictions.
plt.title("True 3s, Predicted 3s")  # What: Set subplot title. Why: Describes the plot. How it fits: Provides context.
plt.subplot(222)  # What: Top-right: 3s predicted as 5s (errors). Why: Continues the grid. How it fits: Positions the second plot.
plot_digits(X_ab[:25], images_per_row=5)  # What: Plot first 25 misclassified 3s. Why: Shows one error type. How it fits: Visualizes mistakes.
plt.title("True 3s, Predicted 5s")  # What: Set subplot title. Why: Describes the plot. How it fits: Provides context.
plt.subplot(223)  # What: Bottom-left: 5s predicted as 3s (errors). Why: Continues the grid. How it fits: Positions the third plot.
plot_digits(X_ba[:25], images_per_row=5)  # What: Plot first 25 misclassified 5s. Why: Shows the reverse error. How it fits: Visualizes mistakes.
plt.title("True 5s, Predicted 3s")  # What: Set subplot title. Why: Describes the plot. How it fits: Provides context.
plt.subplot(224)  # What: Bottom-right: 5s predicted as 5s (correct). Why: Completes the grid. How it fits: Positions the fourth plot.
plot_digits(X_bb[:25], images_per_row=5)  # What: Plot first 25 correct 5s. Why: Shows successful cases. How it fits: Visualizes correct predictions.
plt.title("True 5s, Predicted 5s")  # What: Set subplot title. Why: Describes the plot. How it fits: Provides context.
plt.show()  # What: Display the 2x2 grid of plots. Why: Shows the result. How it fits: Allows error analysis.

# 🌟 Step 38: Creating Multilabel Labels
# What: Creates two binary labels for each digit: whether it’s greater than 5 and whether it’s odd.
# Why: Introduces multilabel classification, where each digit can have multiple labels (e.g., 7 is both >5 and odd).
# How it fits: Expands your skills to handle more complex classification tasks, preparing you for real-world scenarios.
y_train_large = (y_train > 5)  # What: True if digit is >5 (6, 7, 8, 9). Why: Creates one label. How it fits: Adds a property to classify.
y_train_odd = (y_train % 2 == 1)  # What: True if digit is odd (1, 3, 5, 7, 9). Why: Creates another label. How it fits: Adds a second property.
y_multilabel = np.c_[y_train_large, y_train_odd]  # What: Combine into a matrix with two labels. Why: Forms multilabel data. How it fits: Prepares for multilabel task.

# 🌟 Step 39: Training a KNN Classifier for Multilabel Classification
# What: Trains a KNeighborsClassifier to predict both labels (>5 and odd) for each digit.
# Why: KNN is suitable for multilabel tasks, allowing us to predict multiple labels simultaneously.
# How it fits: Enhances your ability to handle complex classification problems, a valuable skill for digit recognition applications.
knn_clf = KNeighborsClassifier()  # What: Initialize KNN classifier. Why: Sets up the model. How it fits: Prepares for multilabel prediction.
knn_clf.fit(X_train, y_multilabel)  # What: Train on training data with multilabel targets. Why: Teaches both labels. How it fits: Builds the classifier.

# 🌟 Step 40: Making a Multilabel Prediction
# What: Predicts the multilabel classification for the first digit (a 5).
# Why: Tests the KNN classifier’s ability to predict multiple labels for a single image.
# How it fits: Confirms the model can handle multilabel tasks, expanding its utility for digit classification.
knn_clf.predict([some_digit])  # What: Predict labels for the first digit. Why: Tests the model. How it fits: Verifies multilabel capability.

# 🌟 Step 41: Evaluating the Multilabel Classifier with F1 Score
# What: Computes the F1 score for the KNN classifier’s multilabel predictions using cross-validation.
# Why: F1 score measures the model’s performance across both labels, with “macro” averaging treating each label equally.
# How it fits: Ensures the multilabel classifier performs well, validating its effectiveness for digit classification.
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)  # What: Get predictions using 3-fold cross-validation. Why: Ensures fair testing. How it fits: Evaluates performance.
f1_score(y_multilabel, y_train_knn_pred, average="macro")  # What: Compute macro-averaged F1 score. Why: Balances both labels. How it fits: Assesses overall accuracy.

# 🌟 Step 42: Setting Up a Multioutput Classification Task (Denoising)
# What: Adds random noise to the images and sets up a task to predict the original (clean) pixel values.
# Why: Introduces multioutput classification, where each pixel is an output, teaching the model to denoise images.
# How it fits: Expands your skills to handle complex tasks like image denoising, a practical application of digit recognition.
noise = np.random.randint(0, 100, (len(X_train), 784))  # What: Generate random noise for each pixel in training set. Why: Simulates corruption. How it fits: Creates a challenge.
X_train_mod = X_train + noise  # What: Add noise to training images. Why: Makes data noisy. How it fits: Sets up the denoising task.
noise = np.random.randint(0, 100, (len(X_test), 784))  # What: Generate noise for test set. Why: Simulates corruption. How it fits: Extends the task.
X_test_mod = X_test + noise  # What: Add noise to test images. Why: Makes data noisy. How it fits: Prepares for testing.
y_train_mod = X_train  # What: Target is the original training images. Why: Defines the clean output. How it fits: Sets the goal.
y_test_mod = X_test  # What: Target is the original test images. Why: Defines the clean output. How it fits: Sets the goal.

# 🌟 Step 43: Training the KNN Classifier for Multioutput Classification
# What: Trains the KNN classifier to predict the clean pixel values from noisy images.
# Why: KNN can predict multiple outputs (each pixel), effectively denoising the images.
# How it fits: Demonstrates a practical application of classification, enhancing your ability to handle diverse ML tasks.
knn_clf.fit(X_train_mod, y_train_mod)  # What: Train KNN on noisy images and clean images. Why: Teaches denoising. How it fits: Builds the model.

# 🌟 Step 44: Denoising an Image and Visualizing the Result
# What: Denoises a test image and plots the result, but some_index and plot_digit() are undefined.
# Why: Tests the model’s ability to remove noise, showing its effectiveness in a multioutput task.
# How it fits: Validates the model’s performance in a real-world scenario, preparing you for advanced ML applications.
some_index = 0  # What: Choose the first test image for denoising. Why: Picks a sample. How it fits: Starts the test.
def plot_digit(digit):  # What: Define function to plot a single digit. Why: Organizes the plotting. How it fits: Enables visualization.
    image = digit.reshape(28, 28)  # What: Reshape the 784-pixel array to 28x28. Why: Matches image shape. How it fits: Prepares for display.
    plt.imshow(image, cmap="binary")  # What: Display in black-and-white. Why: Shows the digit clearly. How it fits: Visualizes the result.
    plt.axis("off")  # What: Hide axes. Why: Focuses on the image. How it fits: Enhances clarity.
    plt.title("Denoised Digit")  # What: Set title. Why: Describes the plot. How it fits: Provides context.

clean_digit = knn_clf.predict([X_test_mod[some_index]])  # What: Predict the clean pixel values. Why: Denoises the image. How it fits: Tests the model.
plot_digit(clean_digit[0])  # What: Plot the denoised digit. Why: Shows the result. How it fits: Visualizes the output.
plt.show()  # What: Display the plot. Why: Shows the denoised image. How it fits: Allows inspection.