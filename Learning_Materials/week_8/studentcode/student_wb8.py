from approvedimports import *

def make_xor_reliability_plot(train_X, train_y):
    """Function to investigate the effect of hidden layer size on MLP performance for XOR problem.
    
    Parameters:
    -----------
    train_X: numpy array
        The input examples.
    train_y: numpy array
        The target labels.
    
    Returns:
    --------
    fig: matplotlib figure object
        The figure containing the plots.
    axs: array of matplotlib axes objects
        The axes containing the two plots (shape (1, 2)).
    """
    # Initialize lists to store results
    hidden_sizes = range(1, 11)  # Hidden layer sizes from 1 to 10 (inclusive)
    success_rates = []
    mean_epochs_successful = []
    num_runs = 10  # Number of runs per hidden layer size (corrected to 10)

    # For each hidden layer size
    for num_nodes in hidden_sizes:
        successful_runs = 0
        epochs_successful = []

        # Perform 10 runs with different random states
        for run in range(num_runs):
            # Create MLP with specified architecture
            xorMLP = MLPClassifier(
                hidden_layer_sizes=(num_nodes,),
                max_iter=1000,
                alpha=1e-4,
                solver="sgd",
                learning_rate_init=0.1,
                random_state=run  # Set random_state for reproducibility
            )
            
            # Train the model
            xorMLP.fit(train_X, train_y)
            
            # Measure accuracy
            accuracy = xorMLP.score(train_X, train_y)
            
            # Check if the run was successful (100% accuracy)
            if np.isclose(accuracy, 1.0, atol=1e-2):  # Allow small numerical tolerance
                successful_runs += 1
                epochs_successful.append(xorMLP.n_iter_)
        
        # Calculate success rate (reliability)
        success_rate = successful_runs / num_runs
        success_rates.append(success_rate)
        
        # Calculate mean epochs for successful runs (efficiency)
        if successful_runs > 0:
            mean_epochs = np.mean(epochs_successful)
        else:
            mean_epochs = 1000  # Default value if no successful runs
        mean_epochs_successful.append(mean_epochs)

    # Create the plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left plot: Reliability
    axs[0].plot(hidden_sizes, success_rates, marker='o')
    axs[0].set_title("Reliability")
    axs[0].set_xlabel("Hidden Layer Width")  # Corrected x-axis label
    axs[0].set_ylabel("Success Rate")  # Corrected y-axis label
    axs[0].set_xticks(list(hidden_sizes))
    axs[0].set_ylim(0, 1)  # Success rate between 0 and 1
    
    # Right plot: Efficiency
    axs[1].plot(hidden_sizes, mean_epochs_successful, marker='o', color='orange')
    axs[1].set_title("Efficiency")
    axs[1].set_xlabel("Hidden Layer Width")  # Corrected x-axis label
    axs[1].set_ylabel("Mean epochs")  # Corrected y-axis label
    axs[1].set_xticks(list(hidden_sizes))
    axs[1].set_ylim(0, 1000)  # Efficiency capped at 1000
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig, axs

# Make sure you have the packages needed
from approvedimports import *

# This is the class to complete where indicated
class MLComparisonWorkflow:
    """Class to implement a basic comparison of supervised learning algorithms on a dataset.""" 
    
    def __init__(self, datafilename: str, labelfilename: str):
        """Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.
        
        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models: dict = {"KNN": [], "DecisionTree": [], "MLP": []}
        self.best_model_index: dict = {"KNN": 0, "DecisionTree": 0, "MLP": 0}
        self.best_accuracy: dict = {"KNN": 0, "DecisionTree": 0, "MLP": 0}

        # Load the data and labels
        self.data_x = np.genfromtxt(datafilename, delimiter=",")
        self.data_y = np.genfromtxt(labelfilename, delimiter=",")

    def preprocess(self):
        """Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if there are more than 2 classes

           Remember to set random_state = 12345 if you use train_test_split()
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        # Train/test split (70:30, stratified)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, random_state=12345, stratify=self.data_y
        )

        # Normalize features
        scaler = MinMaxScaler()
        self.train_x = scaler.fit_transform(self.train_x)
        self.test_x = scaler.transform(self.test_x)

        # One-hot encoding for MLP if 3 or more classes
        num_classes = len(np.unique(self.data_y))
        if num_classes >= 3:
            self.train_y_mlp = np.zeros((len(self.train_y), num_classes))
            self.test_y_mlp = np.zeros((len(self.test_y), num_classes))
            for i in range(len(self.train_y)):
                self.train_y_mlp[i, int(self.train_y[i])] = 1
            for i in range(len(self.test_y)):
                self.test_y_mlp[i, int(self.test_y[i])] = 1
        else:
            self.train_y_mlp = self.train_y
            self.test_y_mlp = self.test_y
    
    def run_comparison(self):
        """Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.
        
        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        """
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier

        # KNN: Test different n_neighbors
        for n_neighbors in [1, 3, 5, 7, 9]:
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(self.train_x, self.train_y)
            self.stored_models["KNN"].append(model)
            accuracy = 100 * model.score(self.test_x, self.test_y)
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = len(self.stored_models["KNN"]) - 1

        # Decision Tree: Test combinations of max_depth, min_samples_split, min_samples_leaf
        for max_depth in [1, 3, 5]:
            for min_samples_split in [2, 5, 10]:
                for min_samples_leaf in [1, 5, 10]:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y)
                    self.stored_models["DecisionTree"].append(model)
                    accuracy = 100 * model.score(self.test_x, self.test_y)
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = len(self.stored_models["DecisionTree"]) - 1

        # MLP: Test combinations of hidden layer sizes and activation functions
        for first_layer in [2, 5, 10]:
            for second_layer in [0, 2, 5]:
                for activation in ["logistic", "relu"]:
                    hidden_layer_sizes = (first_layer,) if second_layer == 0 else (first_layer, second_layer)
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        max_iter=1000,
                        alpha=1e-4,
                        solver="adam",
                        random_state=12345
                    )
                    model.fit(self.train_x, self.train_y_mlp)
                    self.stored_models["MLP"].append(model)
                    accuracy = 100 * model.score(self.test_x, self.test_y_mlp)
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = len(self.stored_models["MLP"]) - 1
    
    def report_best(self):
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN", "DecisionTree", or "MLP"
        
        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        best_acc = max(self.best_accuracy.values())
        best_alg = max(self.best_accuracy, key=self.best_accuracy.get)
        best_model = self.stored_models[best_alg][self.best_model_index[best_alg]]
        return best_acc, best_alg, best_model