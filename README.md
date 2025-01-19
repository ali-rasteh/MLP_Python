# MLP_Python

This project contains an implementation of a Multi-Layer Perceptron (MLP) in Python. It includes code to train and evaluate the MLP on several datasets.

## Project Structure

- **MLP Codes/**: Contains the main code files for the project.
  - `functions.py`: Includes functions for training the MLP model, calculating accuracy, and reading datasets.
  - `main.py`: Main script to run the training process on different datasets.
  - `mlp.py`: Defines the MLP class and its methods for forward and backward propagation.
- **datasets/**: Contains the datasets used for training and evaluation.
  - `Balance.txt`: Balance Scale dataset.
  - `Banana.txt`: Banana dataset.
  - `Haberman's Survival.txt`: Haberman's Survival dataset.
  - Other datasets also used in the project but not listed in this directory.

## How to Run

1. Ensure you have the required dependencies installed:
   - numpy
   - scikit-learn

2. Run the main script to start training the MLP on the datasets:
   ```sh
   python MLP\ Codes/main.py
   ```

## Datasets

The project uses the following datasets for training and evaluation:
- **Banana**: An artificial dataset with banana-shaped clusters.
- **Haberman**: Contains cases from a study on the survival of patients who had undergone surgery for breast cancer.
- **Titanic**: Dataset of Titanic passengers.
- **Balance**: Models psychological experimental results with balance scale data.
- **Hayes-roth**: Dataset with psychological data.
- **Newthyroid**: Dataset on thyroid disease.
- **Wine**: Dataset on wine classification.

## Usage

The main script (`main.py`) initializes and trains the MLP model on each dataset, printing the training and test accuracies.

## License

This project is licensed under the MIT License.
