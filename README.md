# Multi-Class ECG Arrhythmia Classification with Bidirectional GRU

## Project Summary

This project focuses on multi-class classification of ECG signals into five classes:

* Normal
* S-type
* V-type
* F-type
* Unknown

The model is trained on the **MIT-BIH Arrhythmia Database** and uses a recurrent neural network architecture based on **Bidirectional GRU** to learn temporal patterns from ECG signal sequences.

The project includes:

* ECG signal loading and preprocessing
* Class distribution analysis
* Class balancing using resampling
* Transformation of ECG signals into sequential input format
* Bidirectional GRU-based multi-class classification
* Softmax-based classification
* Model training with validation
* Early stopping and learning-rate reduction
* Classification report evaluation
* Confusion matrix visualization

---

## Dataset

The project uses the **MIT-BIH Arrhythmia Database**.

The dataset is divided into training and test files:

```text
mitbih_train.csv
mitbih_test.csv
```

Each sample contains **187 ECG signal values**, followed by its class label.

### Classes

| Class   | Label |
| ------- | ----: |
| Normal  |     0 |
| S-type  |     1 |
| V-type  |     2 |
| F-type  |     3 |
| Unknown |     4 |

### Original Class Distribution

The original training and test datasets are highly imbalanced. The training set contains:

| Class   | Training Samples |
| ------- | ---------------: |
| Normal  |           72,471 |
| S-type  |            2,223 |
| V-type  |            5,788 |
| F-type  |              641 |
| Unknown |            6,431 |

The test set contains:

| Class   | Test Samples |
| ------- | -----------: |
| Normal  |       18,118 |
| S-type  |          556 |
| V-type  |        1,448 |
| F-type  |          162 |
| Unknown |        1,608 |

---

## Dataset Preparation

Because of the significant class imbalance in the training data, resampling was applied to obtain an equal number of samples for each class.

A target of **20,000 samples per class** was used.

The Normal class was sampled without replacement, while the remaining classes were upsampled with replacement.

After balancing:

| Class   | Balanced Training Samples |
| ------- | ------------------------: |
| Normal  |                    20,000 |
| S-type  |                    20,000 |
| V-type  |                    20,000 |
| F-type  |                    20,000 |
| Unknown |                    20,000 |

Total training samples after balancing:

**100,000**

The final input shape is:

```text
(100000, 187, 1)
```

Each ECG sample is therefore represented as a sequence of 187 time steps with one feature per time step.
The test dataset remains separate from the balancing process.

---

## Model Architecture

The classifier is based on a recurrent neural network architecture using **Bidirectional GRU**.

### Architecture

```text
Input ECG Signal
      │
      ▼
Masking
      │
      ▼
Bidirectional GRU (64 units)
      │
      ▼
GRU (64 units)
      │
      ▼
Dense (64, ReLU)
      │
      ▼
Dropout (0.2)
      │
      ▼
Dense (5, Softmax)
      │
      ▼
Five ECG Classes
```

The model uses a Masking layer followed by a Bidirectional GRU with 64 units and a second GRU layer with 64 units. The extracted representation is then passed through a Dense layer, Dropout, and a final five-neuron Softmax layer for multi-class classification.

### Model Configuration

| Parameter         | Value                     |
| ----------------- | ------------------------- |
| Input shape       | 187 × 1                   |
| Masking value     | 0.0                       |
| Bidirectional GRU | 64 units                  |
| GRU               | 64 units                  |
| Dense             | 64 units, ReLU            |
| Dropout           | 0.2                       |
| Output            | 5 neurons, Softmax        |
| Optimizer         | Adam                      |
| Loss              | Categorical Cross-Entropy |
| Evaluation Metric | Accuracy                  |

The model is compiled using the Adam optimizer with categorical cross-entropy loss.

---

## Training

The model was trained using the following configuration:

| Parameter        |                     Value |
| ---------------- | ------------------------: |
| Maximum Epochs   |                        30 |
| Batch Size       |                       128 |
| Validation Split |                       15% |
| Optimizer        |                      Adam |
| Loss Function    | Categorical Cross-Entropy |

Two callbacks were used during training:

* **EarlyStopping** with a patience of 5 epochs and restoration of the best model weights
* **ReduceLROnPlateau** with a factor of 0.5 and a patience of 3 epochs

The learning rate was progressively reduced during training when the validation loss stopped improving.

Training was performed for the full 30 epochs. At the final epoch, the training accuracy reached **99.24%**, while the validation accuracy reached **97.47%**.

---

## Results

The final model was evaluated on the held-out test set containing **21,892 ECG samples**.

### Classification Performance

| Class   | Precision | Recall | F1-Score | Support |
| ------- | --------: | -----: | -------: | ------: |
| Normal  |      0.99 |   0.97 |     0.98 |  18,118 |
| S-type  |      0.59 |   0.84 |     0.69 |     556 |
| V-type  |      0.88 |   0.96 |     0.92 |   1,448 |
| F-type  |      0.52 |   0.86 |     0.65 |     162 |
| Unknown |      0.97 |   0.96 |     0.96 |   1,608 |

### Overall Metrics

* **Test Accuracy:** 96%
* **Macro F1-Score:** 0.84
* **Weighted F1-Score:** 0.96

The classification report is generated directly from predictions on the test set.

The notebook also generates a **5 × 5 confusion matrix** to visualize the classification results across the five ECG classes.

---

## Trained Model

The trained model is saved as:

```text
models/ECG_Binary_Classifier_97.h5
```

## Loading the Model
``` python 
from tensorflow.keras.models import load_model

model = load_model("models/ECG_Binary_Classifier_97.h5")
```

The saved model can then be used for inference on ECG samples with the same input format used during training:

```text
(187, 1)
```



## Prediction

The trained model produces a probability distribution over the five classes using the Softmax output layer.

The predicted class is obtained by selecting the class with the highest predicted probability:

```python
y_pred = np.argmax(model.predict(X_test), axis=1)
```

The output classes are:

```text
0 → Normal
1 → S-type
2 → V-type
3 → F-type
4 → Unknown
```

This prediction procedure and class mapping are implemented in the notebook.

---

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

---

## Key Concepts

This project demonstrates the application of deep learning to sequential biomedical signals, with a focus on:

* ECG signal classification
* Multi-class classification
* Time-series modeling
* Recurrent Neural Networks
* Bidirectional GRU
* GRU-based sequence modeling
* Class imbalance handling
* Resampling and upsampling
* Softmax classification
* Classification metrics
* Confusion matrix analysis
* Validation and learning-rate scheduling

---

## Course Information

**Deep Learning Course**
**University of Isfahan**

This project was developed as part of the coursework for studying deep learning methods and their application to sequential biomedical signal classification.
