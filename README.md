# Handwritten Digit Recognition Using Machine Learning

A beginner-friendly mini-project that recognizes a handwritten digit (0–9) drawn by a user. It uses the MNIST dataset and a Convolutional Neural Network (CNN), with a Streamlit web interface.

## 1. What the project does

1. The user draws a digit on the canvas.
2. The app converts the drawing to grayscale and resizes it to 28 × 28 pixels.
3. Pixel values are normalized to the range 0–1.
4. A trained CNN predicts the digit.
5. The app displays the predicted digit, confidence, and probability for each class.

## 2. Software requirements

- Windows 10/11, Linux, or macOS
- Python 3.10 or 3.11 (recommended)
- VS Code (recommended) or another Python IDE
- Internet connection for the first training run (to download MNIST)

## 3. Installation (Windows / VS Code)

1. Extract the ZIP file.
2. Open the extracted `Handwritten_Digit_Recognition_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```bash
   py -3.10 -m venv .venv
   ```

5. Activate it:

   ```bash
   .venv\\Scripts\\activate
   ```

6. Install the libraries:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the app:

   ```bash
   streamlit run app.py
   ```

The first launch downloads the MNIST dataset and trains the CNN for five epochs. This can take a few minutes. The trained model is saved in `models/mnist_cnn.keras`; later launches reuse it.

### Linux / macOS activation

Use `source .venv/bin/activate` instead of the Windows activation command.

## 4. Train the model separately (optional)

Run:

```bash
python train.py
```

This trains the model and prints test loss and test accuracy. The app will use the saved model.

## 5. Project structure

```text
Handwritten_Digit_Recognition_Project/
├── app.py                 # Streamlit user interface and prediction
├── train.py               # MNIST loading, CNN training and evaluation
├── requirements.txt       # Python dependencies
├── README.md              # Installation and usage guide
├── PROJECT_REPORT.md      # Editable report text
└── models/                # Created automatically after training
    └── mnist_cnn.keras
```

## 6. How to demonstrate it

1. Start the app using `streamlit run app.py`.
2. A browser tab opens with the drawing canvas.
3. Draw a clear digit such as 3, 5, or 8 using the mouse or touchpad.
4. Click **Predict digit**.
5. Show the predicted number and probability chart.
6. Click **Clear canvas** and test another digit.

For a reliable classroom demo, test the app before the presentation and keep the laptop connected to power and the internet for the first training run.

## 7. Important notes

- This is a working academic prototype, not a production handwriting-recognition system.
- The MNIST dataset contains grayscale images of isolated handwritten digits, each 28 × 28 pixels.
- The model may misclassify unusual handwriting, very thin strokes, or drawings placed near the edge.
- The reported accuracy must be taken from your own run of `train.py`. Do not copy an assumed accuracy into your report.
- Keep your code, report, and screenshots. Be prepared to explain the dataset, preprocessing, CNN layers, training, and prediction process.
