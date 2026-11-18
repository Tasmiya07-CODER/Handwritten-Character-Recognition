# Handwritten Character Recognition Using Machine Learning

A beginner-friendly mini-project that recognizes a handwritten character drawn by a user. It supports **digits (0–9)** and **uppercase letters (A–Z)**, giving **36 classes** in total. It uses the MNIST and EMNIST datasets and a Convolutional Neural Network (CNN), with a Streamlit web interface.

## 1. What the project does

1. The user draws a digit or a letter on the canvas.
2. The app converts the drawing to grayscale and resizes it to 28 × 28 pixels.
3. Pixel values are normalized to the range 0–1.
4. A trained CNN predicts the character.
5. The app displays the predicted character, confidence, and probability for each class.

### Supported characters

| Type | Characters | Number of classes |
|------|------------|-------------------|
| Digits | 0 1 2 3 4 5 6 7 8 9 | 10 |
| Uppercase letters | A B C … Z | 26 |
| **Total** | | **36** |

## 2. Software requirements

- Windows 10/11, Linux, or macOS
- Python 3.10 or 3.11 (recommended)
- VS Code (recommended) or another Python IDE
- Internet connection for the first training run (to download the datasets)

## 3. Installation (Windows / VS Code)

1. Extract the ZIP file.
2. Open the extracted `Handwritten_Character_Recognition_Project` folder in VS Code.
3. Open Terminal → New Terminal.
4. Create a virtual environment:

   ```bash
   py -3.10 -m venv .venv
   ```

5. Activate it:

   ```bash
   .venv\Scripts\activate
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

The first launch downloads the datasets and trains the CNN. Because the dataset now includes letters as well as digits, this takes longer than the digit-only version, so allow some extra time. The trained model is saved in `models/char_cnn.keras`; later launches reuse it.

### Linux / macOS activation

Use `source .venv/bin/activate` instead of the Windows activation command.

## 4. Train the model separately (optional)

Run:

```bash
python train.py
```

This loads the datasets, trains the model, and prints test loss and test accuracy. The app will use the saved model.

To retrain from scratch, delete the file inside the `models/` folder and run `train.py` again.

## 5. Project structure

```text
Handwritten_Character_Recognition_Project/
├── app.py                 # Streamlit user interface and prediction
├── train.py               # Dataset loading, CNN training and evaluation
├── requirements.txt       # Python dependencies
├── README.md              # Installation and usage guide
├── PROJECT_REPORT.md      # Editable report text
└── models/                # Created automatically after training
    └── char_cnn.keras
```

## 6. How to demonstrate it

1. Start the app using `streamlit run app.py`.
2. A browser tab opens with the drawing canvas.
3. Draw a clear character using the mouse or touchpad, for example 3, 8, A, or K.
4. Click **Predict character**.
5. Show the predicted character and probability chart.
6. Click **Clear canvas** and test another character.

For a reliable classroom demo, test the app before the presentation and keep the laptop connected to power and the internet for the first training run.

**Tips for better results**

- Draw large, centered characters with thick, clear strokes.
- Draw only **uppercase** letters.
- Draw one character at a time.

## 7. Known limitations

- Some characters look very similar when handwritten, so the model may confuse them. Common pairs are **0 / O**, **1 / I**, **2 / Z**, **5 / S**, and **8 / B**.
- Unusual handwriting, very thin strokes, or drawings placed near the edge may be misclassified.
- Lowercase letters, words, and multiple characters in one drawing are not supported.

## 8. Important notes

- This is a working academic prototype, not a production handwriting-recognition system.
- The training images are grayscale, isolated characters, each 28 × 28 pixels.
- The reported accuracy must be taken from your own run of `train.py`. Do not copy an assumed accuracy into your report.
- Keep your code, report, and screenshots. Be prepared to explain the datasets, preprocessing, CNN layers, training, and prediction process.
