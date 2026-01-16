# AbacaGrade

AbacaGrade is a Python application for classifying abaca fiber grades using machine learning and image processing. The app includes a GUI and pre-trained models for fiber analysis.

---

## **Project Structure**

AbacaGrade/
├── images/ # Image assets
├── abaca_svm_model_pca.pkl # Pre-trained SVM model
├── classes.json # Class labels
├── gui_abaca.py # Main GUI application
├── gui_abaca - Copy.py # Backup/testing file
├── pca.pkl # PCA model
├── requirements.txt # Python dependencies
├── scaler.pkl # Scaler for preprocessing
└── README.md # Project documentation


> ⚠ **Do not include the `venv/` folder.** The virtual environment is system-specific.

---

## **Python Version**

This project was developed and tested using:

Python 3.10.9


It is recommended to use the same Python version to avoid compatibility issues.

Check your Python version with:

```bash
python --version

or on macOS / Linux:
python3 --version

Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/bunnywkwk/AbacaGrade.git
cd AbacaGrade

2️⃣ Create a virtual environment
On Windows:
python -m venv venv

On macOS / Linux:
python3 -m venv venv

3️⃣ Activate the virtual environment
On Windows:
venv\Scripts\activate

On macOS / Linux:
source venv/bin/activate


After activation, your terminal should show (venv) at the beginning.

4️⃣ Install dependencies
pip install -r requirements.txt


This installs all packages required to run the application.

5️⃣ Run the application
python gui_abaca.py


Make sure you run this inside the activated virtual environment.

6️⃣ Deactivate the virtual environment (optional)
deactivate

Notes

All necessary model files (.pkl) and classes.json must be in the project folder.

If you add new packages later, update requirements.txt:

pip freeze > requirements.txt


Python 3.10.9 is recommended; other versions may work but are untested.

Contact / Support

For questions about running the project, please contact the author or check the issues on this repository.


---

This is **ready to commit and push**. It also includes all the key points:  

- Python version ✅  
- Virtual environment instructions ✅  
- Dependency installation ✅  
- Notes about model files and venv ✅  

---
