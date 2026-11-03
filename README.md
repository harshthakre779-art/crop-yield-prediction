# Crop Yield Prediction – Streamlit

## Requirements
- Python 3.10 or newer
- VS Code
- Internet connection for installing Python packages

## Setup in VS Code

Open this folder in VS Code. In Terminal > New Terminal, run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```cmd
.venv\Scripts\activate.bat
```

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Train the model and generate the sample CSV:
```bash
python train_model.py
```

Run the Streamlit app:
```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal (normally http://localhost:8501).

## Important data limitation
The included dataset is generated synthetically for software demonstration and learning. It is not real agricultural data and the output is not a reliable farming recommendation. For research or deployment, use verified historical crop-yield observations with matching crop, location, season, weather and soil measurements. Validate on data from different seasons/locations before drawing conclusions.
