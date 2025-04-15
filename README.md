# 🎯 LoL Match Prediction Pipeline

A comprehensive pipeline for scraping League of Legends (LoL) match data, extracting features, and predicting match outcomes using a neural network model. This project is designed to collect data from professional players, process match details, and train a predictive model to determine the likelihood of a team winning based on in-game metrics.

---

## 🚀 Project Overview

This repository contains scripts to:

- 🪄 Scrape professional players' PUUIDs from LoL's Challenger leagues (`scraping_tools/proplayers_scrap.py`)
- 🔹 Collect match IDs from ranked games of these players (`scraping_tools/proplayers_history.py`)
- ⚖️ Extract detailed match features like gold, kills, towers, and dragons at various timestamps (`scraping_tools/game_info.py`)
- 🏋️ Train a neural network model to predict match outcomes based on extracted features (`main.py`)

The project is structured to handle API rate limits, avoid duplicate data, and provide detailed analysis of model performance through metrics and visualizations.

---

## 📂 Project Structure

```
lol-match-prediction/
│
├── data/
│   ├── challenger.txt         # Stores PUUIDs of Challenger players
│   ├── matches.txt            # Stores match IDs for processing
│   └── match_features.csv     # Stores extracted match features
│
├── models/
│   └── league_scraper.py      # API request handler for Riot Games
│
├── scraping_tools/
│   ├── proplayers_scrap.py        # Scrape Challenger players' PUUIDs
│   ├── proplayers_history.py      # Fetch match IDs from players
│   └── game_info.py               # Extract match features
│
├── config.py                  # Configuration file for API key
├── main.py                    # Neural network model for prediction
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- A Riot Games API key *(get one from [developer.riotgames.com](https://developer.riotgames.com))*

### Setup Steps

1. **Clone the Repository**
```bash
git clone https://github.com/rauljimm/league-analytics-ai.git
cd league-analytics-ai
```

2. **Install Dependencies**
Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**requirements.txt** example:
```
pandas
numpy
requests
scikit-learn
tensorflow
matplotlib
seaborn
```

3. **Configure API Key**
Open `config.py` and add your Riot Games API key:
```python
API_KEY = "YOUR-API-KEY"  # Replace with your actual API key
```

4. **Prepare Data Directory**
Make sure the `data/` directory exists:
```bash
mkdir -p data
```

---

## 📜 Usage

### Step 1: Scrape Challenger Players
Collect PUUIDs of Challenger players from EUW, NA, and KR:
```bash
python scraping_tools/proplayers_scrap.py
```
Output: PUUIDs saved to `data/challenger.txt`

### Step 2: Collect Match IDs
Fetch match IDs from ranked games of these players:
```bash
python scraping_tools/proplayers_history.py
```
Output: Match IDs saved to `data/matches.txt`

### Step 3: Extract Match Features
Process match IDs and extract features like gold differences, kills, towers, dragons:
```bash
python scraping_tools/game_info.py
```
Output: Features saved to `data/match_features.csv`

### Step 4: Train and Evaluate the Model
Train the model and predict match outcomes:
```bash
python main.py
```
Output:
- Trained model and artifacts (`lol_model.h5`, `lol_preprocessor.pkl`, etc.)
- Visualizations: confusion matrix, calibration curve, training history
- Performance metrics printed in the console

---

## 📊 Model Details

The prediction model (`main.py`) uses a neural network to predict the probability of the **Blue** team winning. Features include:

- Gold and kill differences at 5, 10, 15, 20, and 25 minutes
- Towers, dragons, and ward counts
- Cumulative team stats
- Champion picks (one-hot encoded)

### Model Architecture

- **Input Layer:** Scaled features + one-hot champions
- **Hidden Layers:**
  - Dense (64), ReLU, BatchNorm, Dropout
  - Dense (32), ReLU, BatchNorm, Dropout
  - Dense (16), ReLU, BatchNorm, Dropout
- **Output Layer:** Sigmoid (Blue win probability)

### Outputs
- Accuracy, AUC-ROC, F1-Score, Sensitivity, Specificity
- Visuals: Confusion matrix, calibration curve, training plots

---

## ⚠️ Notes

- **API Rate Limits:** Scripts use `time.sleep()` to comply with Riot's rate limits
- **Remakes:** Matches under 15 minutes are skipped
- **In Progress:** Expect improvements (e.g. more features, better calibration, regional support)

---

## 🤝 Contributing

Contributions welcome!
1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file.

---

## 📧 Contact

For questions or suggestions:
**Email:** rauljimm.dev@gmail.com

Happy predicting! 🎉
