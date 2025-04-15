LoL Match Prediction Pipeline 🎮

A comprehensive pipeline for scraping League of Legends (LoL) match data, extracting features, and predicting match outcomes using a neural network model. This project is designed to collect data from professional players, process match details, and train a predictive model to determine the likelihood of a team winning based on in-game metrics.

🚀 Project Overview
This repository contains scripts to:

Scrape professional players' PUUIDs from LoL's Challenger leagues (proplayers_scrap.py).
Collect match IDs from ranked games of these players (proplayers_history.py).
Extract detailed match features like gold, kills, towers, and dragons at various timestamps (league_data_scraper.py).
Train a neural network model to predict match outcomes based on extracted features (main_lol_prediction_optimized.py).

The project is structured to handle API rate limits, avoid duplicate data, and provide detailed analysis of model performance through metrics and visualizations.

📂 Project Structure
lol-match-prediction/
│
├── data/
│   ├── challenger.txt       # Stores PUUIDs of Challenger players
│   ├── matches.txt          # Stores match IDs for processing
│   └── match_features.csv   # Stores extracted match features
│
├── models/
│   └── league_scraper.py    # API request handler for Riot Games
│
├── scraping_tools/
│   ├── proplayers_scrap.py      # Script to scrape Challenger players' PUUIDs
│   ├── proplayers_history.py    # Script to collect match IDs from players
│   └── game_info.py             # Script to extract match features
├── config.py                # Configuration file for API key
├── main.py  # Neural network model for prediction
└── README.md                # Project documentation


🛠️ Installation
Prerequisites

Python 3.8+
A Riot Games API key (get one from developer.riotgames.com)
Required Python packages (install via requirements.txt)

Setup Steps

Clone the Repository
git clone https://github.com/yourusername/lol-match-prediction.git
cd lol-match-prediction


Install DependenciesCreate a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Example requirements.txt:
pandas
numpy
requests
scikit-learn
tensorflow
matplotlib
seaborn


Configure API KeyOpen config.py and add your Riot Games API key:
API_KEY = "YOUR-API-KEY"  # Replace with your actual API key


Prepare Data DirectoryEnsure the data/ directory exists:
mkdir data




📜 Usage
Step 1: Scrape Challenger Players
Run proplayers_scrap.py to collect PUUIDs of Challenger players from EUW, NA, and KR regions:
python proplayers_scrap.py


Output: PUUIDs are saved to data/challenger.txt.

Step 2: Collect Match IDs
Run proplayers_history.py to fetch match IDs from the ranked games of these players:
python proplayers_history.py


Output: Match IDs are saved to data/matches.txt.

Step 3: Extract Match Features
Run league_data_scraper.py to process match IDs and extract features like gold differences, kills, towers, and dragons at specific timestamps:
python league_data_scraper.py


Output: Features are saved to data/match_features.csv.

Step 4: Train and Evaluate the Model
Run main_lol_prediction_optimized.py to train a neural network model and predict match outcomes:
python main_lol_prediction_optimized.py


Output:
Trained model and artifacts (lol_model.h5, lol_preprocessor.pkl, etc.).
Visualizations (e.g., confusion_matrix.png, calibration_curve.png).
Performance metrics and error analysis printed to the console.




📊 Model Details
The prediction model (main_lol_prediction_optimized.py) uses a neural network to predict the probability of the Blue team winning a match. Key features include:

Gold and kill differences at 5, 10, 15, 20, and 25 minutes.
Tower, dragon, and ward counts for both teams.
Cumulative performance metrics (gold, kills, towers, objectives).
Champion selections for both teams (one-hot encoded).

Model Architecture

Input Layer: Scaled numeric features + one-hot encoded champions.
Hidden Layers: 3 dense layers (64, 32, 16 neurons) with ReLU activation, BatchNormalization, and Dropout.
Output Layer: Sigmoid activation for binary classification (Blue Win probability).

Outputs

Metrics: Accuracy, AUC-ROC, Sensitivity, Specificity, F1-Score.
Visualizations:
Confusion Matrix
Calibration Curve
Training History (loss and accuracy)
Feature Importance




⚠️ Notes

API Rate Limits: The scripts include delays (time.sleep) to respect Riot Games API rate limits. Adjust these if needed based on your API key's limits.
Data Quality: Matches shorter than 15 minutes are skipped to avoid remakes.
Work in Progress: The project is actively being developed. Future improvements may include better calibration, additional features (e.g., late-game objectives), and support for more regions.


🤝 Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Commit your changes (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.


📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

📧 Contact
For questions or suggestions, feel free to open an issue or reach out via email: rauljimm.dev@gmail.com.
Happy predicting! 🎉
