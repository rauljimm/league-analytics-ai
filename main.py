import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data from CSV and perform basic cleaning
def load_data(file_path='data/match_features.csv'):
    """
    Loads match data from a CSV file and cleans it by removing invalid characters and null values.
    Returns the cleaned DataFrame.
    """
    df = pd.read_csv(file_path)
    print(f"Initial dataset shape: {df.shape}")
    print(f"Columns with null values: {df.columns[df.isnull().any()].tolist()}")
    
    # Clean object-type columns by removing unwanted characters
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace('**', '')
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass
    
    # Drop rows with null values
    df_cleaned = df.dropna()
    print(f"Shape after dropping nulls: {df_cleaned.shape}")
    
    # Ensure there are enough rows for training
    if len(df_cleaned) < 2:
        raise ValueError("Dataset has fewer than 2 matches after cleaning.")
    
    print(f"Numeric columns: {len(df_cleaned.select_dtypes(include=['int64', 'float64']).columns)}")
    print(f"Categorical columns: {len(df_cleaned.select_dtypes(include=['object']).columns)}")
    
    return df_cleaned

# Process champion columns and calculate cumulative performance metrics
def process_champions(df):
    """
    Extracts individual champions from 'blue_champions' and 'red_champions' columns.
    Adds cumulative performance metrics (gold, kills, towers, objectives) for each team.
    Returns the updated DataFrame.
    """
    # Extract individual champions from team lists
    def extract_champions(champions_str):
        return re.findall(r"'([^']*)'", champions_str)
    
    for team in ['blue', 'red']:
        champions_col = f'{team}_champions'
        if champions_col in df.columns:
            champions_list = df[champions_col].apply(extract_champions)
            assert all(len(champs) == 5 for champs in champions_list), f"Some rows have fewer than 5 champions for {team} team."
            for i in range(5):
                df[f'{team}_champion_{i+1}'] = champions_list.apply(lambda x: x[i] if len(x) > i else None)
    
    # Calculate cumulative metrics over time
    timestamps = [5, 10, 15, 20, 25, 30, 35, 40]
    for team in ['blue', 'red']:
        cumulative_gold = []
        cumulative_kills = []
        cumulative_towers = []
        cumulative_objectives = []
        
        for idx in df.index:
            gold = kills = towers = objectives = 0
            
            for t in timestamps:
                # Accumulate gold and kills (differences for blue, inverted for red)
                if team == 'blue':
                    gold += df.loc[idx, f'min_{t}_gold_diff'] if f'min_{t}_gold_diff' in df.columns else 0
                    kills += df.loc[idx, f'min_{t}_kill_diff'] if f'min_{t}_kill_diff' in df.columns else 0
                else:
                    gold += -df.loc[idx, f'min_{t}_gold_diff'] if f'min_{t}_gold_diff' in df.columns else 0
                    kills += -df.loc[idx, f'min_{t}_kill_diff'] if f'min_{t}_kill_diff' in df.columns else 0
                
                # Accumulate towers and objectives
                towers += df.loc[idx, f'min_{t}_{team}_towers'] if f'min_{t}_{team}_towers' in df.columns else 0
                objectives += df.loc[idx, f'min_{t}_{team}_dragons'] if f'min_{t}_{team}_dragons' in df.columns else 0
                objectives += (df.loc[idx, f'min_{t}_{team}_barons'] if f'min_{t}_{team}_barons' in df.columns else 0) * 2  # Baron has higher weight
                objectives += (df.loc[idx, f'min_{t}_{team}_elders'] if f'min_{t}_{team}_elders' in df.columns else 0) * 3  # Elder has highest weight
            
            cumulative_gold.append(gold)
            cumulative_kills.append(kills)
            cumulative_towers.append(towers)
            cumulative_objectives.append(objectives)
        
        df[f'{team}_cumulative_gold'] = cumulative_gold
        df[f'{team}_cumulative_kills'] = cumulative_kills
        df[f'{team}_cumulative_towers'] = cumulative_towers
        df[f'{team}_cumulative_objectives'] = cumulative_objectives
    
    # Calculate cumulative differences between teams
    df['cumulative_gold_diff'] = df['blue_cumulative_gold'] - df['red_cumulative_gold']
    df['cumulative_kill_diff'] = df['blue_cumulative_kills'] - df['red_cumulative_kills']
    df['cumulative_tower_diff'] = df['blue_cumulative_towers'] - df['red_cumulative_towers']
    df['cumulative_objective_diff'] = df['blue_cumulative_objectives'] - df['red_cumulative_objectives']
    
    return df

# Perform exploratory data analysis and save visualizations
def explore_data(df):
    """
    Analyzes the dataset and generates visualizations for win distribution, game duration, and feature correlations.
    Returns the correlation of numeric features with the target variable.
    """
    print("Descriptive statistics:")
    print(df.describe())
    
    print("\nTarget variable distribution:")
    balance = df['blue_win'].value_counts(normalize=True) * 100
    print(f"Blue team wins: {balance.get(True, 0):.2f}%")
    print(f"Blue team losses: {balance.get(False, 0):.2f}%")
    
    # Plot win distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='blue_win', data=df)
    plt.title('Win Distribution')
    plt.xlabel('Blue Team Win')
    plt.ylabel('Count')
    plt.savefig('win_distribution.png')
    plt.close()
    
    # Plot top champions for each team
    blue_champions = pd.Series([champ for i in range(1, 6) for champ in df[f'blue_champion_{i}']])
    print("\nTop 10 most popular champions (Blue Team):")
    print(blue_champions.value_counts().head(10))
    
    red_champions = pd.Series([champ for i in range(1, 6) for champ in df[f'red_champion_{i}']])
    print("\nTop 10 most popular champions (Red Team):")
    print(red_champions.value_counts().head(10))
    
    # Plot game duration distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['game_duration'] / 60, bins=30)
    plt.title('Game Duration Distribution')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.savefig('game_duration.png')
    plt.close()
    
    # Calculate and plot correlations with target
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation_with_win = df[numeric_cols].corrwith(df['blue_win']).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 10))
    correlation_top = pd.concat([correlation_with_win.head(15), correlation_with_win.tail(15)])
    sns.barplot(x=correlation_top.values, y=correlation_top.index)
    plt.title('Top Features Correlated with Win')
    plt.xlabel('Correlation')
    plt.tight_layout()
    plt.savefig('correlation_with_win.png')
    plt.close()
    
    return correlation_with_win

# Prepare data for training with feature preprocessing
def prepare_data(df):
    """
    Prepares the dataset for training by selecting features, applying preprocessing, and splitting into train/validation/test sets.
    Returns preprocessed data and the preprocessor.
    """
    y = df['blue_win'].astype(int)
    
    champion_cols = [col for col in df.columns if 'champion_' in col]
    numeric_cols = [
        'min_5_gold_diff', 'min_5_kill_diff', 'min_5_blue_towers', 'min_5_red_towers',
        'min_5_blue_dragons', 'min_5_red_dragons', 'min_5_blue_wards', 'min_5_red_wards',
        'min_10_gold_diff', 'min_10_kill_diff', 'min_10_blue_towers', 'min_10_red_towers',
        'min_10_blue_dragons', 'min_10_red_dragons', 'min_10_blue_wards', 'min_10_red_wards',
        'min_15_gold_diff', 'min_15_kill_diff', 'min_15_blue_towers', 'min_15_red_towers',
        'min_15_blue_dragons', 'min_15_red_dragons', 'min_15_blue_wards', 'min_15_red_wards',
        'min_20_gold_diff', 'min_20_kill_diff', 'min_20_blue_towers', 'min_20_red_towers',
        'min_20_blue_dragons', 'min_20_red_dragons', 'min_20_blue_wards', 'min_20_red_wards',
        'min_25_gold_diff', 'min_25_kill_diff', 'min_25_blue_towers', 'min_25_red_towers',
        'min_25_blue_dragons', 'min_25_red_dragons', 'min_25_blue_wards', 'min_25_red_wards',
        'min_30_gold_diff', 'min_30_kill_diff', 'min_30_blue_towers', 'min_30_red_towers',
        'min_30_blue_dragons', 'min_30_red_dragons', 'min_30_blue_wards', 'min_30_red_wards',
        'min_30_blue_barons', 'min_30_red_barons', 'min_30_blue_elders', 'min_30_red_elders',
        'min_35_gold_diff', 'min_35_kill_diff', 'min_35_blue_towers', 'min_35_red_towers',
        'min_35_blue_dragons', 'min_35_red_dragons', 'min_35_blue_wards', 'min_35_red_wards',
        'min_35_blue_barons', 'min_35_red_barons', 'min_35_blue_elders', 'min_35_red_elders',
        'min_40_gold_diff', 'min_40_kill_diff', 'min_40_blue_towers', 'min_40_red_towers',
        'min_40_blue_dragons', 'min_40_red_dragons', 'min_40_blue_wards', 'min_40_red_wards',
        'min_40_blue_barons', 'min_40_red_barons', 'min_40_blue_elders', 'min_40_red_elders',
        'cumulative_gold_diff', 'cumulative_kill_diff', 'cumulative_tower_diff', 'cumulative_objective_diff'
    ]
    
    # Filter numeric columns that exist in the DataFrame
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # Reduce the weight of gold and towers in late game (after 30 minutes)
    for t in [30, 35, 40]:
        gold_col = f'min_{t}_gold_diff'
        if gold_col in df.columns:
            df[f'weighted_{gold_col}'] = df[gold_col] * 0.5
            numeric_cols.append(f'weighted_{gold_col}')
            numeric_cols.remove(gold_col)
        
        tower_cols = [f'min_{t}_blue_towers', f'min_{t}_red_towers']
        for col in tower_cols:
            if col in df.columns:
                df[f'weighted_{col}'] = df[col] * 0.5
                numeric_cols.append(f'weighted_{col}')
                numeric_cols.remove(col)
    
    print(f"Champion columns: {len(champion_cols)}")
    print(f"Numeric columns: {len(numeric_cols)}")
    
    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), champion_cols)
        ],
        remainder='drop'
    )
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        df[numeric_cols + champion_cols], y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Apply preprocessing
    X_train_prep = preprocessor.fit_transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)
    
    print(f"X_train shape after preprocessing: {X_train_prep.shape}")
    print(f"X_val shape after preprocessing: {X_val_prep.shape}")
    print(f"X_test shape after preprocessing: {X_test_prep.shape}")
    
    return X_train_prep, X_val_prep, X_test_prep, y_train, y_val, y_test, preprocessor

# Define the neural network model architecture
def create_model(input_dim):
    """
    Creates a neural network model with dropout and batch normalization for regularization.
    Returns the compiled model.
    """
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

# Train the neural network model
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Trains the neural network model with early stopping and learning rate reduction.
    Returns the training history and trained model.
    """
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val),
        callbacks=callbacks, verbose=1
    )
    
    return history, model

# Evaluate the model on test data
def evaluate_model(model, X_test, y_test, calibrator=None):
    """
    Evaluates the model using accuracy, AUC-ROC, confusion matrix, and other metrics.
    Saves a confusion matrix plot and returns predicted probabilities.
    """
    y_pred_prob = model.predict(X_test).flatten()
    
    if calibrator is not None:
        y_pred_prob = calibrator.predict_proba(y_pred_prob.reshape(-1, 1))[:, 1]
    
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print(f"Model accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_prob):.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Blue Loss', 'Blue Win'], yticklabels=['Blue Loss', 'Blue Win'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return y_pred_prob

# Visualize training history
def plot_training_history(history):
    """
    Plots training and validation loss and accuracy over epochs.
    Saves the plots to a file.
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Analyze probability calibration
def analyze_calibration(y_test, y_pred_prob, n_bins=10):
    """
    Analyzes the calibration of predicted probabilities by comparing predicted vs actual probabilities.
    Saves a calibration curve plot.
    """
    y_pred_prob = y_pred_prob.flatten()
    y_test = y_test.values if hasattr(y_test, 'values') else y_test
    
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_prob, bins, right=True)
    bin_indices = np.clip(bin_indices, 1, n_bins) - 1
    
    bin_probs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (bin_indices == i)
        count = np.sum(mask)
        bin_counts[i] = count
        bin_probs[i] = np.mean(y_test[mask]) if count > 0 else np.nan
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    valid_bins = ~np.isnan(bin_probs)
    plt.plot(bin_centers[valid_bins], bin_probs[valid_bins], 'o-', label='Model')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Positive Fraction')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('calibration_curve.png')
    plt.close()
    
    print("Calibration stats per bin:")
    for i in range(n_bins):
        if not np.isnan(bin_probs[i]):
            print(f"Bin {i+1} [{bins[i]:.2f}-{bins[i+1]:.2f}]: Mean Prob = {bin_probs[i]:.4f}, Samples = {int(bin_counts[i])}")

# Predict the outcome of a single match
def predict_match(model, preprocessor, match_data, calibrator=None):
    """
    Predicts the win probability for a single match.
    Returns a dictionary with probabilities for both teams.
    """
    if not isinstance(match_data, pd.DataFrame):
        raise ValueError("match_data must be a pandas DataFrame")
    
    X_pred = preprocessor.transform(match_data)
    prob_blue = model.predict(X_pred)[0][0]
    
    if calibrator is not None:
        prob_blue = calibrator.predict_proba(np.array([[prob_blue]]))[:, 1][0]
    
    prob_red = 1 - prob_blue
    
    return {
        'Blue Win Probability': f"{prob_blue*100:.2f}%",
        'Red Win Probability': f"{prob_red*100:.2f}%",
        'Numeric Values': {'prob_blue': float(prob_blue), 'prob_red': float(prob_red)}
    }

# Save the model and preprocessing artifacts
def save_model(model, preprocessor, champion_cols, numeric_cols, calibrator=None):
    """
    Saves the trained model, preprocessor, and calibrator to disk.
    """
    model.save('lol_model.h5')
    joblib.dump(preprocessor, 'lol_preprocessor.pkl')
    if calibrator is not None:
        joblib.dump(calibrator, 'lol_calibrator.pkl')
    artifacts = {'champion_cols': champion_cols, 'numeric_cols': numeric_cols}
    joblib.dump(artifacts, 'lol_columns.pkl')
    print("Model and artifacts saved successfully.")

# Analyze prediction errors
def analyze_errors(y_test, y_pred_prob, X_test, df_test, n=5):
    """
    Analyzes the top N matches with the largest prediction errors.
    Prints detailed stats for each match.
    """
    y_pred_prob = y_pred_prob.flatten()
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    error = np.abs(y_pred_prob - y_test_np)
    top_error_indices = np.argsort(error)[-n:][::-1]
    
    print(f"Top {n} matches with largest prediction errors:")
    for i, idx in enumerate(top_error_indices):
        test_idx = y_test.index[idx]
        actual = "Blue Win" if y_test.iloc[idx] == 1 else "Red Win"
        prob_blue = y_pred_prob[idx]
        prob_red = 1 - prob_blue
        predicted = "Blue Win" if prob_blue > 0.5 else "Red Win"
        
        print(f"\nMatch {i+1}:")
        print(f"DataFrame index: {test_idx}")
        print(f"Actual result: {actual}")
        print(f"Prediction: {predicted} (Blue: {prob_blue*100:.2f}%, Red: {prob_red*100:.2f}%)")
        print(f"Error: {error[idx]:.4f}")
        
        match = df_test.loc[test_idx]
        print(f"Duration: {match['game_duration']/60:.2f} minutes")
        print(f"Blue Team: {', '.join([match[f'blue_champion_{i+1}'] for i in range(5)])}")
        print(f"Red Team: {', '.join([match[f'red_champion_{i+1}'] for i in range(5)])}")
        
        stats_cols = [
            'min_15_gold_diff', 'min_15_kill_diff', 'min_15_blue_towers', 'min_15_red_towers',
            'min_20_gold_diff', 'min_20_kill_diff', 'min_20_blue_towers', 'min_20_red_towers',
            'min_25_gold_diff', 'min_25_kill_diff', 'min_25_blue_towers', 'min_25_red_towers',
            'weighted_min_30_gold_diff', 'min_30_kill_diff', 'weighted_min_30_blue_towers', 'weighted_min_30_red_towers',
            'min_30_blue_dragons', 'min_30_red_dragons', 'min_30_blue_barons', 'min_30_red_barons',
            'min_30_blue_elders', 'min_30_red_elders',
            'cumulative_gold_diff', 'cumulative_kill_diff', 'cumulative_tower_diff', 'cumulative_objective_diff'
        ]
        stats_cols = [col for col in stats_cols if col in match.index]
        stats = match[stats_cols].to_dict()
        print("Key statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("-" * 50)

# Calculate feature importance for numeric features
def feature_importance(model, numeric_cols, preprocessor, top_n=20):
    """
    Calculates the approximate importance of numeric features based on the first layer weights.
    Saves a bar plot of the top features.
    Returns a DataFrame with feature importance.
    """
    weights = model.layers[0].get_weights()[0]
    n_numeric = len(numeric_cols)
    numeric_weights = weights[:n_numeric, :]
    importance = np.mean(np.abs(numeric_weights), axis=1)
    
    importance_df = pd.DataFrame({'Feature': numeric_cols, 'Importance': importance})
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(14, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
    plt.title(f'Top {top_n} Most Important Numeric Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance_df

# Main function to run the entire pipeline
def main():
    """
    Main function to execute the full pipeline: data loading, preprocessing, training, evaluation, and analysis.
    """
    try:
        print("=== League of Legends Match Prediction Pipeline ===")
        
        print("\n[1] Loading data...")
        df = load_data()
        
        print("\n[2] Processing champions and cumulative metrics...")
        df = process_champions(df)
        
        print("\n[3] Exploring data...")
        explore_data(df)
        
        print("\n[4] Preparing data for training...")
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = prepare_data(df)
        
        print("\n[5] Creating neural network model...")
        model = create_model(X_train.shape[1])
        
        print("\n[6] Training model...")
        history, trained_model = train_model(model, X_train, y_train, X_val, y_val)
        
        print("\n[7] Evaluating model (uncalibrated)...")
        y_pred_prob_uncalibrated = evaluate_model(trained_model, X_test, y_test)
        
        print("\n[8] Calibrating probabilities...")
        y_val_prob = trained_model.predict(X_val).flatten()
        print("Validation set predicted probability distribution:")
        print(pd.Series(y_val_prob).describe())
        calibrator = LogisticRegression()
        calibrator.fit(y_val_prob.reshape(-1, 1), y_val)
        
        print("\n[9] Evaluating model (calibrated)...")
        y_pred_prob = evaluate_model(trained_model, X_test, y_test, calibrator)
        
        print("\n[10] Plotting training history...")
        plot_training_history(history)
        
        print("\n[11] Analyzing probability calibration...")
        analyze_calibration(y_test, y_pred_prob)
        
        print("\n[12] Saving model and artifacts...")
        champion_cols = [col for col in df.columns if 'champion_' in col]
        numeric_cols = [
            'min_5_gold_diff', 'min_5_kill_diff', 'min_5_blue_towers', 'min_5_red_towers',
            'min_5_blue_dragons', 'min_5_red_dragons', 'min_5_blue_wards', 'min_5_red_wards',
            'min_10_gold_diff', 'min_10_kill_diff', 'min_10_blue_towers', 'min_10_red_towers',
            'min_10_blue_dragons', 'min_10_red_dragons', 'min_10_blue_wards', 'min_10_red_wards',
            'min_15_gold_diff', 'min_15_kill_diff', 'min_15_blue_towers', 'min_15_red_towers',
            'min_15_blue_dragons', 'min_15_red_dragons', 'min_15_blue_wards', 'min_15_red_wards',
            'min_20_gold_diff', 'min_20_kill_diff', 'min_20_blue_towers', 'min_20_red_towers',
            'min_20_blue_dragons', 'min_20_red_dragons', 'min_20_blue_wards', 'min_20_red_wards',
            'min_25_gold_diff', 'min_25_kill_diff', 'min_25_blue_towers', 'min_25_red_towers',
            'min_25_blue_dragons', 'min_25_red_dragons', 'min_25_blue_wards', 'min_25_red_wards',
            'weighted_min_30_gold_diff', 'min_30_kill_diff', 'weighted_min_30_blue_towers', 'weighted_min_30_red_towers',
            'min_30_blue_dragons', 'min_30_red_dragons', 'min_30_blue_wards', 'min_30_red_wards',
            'min_30_blue_barons', 'min_30_red_barons', 'min_30_blue_elders', 'min_30_red_elders',
            'weighted_min_35_gold_diff', 'min_35_kill_diff', 'weighted_min_35_blue_towers', 'weighted_min_35_red_towers',
            'min_35_blue_dragons', 'min_35_red_dragons', 'min_35_blue_wards', 'min_35_red_wards',
            'min_35_blue_barons', 'min_35_red_barons', 'min_35_blue_elders', 'min_35_red_elders',
            'weighted_min_40_gold_diff', 'min_40_kill_diff', 'weighted_min_40_blue_towers', 'weighted_min_40_red_towers',
            'min_40_blue_dragons', 'min_40_red_dragons', 'min_40_blue_wards', 'min_40_red_wards',
            'min_40_blue_barons', 'min_40_red_barons', 'min_40_blue_elders', 'min_40_red_elders',
            'cumulative_gold_diff', 'cumulative_kill_diff', 'cumulative_tower_diff', 'cumulative_objective_diff'
        ]
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        save_model(trained_model, preprocessor, champion_cols, numeric_cols, calibrator)
        
        print("\n[13] Analyzing feature importance...")
        importance = feature_importance(trained_model, numeric_cols, preprocessor)
        print("Top 15 most important features:")
        print(importance.head(15))
        
        print("\n[14] Predicting a sample match...")
        test_indices = y_test.index
        sample_idx = test_indices[0]
        sample_match = df.loc[sample_idx, numeric_cols + champion_cols].to_frame().T
        prediction = predict_match(trained_model, preprocessor, sample_match, calibrator)
        print("Sample match prediction:")
        print(prediction)
        print(f"Actual result: {'Blue Win' if df.loc[sample_idx, 'blue_win'] else 'Red Win'}")
        
        print("\n[15] Analyzing prediction errors...")
        analyze_errors(y_test, y_pred_prob, X_test, df.loc[y_test.index])
        
        print("\n=== Pipeline Completed ===")
    
    except Exception as e:
        import traceback
        print(f"\nError occurred during execution:")
        print(f"Error: {str(e)}")
        print("\nFull error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()