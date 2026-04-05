import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime


def generate_gbm(start_price=100, mu=0.08, sigma=0.2, days=500, seed=42):
    np.random.seed(seed)
    dt = 1/252
    prices = [start_price]
    for _ in range(days-1):
        drift = (mu - 0.5 * sigma**2) * dt
        shock = sigma * np.sqrt(dt) * np.random.normal()
        price = prices[-1] * np.exp(drift + shock)
        prices.append(price)
    dates = pd.date_range(end=datetime.date.today(), periods=days)
    return pd.DataFrame({'date': dates, 'close': prices})


def compute_features(df, window=20):
    df = df.copy()
    df['log_return'] = np.log(df['close']).diff()
    df['volatility'] = df['log_return'].rolling(window).std() * np.sqrt(252)
    df = df.dropna().reset_index(drop=True)
    features = df[['log_return', 'volatility']].values
    return df, features


def fit_hmm(features, n_states=3, random_state=42):
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=random_state)
    model.fit(features)
    hidden_states = model.predict(features)
    regime_probs = model.predict_proba(features)
    transmat = model.transmat_
    means = model.means_
    covars = np.sqrt(np.array([np.diag(c) for c in model.covars_]))
    return hidden_states, regime_probs, transmat, means, covars

# KMeans Model

def fit_kmeans(features, n_states=3, random_state=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    return labels, centers


# Regime Statistics

def compute_regime_stats(df, labels, n_states):
    stats = []
    for i in range(n_states):
        mask = labels == i
        mean_ret = df.loc[mask, 'log_return'].mean()
        vol = df.loc[mask, 'log_return'].std() * np.sqrt(252)
        stats.append({'Regime': i, 'Mean Return': mean_ret, 'Volatility': vol, 'Count': mask.sum()})
    return pd.DataFrame(stats)

# =============================
# Visualization
# =============================
def plot_regimes(df, labels, title="Price Chart by Regime"):
    plt.figure(figsize=(14,6))
    palette = sns.color_palette("Set1", np.unique(labels).max()+1)
    for regime in np.unique(labels):
        mask = labels == regime
        plt.plot(df['date'][mask], df['close'][mask], '.', label=f'Regime {regime}', color=palette[regime], alpha=0.7)
    plt.plot(df['date'], df['close'], color='gray', alpha=0.3, linewidth=1)
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.tight_layout()
    plt.show()

def plot_regime_probs(df, regime_probs):
    plt.figure(figsize=(14,4))
    for i in range(regime_probs.shape[1]):
        plt.plot(df['date'], regime_probs[:,i], label=f'Regime {i}')
    plt.title('Regime Probabilities')
    plt.xlabel('Date')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(transmat):
    plt.figure(figsize=(6,5))
    sns.heatmap(transmat, annot=True, cmap='Blues', fmt='.2f', square=True)
    plt.title('Regime Transition Matrix')
    plt.xlabel('To Regime')
    plt.ylabel('From Regime')
    plt.tight_layout()
    plt.show()


# Main Function 

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Market Regime Detection Engine")
    parser.add_argument('--csv', type=str, help='Path to CSV file (date, close)')
    parser.add_argument('--gbm', action='store_true', help='Use synthetic GBM data')
    parser.add_argument('--n_states', type=int, default=3, help='Number of regimes')
    parser.add_argument('--window', type=int, default=20, help='Rolling window size')
    parser.add_argument('--model', type=str, default='HMM', choices=['HMM', 'KMeans'], help='Model type')
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=['date'])
    elif args.gbm:
        df = generate_gbm()
    else:
        print("Please provide --csv or --gbm")
        return

    df, features = compute_features(df, args.window)

    if args.model == 'HMM':
        labels, regime_probs, transmat, means, covars = fit_hmm(features, args.n_states)
        print("Transition Matrix:\n", transmat)
        print("State Means:\n", means)
        print("State Volatilities:\n", covars)
    else:
        labels, centers = fit_kmeans(features, args.n_states)
        regime_probs = np.zeros((len(labels), args.n_states))
        regime_probs[np.arange(len(labels)), labels] = 1
        transmat = None
        print("Cluster Centers:\n", centers)

    stats = compute_regime_stats(df, labels, args.n_states)
    print("Regime Statistics:\n", stats)

    # Plots
    plot_regimes(df, labels)
    plot_regime_probs(df, regime_probs)
    if transmat is not None:
        plot_transition_matrix(transmat)

if __name__ == "__main__":
    main()
