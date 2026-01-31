"""
Elo-Based Tennis Match Outcome Model
- Implements a probabilistic framework for predicting head-to-head tennis match outcomes
  using Elo ratings.
- Represents player strength as a latent scalar rating updated sequentially from historical
  win–loss results.
- Supports both global Elo ratings and surface-specific Elo ratings to capture
  surface-dependent performance differences.
- Uses Monte Carlo simulation to illustrate outcome variability and sampling uncertainty
  given a fixed win probability.
- Evaluates predictive performance using proper scoring rules, including log loss and the
  Brier score, along with calibration diagnostics on a held-out test set.
- Intended for demonstration of probabilistic modeling techniques
  in sports analytics.
"""
# Importing necessary libraries
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
# Setting random seed for reproducibility
np.random.seed(42)
# Defining the EloModel class
@dataclass
class EloModel:
    # Setting hyperparameters and data holders
    initial_elo: float = 1500.0 # starting Elo rating for new players
    k_factor: float = 25.0 # how quickly Elo ratings adapt to new information
    split_date: str = "2022-01-01"
    global_elos: Dict[str, float] = field(default_factory=dict)
    surface_elos: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    trained: bool = False
    train_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None
    @staticmethod
    # Elo probability function
    def elo_win_prob(elo_a: float, elo_b: float) -> float:
        """
        Computing the probability that player A wins against player B
        using the Elo logistic function.
        Parameters:
        elo_a (float): Elo rating of player A
        elo_b (float): Elo rating of player B

        Returns:
        float: Probability that player A wins
        """
        return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

    def update_elo(self, elo_a: float, elo_b: float, result_a: int) -> Tuple[float, float]:
        """
        Updating Elo ratings for a single match.
        Parameters:
            elo_a (float): current Elo rating of player A
            elo_b (float): current Elo rating of player B
            result_a (int): 1 if A wins, 0 otherwise
            k (float): K-factor
        
        Returns:
            float, float : updated Elo ratings for player A and B
        """

        p_a = self.elo_win_prob(elo_a, elo_b)
        elo_a_new = elo_a + self.k_factor * (result_a - p_a)
        elo_b_new = elo_b + self.k_factor * ((1 - result_a) - (1 - p_a))
        return elo_a_new, elo_b_new

    @staticmethod
    def _get_surface_from_row(row: pd.Series) -> str:
        """
        Return surface name string for a row.
        Parameters:
            row (pd.Series): DataFrame row with surface information
        Returns:
            str: Surface name ('Clay', 'Grass', 'Hard', or 'Unknown')
        """
        if "surface" in row.index and pd.notna(row["surface"]):
            return str(row["surface"]).capitalize()
        for s in ("surface_Clay", "surface_Grass", "surface_Hard"):
            if s in row.index and row[s] == 1:
                return s.split("_", 1)[1]
        return "Unknown"

    # Training the model
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the Elo model on historical match data.
        Parameters:
            df (pd.DataFrame): DataFrame with columns ['player_A', 'player_B', 'A_wins', 'tourney_date', 'surface']
        Returns:
            None
        """
        if "tourney_date" in df.columns:
            df = df.copy()
            df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
            df = df.sort_values("tourney_date")
        else:
            df = df.copy()
        # Train-test split
        split_dt = pd.to_datetime(self.split_date)
        train = df[df["tourney_date"] < split_dt].copy() if "tourney_date" in df.columns else df.iloc[:int(0.8 * len(df))].copy()
        test  = df[df["tourney_date"] >= split_dt].copy() if "tourney_date" in df.columns else df.iloc[int(0.8 * len(df)):].copy()
        # Initialize Elo dictionaries
        self.global_elos = {}
        self.surface_elos = defaultdict(dict)
        # Sequentially update global
        for _, row in train.iterrows():
            a, b = row["player_A"], row["player_B"]
            y = int(row["A_wins"])
            ea = self.global_elos.get(a, self.initial_elo)
            eb = self.global_elos.get(b, self.initial_elo)
            ea_new, eb_new = self.update_elo(ea, eb, y)
            self.global_elos[a] = ea_new
            self.global_elos[b] = eb_new
        # Surface-specific ratings
        for _, row in train.iterrows():
            surf = self._get_surface_from_row(row)
            a, b = row["player_A"], row["player_B"]
            y = int(row["A_wins"])
            sa = self.surface_elos[surf].get(a, self.initial_elo)
            sb = self.surface_elos[surf].get(b, self.initial_elo)
            sa_new, sb_new = self.update_elo(sa, sb, y)
            self.surface_elos[surf][a] = sa_new
            self.surface_elos[surf][b] = sb_new
        self.trained = True
        self.train_df = train
        self.test_df = test

    # Prediction and simulation
    def predict(self, player_a: str, player_b: str, surface: Optional[str] = None, alpha: float = 0.5, default_elo: Optional[float] = None) -> Dict:
        """
        Predict the probability that player A wins against player B.
        Parameters:
            player_a (str): Name of player A
            player_b (str): Name of player B
            surface (Optional[str]): Surface type ('Clay', 'Grass', 'Hard') or None
            alpha (float): Blending weight for surface-specific probability
            default_elo (Optional[float]): Default Elo rating for unknown players
        Returns:
            Dict: Dictionary with Elo ratings and win probabilities
        """
        if default_elo is None:
            default_elo = self.initial_elo
        # Global Elo probabilities
        elo_a = self.global_elos.get(player_a, default_elo)
        elo_b = self.global_elos.get(player_b, default_elo)
        p_global = self.elo_win_prob(elo_a, elo_b)
        # Surface-specific Elo probabilities
        p_surface = None
        if surface is not None and surface in self.surface_elos:
            sdict = self.surface_elos[surface]
            sa = sdict.get(player_a, default_elo)
            sb = sdict.get(player_b, default_elo)
            p_surface = self.elo_win_prob(sa, sb)

        if p_surface is None:
            p_used = p_global
        else:
            p_used = alpha * p_surface + (1 - alpha) * p_global

        return {
            "player_A": player_a,
            "player_B": player_b,
            "elo_A_global": elo_a,
            "elo_B_global": elo_b,
            "p_global": p_global,
            "p_surface": p_surface,
            "p_used": p_used
        }
    def simulate(self, p: float, n_sim: int = 10000) -> Dict:
        """
        Simulate match outcomes given win probability p using Monte Carlo.
        Parameters:
            p (float): Probability that player A wins
            n_sim (int): Number of simulations to run
        Returns:
            Dict: Dictionary with mean win probability and 95% confidence interval
        """
        sims = np.random.binomial(1, p, size=n_sim)
        mean = sims.mean()
        se = np.sqrt(p * (1 - p) / n_sim)
        z = 1.96
        ci_lower = max(0.0, p - z * se)
        ci_upper = min(1.0, p + z * se)
        return {"mean": mean, "ci_lower": ci_lower, "ci_upper": ci_upper, "n_sim": n_sim}

    # Evaluation on test set
    def evaluate_on_test(self, sample_size: int = 2000) -> Dict:
        """
        Evaluate model performance on the test set using log loss and Brier score.
        Parameters:
            sample_size (int): Number of test samples to evaluate
        Returns:
            Dict: Dictionary with log loss, Brier score, and calibration data
        """
        if not self.trained or self.test_df is None or len(self.test_df) == 0:
            return {"error": "Model not trained or no test data"}
        # Sample test data
        test = self.test_df.sample(n=min(sample_size, len(self.test_df)), random_state=42).copy()
        probs = []
        for _, r in test.iterrows():
            surf = self._get_surface_from_row(r)
            pred = self.predict(r["player_A"], r["player_B"], surface=surf, alpha=0.5)
            probs.append(pred["p_used"])
        test["pred_prob"] = probs
        ll = log_loss(test["A_wins"], test["pred_prob"])
        br = brier_score_loss(test["A_wins"], test["pred_prob"])
        # Calibration plot data
        bins = np.linspace(0, 1, 11)
        test["bin"] = pd.cut(test["pred_prob"], bins=bins)
        calib = test.groupby("bin").agg(predicted=("pred_prob", "mean"), observed=("A_wins", "mean")).dropna().reset_index()
        return {"logloss": ll, "brier": br, "calib": calib}
"""
Limitations and Future Extensions
While effective, this model omits several important factors:

- Match format (best-of-3 vs best-of-5)
- Outdoor conditions
- Recent form and injuries
- Head-to-head matchup effects
- Uncertainty in Elo rating estimates

Future work could extend this framework using hierarchical models,
dynamic K-factors, or Bayesian Elo formulations.
"""