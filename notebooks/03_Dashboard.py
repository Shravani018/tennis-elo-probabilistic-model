"""
Tennis ELO Dashboard with Monte Carlo Simulation

To explore head to head tennis match outcomes using an Elo based probabilistic model.
Monte Carlo simulation is used to demonstrate convergence properties and simulate tournament outcomes.
"""
# Importing necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# Importing the EloModel from model.py
from model import EloModel
# Setting up the Streamlit app
st.set_page_config(page_title="Tennis Elo Dashboard + Monte Carlo", layout="wide", page_icon="🎾")
st.title("🎾 Tennis Elo Dashboard with Monte Carlo Simulation")
# Sidebar inputs
st.sidebar.header("Data & Model Parameters")
data_path = "./atp_matches_processed.csv"
initial_elo = st.sidebar.number_input("Initial Elo", value=1500)
k_factor = st.sidebar.number_input("K-factor", value=25)
split_date = st.sidebar.text_input("Train/Test split date", value="2022-01-01")
n_sim=10_000
alpha=st.sidebar.slider("Surface weight (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
# Loading and caching the model
@st.cache_data
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    return df
try:
    df = load_df(data_path)
except Exception as e:
    st.error(f"Could not load CSV at {data_path}: {e}")
    st.stop()

# Training the Elo model
model = EloModel(initial_elo=initial_elo, k_factor=k_factor, split_date=split_date)
with st.spinner("Training Elo model..."):
    model.fit(df)

# Selecting players and surface
players = sorted(set(df["player_A"]).union(df["player_B"]))
surfaces = sorted(list(model.surface_elos.keys()))
if not surfaces:
    surfaces = ["Hard", "Clay", "Grass", "Unknown"]
col_top = st.columns([3, 2])
with col_top[0]:
    st.subheader("Match prediction inputs")
    player_a = st.selectbox("Player A", players, index=0)
    player_b = st.selectbox("Player B", [p for p in players if p != player_a] or players, index=0)
    surface = st.selectbox("Surface", surfaces, index=0)

with col_top[1]:
    st.subheader("Simulation inputs")
    n_sim = st.number_input("Number of Monte Carlo sims (for tournament sim)", min_value=1000, max_value=100000, value=n_sim, step=1000)
    tournament_size = st.number_input("Matches per tournament (for tournament sim)", min_value=10, max_value=1000, value=100, step=10)
    n_tournaments = st.number_input("Number of tournaments to simulate", min_value=10, max_value=1000, value=100, step=10)

st.markdown("---")

# Prediction and simulation
pred = model.predict(player_a, player_b, surface=surface, alpha=alpha)
sim_result = model.simulate(pred["p_used"], n_sim=n_sim)
st.subheader("Prediction Summary")

stats_col, pie_col = st.columns([2, 1])
with stats_col:
    st.markdown(f"**{player_a} win probability**")
    st.markdown(
        f"<h1 style='margin-bottom:10px'>{pred['p_used']*100:.2f}%</h1>",
        unsafe_allow_html=True
    )

    st.write(f"Global prob: {pred['p_global']*100:.2f}%")
    st.write(
        f"Surface prob: {pred['p_surface']*100:.2f}%"
        if pred["p_surface"] is not None else
        "Surface prob: N/A"
    )
    st.write(f"Simulated mean (n={sim_result['n_sim']:,}): {sim_result['mean']*100:.2f}%")
    st.write(
        f"Approx 95% CI: "
        f"[{sim_result['ci_lower']*100:.2f}%, {sim_result['ci_upper']*100:.2f}%]"
    )
with pie_col:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=[player_a, player_b],
                values=[pred["p_used"], 1 - pred["p_used"]],
                hole=0.45
            )
        ]
    )
    fig.update_traces(
        textinfo="percent",
        hoverinfo="label+percent"
    )
    fig.update_layout(
        title="Win / Loss",
        margin=dict(t=40, b=0, l=0, r=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
# Monte Carlo convergence demonstration
st.header("Monte Carlo Convergence Demonstration")
st.markdown(
    """
    Shows how the Monte Carlo win rate converges to the input probability as the number of simulations increases.
    """
)
#
col_c1, col_c2 = st.columns([2, 1])
with col_c1:
    use_elo_p = st.checkbox("Use Elo-derived p (from selected players)", value=True)
    p_true = None
    if not use_elo_p:
        p_true = st.sidebar.number_input("Custom p_true (0..1)", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
    else:
        p_true = float(pred["p_used"])
    convergence=[10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    sizes = sorted(convergence)
    run_conv = st.button("Run convergence demo")

with col_c2:
    st.write("Selected probability")
    st.metric("p_true", f"{p_true:.3f}")

if run_conv:
    # Plotting convergence
    np.random.seed(42)
    simulated_probs = []
    for n in sizes:
        sims = np.random.binomial(1, p_true, size=n)
        simulated_probs.append(sims.mean())
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sizes, simulated_probs, marker='o', linewidth=2, markersize=6, label='Simulated win rate')
    ax.axhline(p_true, color='red', linestyle='--', linewidth=2, label=f'True probability ({p_true:.3f})')
    ax.set_xscale('log')
    ax.set_xlabel('Number of simulations (log scale)')
    ax.set_ylabel('Simulated win rate')
    ax.set_title('Monte Carlo Convergence')
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Display last two simulated probabilities
    last_two = simulated_probs[-2:] if len(simulated_probs) >= 2 else simulated_probs
    st.write(f"Simulated win rate at n={sizes[-2]}: {last_two[0]:.4f}" if len(last_two) >= 2 else "")
    st.write(f"Simulated win rate at n={sizes[-1]}: {last_two[-1]:.4f}")
    st.success("Simulation complete — as expected the simulated rate converges toward the input probability.")

st.markdown("---")

# Tournament simulation
st.header("Tournament Simulation: variability across repeated tournaments")
st.markdown(
    """
    Simulates multiple independent tournaments and records how often Player A wins. 
    This highlights variability around the expected number of wins given the model probability.
    """
)

col_t1, col_t2 = st.columns([2, 1])
with col_t1:
    run_tourn = st.button("Run tournament simulation")

with col_t2:
    st.write(f"Matches per tournament: {tournament_size}")
    st.write(f"Number of tournaments: {n_tournaments}")
    st.write(f"Using probability p = {pred['p_used']:.3f}")
# Function to run tournament simulation
def run_tournament_simulation(p, tournament_size, n_tournaments, seed=42):
    rng = np.random.RandomState(seed)
    wins = rng.binomial(n=tournament_size, p=p, size=n_tournaments)
    return wins

if run_tourn:
    wins = run_tournament_simulation(pred["p_used"], int(tournament_size), int(n_tournaments), seed=42)
    # Plotting histogram of wins
    plt.style.use("dark_background")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.hist(wins, bins=30, edgecolor='black', alpha=0.7)
    expected = pred["p_used"] * tournament_size
    ax2.axvline(expected, color='red', linestyle='--', linewidth=2, label=f'Expected wins: {expected:.1f}')
    ax2.set_xlabel(f'Number of wins for {player_a} (out of {tournament_size})')
    ax2.set_ylabel('Frequency (number of simulated tournaments)')
    ax2.set_title(f'Simulated distribution of wins: {player_a} vs {player_b}')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    st.pyplot(fig2)

    # Summary statistics
    median_wins = int(np.median(wins))
    p2_5 = int(np.percentile(wins, 2.5))
    p97_5 = int(np.percentile(wins, 97.5))
    st.write(f"Median wins for {player_a}: **{median_wins}**")
    st.write(f"95% of tournaments: **{p2_5}** to **{p97_5}** wins")
    st.write(f"Observed range: **{wins.min()}** to **{wins.max()}** wins")
    st.success("Tournament simulation complete.")
st.markdown("---")
st.subheader("Notes & caveats")
st.markdown(
    """
    - The Elo model used here is a simplified representation and may not capture all nuances of tennis match outcomes.
    - Monte Carlo sampling approximates the input probability and does not model Elo parameter uncertainty.
    - The model assumes independence between matches, which may not hold in real tournament settings.
    - Surface-specific Elo ratings help account for player performance variations across different court types.
    - Users should interpret simulation results in the context of model limitations and real-world variability.
    """
)