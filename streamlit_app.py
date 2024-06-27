import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from collections import defaultdict

# Hardcoded odds (scrape in future versions)
odds = {
    'Verstappen': {'top1': -200, 'top2': -500, 'top3': -600, 'top4': -750, 'top6': -900, 'top10': -1100},
    'Norris': {'top1': +285, 'top2': -140, 'top3': -350, 'top4': -500, 'top6': -700, 'top10': -1000},
    'Leclerc': {'top1': +1800, 'top2': +200, 'top3': -105, 'top4': -350, 'top6': -600, 'top10': -900},
    'Perez': {'top1': +2200, 'top2': +650, 'top3': +275, 'top4': +135, 'top6': -200, 'top10': -700},
    'Hamilton': {'top1': +2200, 'top2': +500, 'top3': +250, 'top4': -110, 'top6': -225, 'top10': -800},
    'Russell': {'top1': +2200, 'top2': +600, 'top3': +250, 'top4': +110, 'top6': -225, 'top10': -800},
    'Piastri': {'top1': +2500, 'top2': +850, 'top3': +300, 'top4': +200, 'top6': -180, 'top10': -700},
    'Sainz': {'top1': +2500, 'top2': +850, 'top3': +300, 'top4': +175, 'top6': -190, 'top10': -700},
    'Alonso': {'top1': +15000, 'top2': +6500, 'top3': +3500, 'top4': +700, 'top6': +225, 'top10': -200},
    'Gasly': {'top1': +25000, 'top2': +10000, 'top3': +6500, 'top4': +1200, 'top6': +600, 'top10': -140},
    'Ocon': {'top1': +25000, 'top2': +10000, 'top3': +6500, 'top4': +1200, 'top6': +600, 'top10': -140},
    'Tsunoda': {'top1': +30000, 'top2': +20000, 'top3': +15000, 'top4': +8000, 'top6': +800, 'top10': +200},
    'Stroll': {'top1': +30000, 'top2': +15000, 'top3': +10000, 'top4': +6500, 'top6': +700, 'top10': +135},
    'Ricciardo': {'top1': +30000, 'top2': +20000, 'top3': +15000, 'top4': +8000, 'top6': +800, 'top10': +225},
    'Hulkenberg': {'top1': +40000, 'top2': +25000, 'top3': +20000, 'top4': +10000, 'top6': +2500, 'top10': +320},
    'Magnussen': {'top1': +40000, 'top2': +25000, 'top3': +20000, 'top4': +10000, 'top6': +2500, 'top10': +350},
    'Albon': {'top1': +40000, 'top2': +25000, 'top3': +20000, 'top4': +10000, 'top6': +3500, 'top10': +400},
    'Bottas': {'top1': +50000, 'top2': +30000, 'top3': +25000, 'top4': +15000, 'top6': +4000, 'top10': +500},
    'Sargeant': {'top1': +50000, 'top2': +30000, 'top3': +25000, 'top4': +15000, 'top6': +5000, 'top10': +1600},
    'Zhou': {'top1': +50000, 'top2': +30000, 'top3': +25000, 'top4': +15000, 'top6': +4000, 'top10': +750}
}


def odds_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


# Convert odds to probabilities
probs = {driver: {pos: odds_to_prob(odd) for pos, odd in positions.items()}
         for driver, positions in odds.items()}

# Normalize probabilities
for pos in ['top1', 'top2', 'top3', 'top4', 'top6', 'top10']:
    total = sum(driver[pos] for driver in probs.values())
    for driver in probs:
        probs[driver][pos] /= total


def simulate_race(probs):
    result = []
    remaining = list(probs.keys())

    # Simulate 1st place
    r = np.random.random()
    cumulative = 0
    for driver in remaining:
        cumulative += probs[driver]['top1']
        if r <= cumulative:
            result.append(driver)
            remaining.remove(driver)
            break

    # Simulate 2nd place
    if remaining:
        r = np.random.random()
        cumulative = 0
        total_prob = sum(probs[d]['top2'] for d in remaining)
        for driver in remaining:
            cumulative += probs[driver]['top2'] / total_prob
            if r <= cumulative:
                result.append(driver)
                remaining.remove(driver)
                break

    # Simulate 3rd place
    if remaining:
        r = np.random.random()
        cumulative = 0
        total_prob = sum(probs[d]['top3'] for d in remaining)
        for driver in remaining:
            cumulative += probs[driver]['top3'] / total_prob
            if r <= cumulative:
                result.append(driver)
                remaining.remove(driver)
                break

    # Simulate 4th place
    if remaining:
        r = np.random.random()
        cumulative = 0
        total_prob = sum(probs[d]['top4'] for d in remaining)
        for driver in remaining:
            cumulative += probs[driver]['top4'] / total_prob
            if r <= cumulative:
                result.append(driver)
                remaining.remove(driver)
                break

    # Simulate 5th to 6th
    while len(result) < 6 and remaining:
        r = np.random.random()
        cumulative = 0
        total_prob = sum(probs[d]['top6'] for d in remaining)
        for driver in remaining:
            cumulative += probs[driver]['top6'] / total_prob
            if r <= cumulative:
                result.append(driver)
                remaining.remove(driver)
                break

    # Simulate 7th to 20th
    while remaining:
        r = np.random.random()
        cumulative = 0
        total_prob = sum(probs[d]['top10'] for d in remaining)
        for driver in remaining:
            cumulative += probs[driver]['top10'] / total_prob
            if r <= cumulative:
                result.append(driver)
                remaining.remove(driver)
                break

    # # Randomly assign remaining positions
    # result.extend(np.random.permutation(remaining))

    return result


# Streamlit app
st.title("Monte Carlo F1 Race Simulator")

# n_simulations = st.number_input("Number of Simulations", min_value=10000, max_value=100000, value=100000, step=10000)
n_simulations = 100000

# if st.button("Run Simulation"):
results = defaultdict(lambda: defaultdict(int))

for _ in range(n_simulations):
    race_result = simulate_race(probs)
    for position, driver in enumerate(race_result, 1):
        results[driver][position] += 1

# Convert results to DataFrame
df_results = pd.DataFrame(results).T.fillna(0)
df_results.columns = [f"P{i}" for i in range(1, len(df_results.columns) + 1)]
df_results['Driver'] = df_results.index

# Calculate average position
df_results['Avg Position'] = df_results.apply(
    lambda row: sum((i + 1) * row[f'P{i + 1}'] for i in range(len(odds))) / n_simulations, axis=1)

# Calculate median position
df_results['Median Position'] = df_results.apply(
    lambda row: np.median([i + 1 for i in range(len(odds)) for _ in range(int(row[f'P{i + 1}']))]), axis=1)

df_results['Win Probability'] = df_results['P1'] / n_simulations

st.subheader("Simulation Results")
st.dataframe(df_results[['Driver', 'Avg Position', 'Median Position', 'Win Probability']].sort_values('Avg Position'))

st.subheader("Finishing Position Distribution")

# Create a long-form DataFrame for Plotly
df_long = df_results.melt(id_vars=['Driver'],
                          value_vars=[f'P{i}' for i in range(1, len(odds) + 1)],
                          var_name='Position',
                          value_name='Count')
df_long['Position'] = df_long['Position'].str.replace('P', '').astype(int)

# Calculate percentages
df_long['Percentage'] = df_long['Count'] / n_simulations

# Create Plotly figure
fig = px.bar(df_long, x='Position', y='Percentage', color='Driver', barmode='group',
             title='Finishing Position Distribution (Percentage)',
             labels={'Percentage': 'Percentage', 'Position': 'Finishing Position'},
             hover_data=['Driver', 'Position', 'Percentage'])

fig.update_layout(xaxis_tickmode='linear', xaxis_dtick=1)
st.plotly_chart(fig)

# Individual driver distributions
st.subheader("Individual Driver Distributions")
driver = st.selectbox("Select a driver", df_results['Driver'])

driver_data = df_long[df_long['Driver'] == driver]
fig_driver = px.bar(driver_data, x='Position', y='Percentage',
                    title=f'Finishing Position Distribution for {driver} (Percentage)',
                    labels={'Percentage': 'Percentage (%)', 'Position': 'Finishing Position'})
fig_driver.update_layout(xaxis_tickmode='linear', xaxis_dtick=1)
st.plotly_chart(fig_driver)