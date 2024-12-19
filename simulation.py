import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# The TSX Total Return Index is the S&P/TSX Composite Total Return
# The US Bond Return is the Barclays US Aggregate Bond Index
# The SPY Total Return Index is the S&P 500
# The FTSE Canada Universe Bond Index Series is not available to us, so right now the Canadian Bond Returns are replaced
# By a placeholder which is just the US bond returns.
# Each of the above is available for purchase - The S&P/TSX Composite and S&P 500 are available from S&P
# The US Bond returns are available from Barclays
# The FTSE Canada Unniverse bond index is available from the LSGE
# We obtain this via our institutional access to Bloomberg

tsx_returns = pd.read_csv("tsxrf.csv")
usbond_returns = pd.read_csv("usbrf.csv")
spy_returns = pd.read_csv("spyrf.csv")
canbond_returns = usbond_returns.copy()

# Take input from user on simulation inputs
Year = int(input("Enter the number of years: "))
initial_investment = float(input("Enter the initial investment amount: "))
annual_contribution = float(input("Enter the annual contribution: "))

# Take input from user on portfolio weights
tsx_weight = float(input("Enter percentage of portfolio in TSX: ")) / 100
us_bond_weight = float(input("Enter percentage of portfolio in US Bonds: ")) / 100
can_bond_weight = (
    float(input("Enter percentage of portfolio in Canadian Bonds: ")) / 100
)
spy_weight = float(input("Enter percentage of portfolio in SPY: ")) / 100

# Check if the weights sum up to 1, throw an error if not
total_weight = tsx_weight + us_bond_weight + can_bond_weight + spy_weight
if abs(total_weight - 1) > 0.0001:
    print("Error: Portfolio weights must sum up to 100%.")
    exit()

# Write weights to dict
portfolio_weights = {
    "tsx": tsx_weight,
    "us_bonds": us_bond_weight,
    "can_bonds": can_bond_weight,
    "spy": spy_weight,
}


# function to perform a single simulation
def simulate_cumulative_return(sample_size, returns_data):
    # Randomly select `sample_size` returns
    sample = (
        returns_data[["return"]]
        .sample(n=sample_size, replace=True)
        .reset_index(drop=True)
    )

    # Calculate cumulative return
    sample["r1"] = 1 + sample["return"]
    sample["ln_r1"] = np.log(sample["r1"])
    sample["cum_ln_return"] = sample["ln_r1"].cumsum()
    sample["cumulative_factor"] = np.exp(sample["cum_ln_return"])
    sample["cumulative_return"] = sample["cumulative_factor"] - 1

    # Return only the final cumulative return
    return sample["cumulative_return"].iloc[-1]


# Function to run simulations for an asset
def run_simulations_for_asset(returns_data, sample_size, num_simulations=1000):
    # Run simulations
    sim_results = [
        simulate_cumulative_return(sample_size, returns_data)
        for _ in range(num_simulations)
    ]

    # Convert to DataFrame and Winsorize the top and bottom 10%
    sim_df = pd.DataFrame(sim_results, columns=["cumulative_return"])
    lower_limit = sim_df["cumulative_return"].quantile(0.10)
    upper_limit = sim_df["cumulative_return"].quantile(0.90)
    sim_df["cumulative_return"] = sim_df["cumulative_return"].clip(
        lower=lower_limit, upper=upper_limit
    )

    # Calculate statistics
    mean_return = sim_df["cumulative_return"].mean()
    min_return = sim_df["cumulative_return"].min()
    max_return = sim_df["cumulative_return"].max()

    # Return the statistics
    return {
        "sample_size": sample_size,
        "mean": mean_return,
        "min": min_return,
        "max": max_return,
    }


# Initialize a dictionary to hold the simulation results for each asset
simulation_results = {}

# List of assets and their returns data
assets = {
    "tsx": tsx_returns,
    "us_bonds": usbond_returns,
    "can_bonds": canbond_returns,
    "spy": spy_returns,
}

# Run simulations for each asset
for asset_name, returns_data in assets.items():
    results = run_simulations_for_asset(
        returns_data, sample_size=Year, num_simulations=1000
    )
    simulation_results[asset_name] = results


# Calculate annualized return (AR) for min, mean, and max scenarios for each asset
ar_results = {}
for asset_name, result in simulation_results.items():
    min_cum_return = result["min"]
    mean_cum_return = result["mean"]
    max_cum_return = result["max"]

    AR_min = (1 + min_cum_return) ** (1 / Year) - 1
    AR_mean = (1 + mean_cum_return) ** (1 / Year) - 1
    AR_max = (1 + max_cum_return) ** (1 / Year) - 1

    ar_results[asset_name] = {"AR_min": AR_min, "AR_mean": AR_mean, "AR_max": AR_max}

# Initialize DataFrame for the cumulative return and investment growth
years = pd.DataFrame({"year": range(1, Year + 1)})

# For each asset, calculate investment values over time
investment_values = {}
for scenario in ["min", "mean", "max"]:
    investment_values[scenario] = pd.DataFrame({"year": years["year"]})
    for asset_name in assets.keys():
        # Get the annualized return for this scenario
        AR = ar_results[asset_name][f"AR_{scenario}"]
        weight = portfolio_weights[asset_name]
        # Initialize investment value
        invest_values = np.zeros(Year)
        # Set initial investment for the asset
        initial_asset_investment = (initial_investment * weight) + (
            annual_contribution * weight
        )
        invest_values[0] = initial_asset_investment * (1 + AR)
        # Calculate investment values over time for the asset
        for t in range(1, Year):
            annual_contribution_asset = annual_contribution * weight
            invest_values[t] = (invest_values[t - 1] + annual_contribution_asset) * (
                1 + AR
            )
        # Add to DataFrame
        investment_values[scenario][asset_name] = invest_values

# Sum the investment values across assets for each scenario
years["invest_value_min"] = investment_values["min"][assets.keys()].sum(axis=1)
years["invest_value_mean"] = investment_values["mean"][assets.keys()].sum(axis=1)
years["invest_value_max"] = investment_values["max"][assets.keys()].sum(axis=1)

# Reshape data for plotting
investment_values_melted = years[
    ["year", "invest_value_min", "invest_value_mean", "invest_value_max"]
].melt(id_vars="year", var_name="scenario", value_name="investment_value")

# Plot investment values over time
plt.figure(figsize=(10, 6))
for scenario, color, style in zip(
    ["invest_value_mean", "invest_value_min", "invest_value_max"],
    ["blue", "red", "green"],
    ["solid", "dashed", "dotted"],
):
    data = investment_values_melted[investment_values_melted["scenario"] == scenario]
    plt.plot(
        data["year"],
        data["investment_value"],
        label=scenario.replace("invest_value_", "").capitalize(),
        color=color,
        linestyle=style,
    )

# Display ending values as text on the plot
mean_end_value = years["invest_value_mean"].iloc[-1]
min_end_value = years["invest_value_min"].iloc[-1]
max_end_value = years["invest_value_max"].iloc[-1]

plt.text(
    Year,
    mean_end_value,
    f"Ending Value: ${mean_end_value:,.0f}",
    color="blue",
    ha="right",
    va="bottom",
)
plt.text(
    Year,
    min_end_value,
    f"Ending Value: ${min_end_value:,.0f}",
    color="red",
    ha="right",
    va="top",
)
plt.text(
    Year,
    max_end_value,
    f"Ending Value: ${max_end_value:,.0f}",
    color="green",
    ha="right",
    va="bottom",
)

plt.xlabel("Year")
plt.ylabel("Investment Value ($)")
plt.title("Investment Growth Over Time")
plt.legend()
plt.grid(True)
plt.show()

# weighted annualized returns for inspection
print(
    f"Weighted Annualized Return (Min): {sum(ar_results[asset]['AR_min'] * portfolio_weights[asset] for asset in assets.keys()):.4%}"
)
print(
    f"Weighted Annualized Return (Mean): {sum(ar_results[asset]['AR_mean'] * portfolio_weights[asset] for asset in assets.keys()):.4%}"
)
print(
    f"Weighted Annualized Return (Max): {sum(ar_results[asset]['AR_max'] * portfolio_weights[asset] for asset in assets.keys()):.4%}"
)

# individual asset annualized returns for inspection
print("\nAnnualized Returns for Each Asset:")
for asset_name in assets.keys():
    print(f"\nAsset: {asset_name.capitalize()}")
    print(f"  AR_min: {ar_results[asset_name]['AR_min']:.4%}")
    print(f"  AR_mean: {ar_results[asset_name]['AR_mean']:.4%}")
    print(f"  AR_max: {ar_results[asset_name]['AR_max']:.4%}")
