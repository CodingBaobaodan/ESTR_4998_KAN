import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sb


def cap_weighted_composite_index(stock_df, start_end_string):

    closing_prices = stock_df.xs('Close', level=1, axis=1)
    volumes = stock_df.xs('Volume', level=1, axis=1)

    # Calculate the market value (price * volume) for each stock
    market_values = closing_prices * volumes

    # Calculate the total market value for each timestamp
    total_market_value = market_values.sum(axis=0)

    # Calculate the weights based on the total market value
    weights = total_market_value / total_market_value.sum()

    # Calculate the weighted composite index as the weighted sum of stock prices
    cap_weighted_composite_index = (closing_prices * weights).sum(axis=1)

    # Create a DataFrame for the composite index
    cap_weighted_composite_index_df = pd.DataFrame({'Cap Weighted Composite_Index': cap_weighted_composite_index})
    output_dir = 'csv'
    cap_weighted_composite_index_df.to_csv(f"{output_dir}/cap_weighted_composite_index_df.csv", index=True)

    output_dir = f"{start_end_string}/plots"
    # Plot the composite index
    plt.figure(figsize=(12, 6))
    plt.plot(cap_weighted_composite_index_df.index, cap_weighted_composite_index_df['Cap Weighted Composite_Index'], label="Cap Weighted Composite Index", color='blue', linewidth=2)
    plt.xticks([0, len(cap_weighted_composite_index_df.index) - 1], [start_end_string.split("_")[0], start_end_string.split("_")[1]])
    plt.title(f"Cap Weighted Composite Index of 20 Stocks ({start_end_string})")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend()
    plt.savefig(f"{output_dir}/cap_weighted_composite_index_plot.png")
    plt.close()

    return cap_weighted_composite_index_df


def cap_weighted_correlation_plots(composite_index_df, macro_df, start_end_string, k):
    # Concatenate dataframes and calculate correlation
    correlation_df = pd.concat([composite_index_df, macro_df], axis=1).corr()
    
    # Get the absolute values of the correlations with the first column
    abs_correlation = correlation_df.iloc[:, 0].abs()
    
    # Sort and select the top k absolute correlations
    top_k_correlations = abs_correlation.sort_values(ascending=False).head(k)

    # Set output directory for plots
    output_dir = f"{start_end_string}/plots"
    plt.figure(figsize=(20, 15))
    
    # Create the heatmap
    dataplot = sb.heatmap(correlation_df, cmap="YlGnBu", annot=True)  # Set annot=True to show correlation values
    plt.title('Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.savefig(f"{output_dir}/cap_weighted_correlation_plot.png")
    plt.close()

    # Set output directory for CSV
    output_dir = f"{start_end_string}/csv"
    correlation_df.to_csv(f"{output_dir}/cap_weighted_correlation_df.csv", index=True)

    return top_k_correlations