import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import io

    sns.set_theme(style="whitegrid")
    return io, mo, pd, plt, sns


@app.cell
def _(pd):
    # Load the JSON data directly by filename as it seems to be accessible.
    # This bypasses potential issues with file_content_fetcher's return type in this environment.
    df = pd.read_json('run.json')

    # Normalize the nested 'runs' data
    df = pd.json_normalize(df['runs'])

    # Convert 'timestamp' to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert 'percent_active' to numeric, replacing '%' and converting to float
    df['percent_active'] = df['percent_active'].str.replace('%', '').astype(float)

    df
    return (df,)


@app.cell
def _(df, plt, sns):
    # Plot 1: Bar plot of Mean Seed Algorithm Duration
    plt.figure(figsize=(12, 6))
    sns.barplot(x='graph', y='seed_alg_duration', hue='seed_alg', data=df, errorbar=None)
    plt.title('Duration to Find 10 seeds in Graphs')
    plt.xlabel('Graph Type')
    plt.ylabel('Seed Algorithm Duration (s)')
    plt.yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Seed Algorithm')
    plt.savefig('duration.png')
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Plot 2: Bar plot of Mean Percent Active
    plt.figure(figsize=(12, 6))
    sns.barplot(x='graph', y='percent_active', hue='seed_alg', data=df, errorbar=None)
    plt.title('Percent Active by Graph Type and Seed Algorithm')
    plt.xlabel('Graph Type')
    plt.ylabel('Amount of the Graph is Active (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Seed Algorithm')
    plt.savefig('active.png')
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
