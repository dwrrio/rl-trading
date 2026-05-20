import pandas as pd


def load_multi_stock_data():

    aapl = pd.read_csv("data/apple.csv")
    nvda = pd.read_csv("data/nvidia.csv")

    if "Adj Close" in aapl.columns:
        aapl = aapl.drop(columns=["Adj Close"])

    if "Adj Close" in nvda.columns:
        nvda = nvda.drop(columns=["Adj Close"])

    aapl["Date"] = pd.to_datetime(aapl["Date"], utc=True)
    nvda["Date"] = pd.to_datetime(nvda["Date"], utc=True)

    aapl["Date"] = aapl["Date"].dt.strftime("%Y-%m-%d")
    nvda["Date"] = nvda["Date"].dt.strftime("%Y-%m-%d")

    start_date = max(
        aapl["Date"].min(),
        nvda["Date"].min()
    )

    end_date = min(
        aapl["Date"].max(),
        nvda["Date"].max()
    )

    print("COMMON RANGE:")
    print(start_date, "→", end_date)

    aapl = aapl[
        (aapl["Date"] >= start_date) &
        (aapl["Date"] <= end_date)
    ]

    nvda = nvda[
        (nvda["Date"] >= start_date) &
        (nvda["Date"] <= end_date)
    ]

    aapl = aapl.rename(columns={

        "Open": "AAPL_Open",
        "High": "AAPL_High",
        "Low": "AAPL_Low",
        "Close": "AAPL_Close",
        "Volume": "AAPL_Volume"

    })

    nvda = nvda.rename(columns={

        "Open": "NVDA_Open",
        "High": "NVDA_High",
        "Low": "NVDA_Low",
        "Close": "NVDA_Close",
        "Volume": "NVDA_Volume"

    })

    df = pd.merge(
        aapl,
        nvda,
        on="Date",
        how="inner"
    )
    
    df = df.sort_values("Date")
    df = df.reset_index(drop=True)

    return df
