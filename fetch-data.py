import pandas as pd
import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

tickers = {
	"copper": "HG=F",
	"gold": "GC=F",
	"sp500": "^GSPC",
	"kospi": "^KS11",
	"us10y": "^TNX",
	"dxy": "DX-Y.NYB",
	"wti": "CL=F",
}

start = "2006-01-01"
end = None 

series = []

for name, ticker in tickers.items():
	df = yf.download(ticker, start=start, end=end, interval="1mo", auto_adjust=False, progress=False)
	if df.empty:
		print(f"[WARN] No data for {name} ({ticker})")
		continue

	# keep Adj Close if available, else Close
	col = "Adj Close" if "Adj Close" in df.columns else "Close"
	s = df[col].copy()
	s.name = name

	out = s.reset_index()
	out.columns = ["Date", name]
	out["Date"] = pd.to_datetime(out["Date"]).dt.to_period("M").dt.to_timestamp()
	out.to_csv(RAW_DIR / f"{name}.csv", index=False)

	series.append(out.set_index("Date")[name])

# merge all series
master = pd.concat(series, axis=1).sort_index()
master.to_csv(PROCESSED_DIR / "master_monthly.csv")

print("Saved:")
print(f"- Raw files in {RAW_DIR}")
print(f"- Master file: {PROCESSED_DIR / 'master_monthly.csv'}")
print(master.tail())