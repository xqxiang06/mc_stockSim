# Stock data downloader with command-line options
# Supports any ticker symbol and custom date ranges

import yfinance as yf
import argparse

def download_stock_data(
    symbol,             # stock ticker
    period=None,        # Time period (1mo, 6mo, 2y, 5y...max)
    start_date=None,    # (YYYY-MM-DD)
    end_date=None,
    output_file=None    # output csv file
):

    print(f"  Stock Data Downloader - {symbol}\n")
    
    try:
        stock = yf.Ticker(symbol)
        
        # Download data based on period or date range
        if start_date and end_date:
            print(f"Downloading from {start_date} to {end_date}...")
            df = stock.history(start=start_date, end=end_date)
        elif period:
            print(f"Downloading last {period}...")
            df = stock.history(period=period)
        else:
            print("Downloading last 2 years (default)...")
            df = stock.history(period="2y")
        
        if df.empty:
            print(f"Error: No data found for {symbol}")
            return False
        
        # Process data
        df = df.reset_index()
        df['Adj Close'] = df['Close']  # yfinance Close is already adjusted
        df = df[['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']]
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        # Determine output filename
        if not output_file:
            output_file = f"{symbol}_STOCK_DATA.csv"
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Display summary
        print(f"  Ticker:     {symbol}")
        print(f"  Rows:       {len(df)}")
        print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        print(f"  Output:     {output_file}")
        
        # Latest data
        latest = df.iloc[-1]
        print(f"\nLatest ({latest['Date']}):")
        print(f"  Price:  ${latest['Adj Close']:.2f}")
        print(f"  High:   ${latest['High']:.2f}")
        print(f"  Low:    ${latest['Low']:.2f}")
        print(f"  Volume: {latest['Volume']:,.0f}")
        
        # Calculate returns
        returns = ((df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1) * 100
        print(f"\nTotal Return: {returns:+.2f}%")
        
        return True
        
    except Exception as e:
        print(f"\nError: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download stock data from Yahoo Finance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Download NVIDIA (2 years)
                python download_stock.py NVDA
                # Download Apple (1 year)
                python download_stock.py AAPL -p 1y
                # Download Tesla (custom date range)
                python download_stock.py TSLA -s 2023-01-01 -e 2024-12-31
                # Download with custom output file
                python download_stock.py MSFT -p 5y -o microsoft_data.csv
                # Download maximum history
                python download_stock.py NVDA -p max
                """
    )
    
    parser.add_argument(
        'symbol',
        help='Stock ticker symbol (e.g., NVDA, AAPL, TSLA)'
    )
    
    parser.add_argument(
        '-p', '--period',
        choices=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        default='2y',
        help='Time period to download (default: 2y)'
    )
    
    parser.add_argument(
        '-s', '--start',
        help='Start date (YYYY-MM-DD) - overrides period'
    )
    
    parser.add_argument(
        '-e', '--end',
        help='End date (YYYY-MM-DD) - overrides period'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output CSV filename (default: SYMBOL_STOCK_DATA.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate date range if provided
    if (args.start and not args.end) or (args.end and not args.start):
        print("Error: Must provide both --start and --end for date range")
        return
    
    # Download data
    success = download_stock_data(
        symbol=args.symbol.upper(),
        period=args.period if not args.start else None,
        start_date=args.start,
        end_date=args.end,
        output_file=args.output
    )
    
    if success:
        print("Next steps:")
        print("  1. Update csv_file in main.cpp if needed")
        print("  2. Compile: make clean && make")
        print("  3. Run: ./bin/monte_carlo_sim")
    else:
        print("\nTroubleshooting")


if __name__ == "__main__":
    main()
