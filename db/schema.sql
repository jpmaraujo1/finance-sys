-- prices: OHLCV time series
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT NOT NULL,
    volume FLOAT,
    PRIMARY KEY (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON prices(symbol, timestamp DESC);

-- signals: generated trading signals
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    signal_type TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    reasons JSONB NOT NULL DEFAULT '[]',
    current_price FLOAT NOT NULL,
    rsi FLOAT,
    macd FLOAT,
    win_rate FLOAT
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type, timestamp DESC);

-- news: financial news with sentiment
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    source TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    sentiment_score FLOAT NOT NULL,
    symbols JSONB NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news(timestamp DESC);

-- watchlist: assets to track
CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    display_name TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE
);
