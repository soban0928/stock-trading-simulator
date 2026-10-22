"""Streamlit dashboard: browse saved simulation runs and compare them with the S&P 500.

Launch from the project root:
    streamlit run dashboard/app.py

This file only draws the page. Loading and calculations live in
dashboard/data.py and the simulator package. The dashboard only reads saved
results: it never places trades, and no real money is involved.
"""

import html
import sys
from pathlib import Path

# `streamlit run dashboard/app.py` puts only the dashboard/ folder on Python's
# import path, so add the project root to be able to import `simulator`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt  # noqa: E402  (must come after the sys.path line above)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import data  # noqa: E402
from simulator import config  # noqa: E402
from simulator.comparison import BENCHMARK_STRATEGY, results_table  # noqa: E402
from simulator.data_loader import DataError  # noqa: E402
from simulator.engine import EXECUTION_MODES, NEXT_CLOSE, SAME_CLOSE  # noqa: E402
from simulator.storage import StorageError, list_runs  # noqa: E402

SAVED_RUN_VIEW, COMPARISON_VIEW = "Saved run", "Strategy comparison"
VIEWS = [SAVED_RUN_VIEW, COMPARISON_VIEW]

# Dark "terminal" palette. The two series colours are a colourblind-safe pair.
SURFACE = "#1a1a19"
INK_SECONDARY = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"
SERIES_COLORS = {"strategy": "#3987e5", "benchmark": "#d95926"}  # blue, orange
# Non-benchmark strategies in the comparison, in a fixed order: blue, aqua, violet.
OTHER_STRATEGY_COLORS = ["#3987e5", "#199e70", "#9085e9"]

CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
  .terminal-header {{
    display: flex; justify-content: space-between; align-items: flex-end;
    flex-wrap: wrap; gap: 0.75rem;
    border-bottom: 1px solid {GRID}; padding-bottom: 0.9rem; margin-bottom: 1.2rem;
  }}
  .terminal-kicker, .section-label {{
    font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: {MUTED};
  }}
  .section-label {{ margin: 1.6rem 0 0.5rem; }}
  .terminal-title {{ font-size: 1.7rem; font-weight: 600; line-height: 1.25; }}
  .terminal-sub {{ font-size: 0.85rem; color: {INK_SECONDARY}; font-variant-numeric: tabular-nums; }}
  .sim-badge {{
    font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: {INK_SECONDARY};
    border: 1px solid {AXIS}; border-radius: 4px; padding: 0.3rem 0.6rem; white-space: nowrap;
  }}
  [data-testid="stMetricValue"] {{ font-variant-numeric: tabular-nums; }}
  [data-testid="stMetricLabel"] p {{
    text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; color: {MUTED};
  }}
</style>
"""

DISCLAIMER = (
    "Educational simulation on historical prices from Yahoo Finance (via yfinance). "
    "No real trades are placed and no real money is involved. "
    "Past performance does not predict future results."
)


# ----- loading -----------------------------------------------------------

@st.cache_data(show_spinner="Analysing run…")
def load_run(name: str, risk_free_rate: float, db_path: str, data_dir: str, db_modified: float):
    # db_modified is only part of the cache key: re-saving a run changes the
    # file's modification time, so the dashboard reloads instead of showing old data.
    return data.load_run(name, Path(db_path), Path(data_dir), risk_free_rate)


def stop_with_no_runs_message():
    st.error(
        "**No saved simulations yet.** Run a simulation and save it first, from the "
        "project folder:\n\n"
        "```bash\npython -m simulator.download      # if you have no price data\n"
        "python -m examples.buy_and_hold\n```\n\nThen refresh this page."
    )
    st.stop()


# ----- charts ------------------------------------------------------------

def time_series_chart(df, field: str, y_title: str, axis_format: str, tip_format: str, area=False):
    """Line chart of one or two series, with a hover crosshair and tooltip."""
    keys = list(dict.fromkeys(df["key"]))
    labels = list(dict.fromkeys(df["series"]))
    color = alt.Color(
        "series:N",
        scale=alt.Scale(domain=labels, range=[SERIES_COLORS[k] for k in keys]),
        # symbolOpacity=1 so the legend keys stay solid even under the faint area wash.
        legend=alt.Legend(orient="top", direction="horizontal", title=None, symbolType="stroke",
                          symbolStrokeWidth=2, symbolOpacity=1, labelColor=INK_SECONDARY),
    )
    # Lines are drawn in data order, so put the strategy last to draw it on top.
    df = df.iloc[::-1]
    base = alt.Chart(df).encode(
        x=alt.X("date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
        y=alt.Y(f"{field}:Q", title=y_title, axis=alt.Axis(format=axis_format)),
        color=color,
    )
    marks = base.mark_line(strokeWidth=2, strokeJoin="round", strokeCap="round")
    if area:
        marks = base.mark_area(opacity=0.10) + marks  # a faint wash under each line

    # Hovering anywhere snaps to the nearest date and shows every series' value.
    hover = alt.selection_point(fields=["date"], nearest=True, on="pointermove",
                                clear="pointerout", empty=False)
    tooltip = [alt.Tooltip("date:T", title="Date", format="%d %b %Y")] + [
        alt.Tooltip(f"{key}:Q", title=label, format=tip_format) for key, label in zip(keys, labels)
    ]
    crosshair = (
        alt.Chart(df)
        .transform_pivot("key", value=field, groupby=["date"])
        .mark_rule(color=MUTED, strokeWidth=1)
        .encode(x="date:T", opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    dot = dict(filled=True, size=70, stroke=SURFACE, strokeWidth=2, opacity=1)
    hover_dots = base.mark_point(**dot).transform_filter(hover)
    end_dots = (
        base.transform_window(rank="rank()", sort=[alt.SortField("date", order="descending")], groupby=["key"])
        .transform_filter("datum.rank == 1")
        .mark_point(**dot)
    )
    return terminal_style((marks + crosshair + hover_dots + end_dots).properties(height=340))


def terminal_style(chart):
    """Dark chart surface, quiet axes and hairline gridlines, shared by every chart."""
    return (
        chart
        .configure(background=SURFACE, padding={"left": 8, "right": 16, "top": 12, "bottom": 8})
        .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor=AXIS, tickColor=AXIS,
                        gridColor=GRID, labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
        .configure_axisX(grid=False)
        .configure_legend(labelFontSize=12)
        .configure_view(strokeWidth=0)
    )


# ----- page sections -----------------------------------------------------

def section(label: str):
    st.markdown(f'<div class="section-label">{html.escape(label)}</div>', unsafe_allow_html=True)


def header(run: data.RunData):
    s = run.strategy
    st.markdown(
        f"""
        <div class="terminal-header">
          <div>
            <div class="terminal-kicker">Stock Trading Simulator · Backtest</div>
            <div class="terminal-title">{html.escape(run.name)}</div>
            <div class="terminal-sub">{s.first_day} → {s.last_day} · {s.years:.1f} years ·
              {len(run.snapshots):,} trading days ·
              {len(run.trades):,} trade{"" if len(run.trades) == 1 else "s"}</div>
          </div>
          <div class="sim-badge">Simulated · Historical data · No real money</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview(run: data.RunData):
    s, comparison = run.strategy, run.comparison
    vs_spy = None
    if comparison is not None:
        vs_spy = f"{comparison.excess_return_pct * 100:+.2f} pp vs S&P 500"

    cols = st.columns(4)
    cols[0].metric("Starting cash", data.format_money(s.starting_value), border=True)
    cols[1].metric("Final portfolio value", data.format_money(s.final_value), border=True)
    cols[2].metric("Total profit/loss", data.format_money(s.total_return),
                   delta=data.format_pct(s.total_return_pct), border=True)
    cols[3].metric("Total return", data.format_pct(s.total_return_pct), delta=vs_spy, border=True,
                   help="Percentage points (pp) above or below buying the S&P 500 on the same dates.")


def risk_metrics(run: data.RunData, risk_free_rate: float):
    s = run.strategy
    cols = st.columns(4)
    cols[0].metric("Annualised return", data.format_pct(s.annualised_return), border=True,
                   help="Steady yearly return that gives the same final value (CAGR). "
                        "Unreliable for periods under a year.")
    cols[1].metric("Max drawdown", data.format_pct(-s.max_drawdown.fraction), border=True,
                   help="Largest fall from a previous high to a later low.")
    cols[2].metric("Volatility (annualised)", data.format_pct(s.volatility, signed=False), border=True,
                   help="Standard deviation of daily returns × √252.")
    cols[3].metric("Sharpe ratio", data.format_ratio(s.sharpe_ratio), border=True,
                   help=f"Return per unit of risk, above a {risk_free_rate:.2%} risk-free rate.")


def tables(run: data.RunData):
    money = st.column_config.NumberColumn(format="dollar")
    day = st.column_config.DateColumn(format="YYYY-MM-DD")
    trades_tab, history_tab = st.tabs([f"Trade history ({len(run.trades)})", "Daily portfolio history"])

    with trades_tab:
        if run.trades:
            st.dataframe(
                data.trades_table(run.trades), hide_index=True, width="stretch",
                column_config={"Date": day, "Price": money, "Commission": money,
                               "Quantity": st.column_config.NumberColumn(format="plain")},
            )
        else:
            st.info("This run made no trades.")

    with history_tab:
        history = data.history_table(run.snapshots)
        st.dataframe(
            history, hide_index=True, width="stretch", height=420,
            column_config={"Date": day, "Cash": money, "Market value": money,
                           "Total value": money, "Profit/loss": money},
        )
        st.download_button("Download as CSV", history.to_csv(index=False),
                           file_name=f"{run.name}_daily_history.csv", mime="text/csv")


# ----- strategy comparison view -------------------------------------------

@st.cache_data(show_spinner="Running every strategy over every period…", max_entries=8)
def load_comparison(execution: str, risk_free_rate: float, data_dir: str, prices_modified: float):
    # prices_modified is only part of the cache key, so re-downloaded prices are picked up.
    return data.load_comparison(Path(data_dir), execution, risk_free_rate)


def prices_modified(data_dir: Path) -> float:
    return max((f.stat().st_mtime for f in data_dir.glob("*.csv")), default=0.0)


def period_bar_chart(df, field: str, y_title: str):
    """Grouped bars: one group per period, one bar per strategy."""
    labels = list(dict.fromkeys(df["series"]))
    keys = list(dict.fromkeys(df["key"]))
    others = iter(OTHER_STRATEGY_COLORS)
    colors = [SERIES_COLORS["benchmark"] if key == BENCHMARK_STRATEGY else next(others) for key in keys]
    bars = alt.Chart(df).mark_bar(size=22, cornerRadiusEnd=4).encode(
        x=alt.X("period:N", title=None, sort=list(dict.fromkeys(df["period"])), axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("series:N", sort=labels, scale=alt.Scale(paddingInner=0.15)),
        y=alt.Y(f"{field}:Q", title=y_title, axis=alt.Axis(format=".0%")),
        color=alt.Color("series:N", scale=alt.Scale(domain=labels, range=colors),
                        legend=alt.Legend(orient="top", direction="horizontal", title=None,
                                          labelColor=INK_SECONDARY)),
        tooltip=[alt.Tooltip("period:N", title="Period"), alt.Tooltip("series:N", title="Strategy"),
                 alt.Tooltip(f"{field}:Q", title=y_title, format="+.2%")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=AXIS).encode(y="y:Q")
    return terminal_style((bars + zero).properties(height=300))


def comparison_view():
    with st.sidebar:
        execution = st.segmented_control(
            "Execution", EXECUTION_MODES, default=SAME_CLOSE, required=True,
            format_func={SAME_CLOSE: "Same-day close", NEXT_CLOSE: "Next-day close"}.get,
            help="When orders fill. Every strategy, including the benchmark, uses the same setting.",
        )
        rate_pct = risk_free_input()
    risk_free_rate = rate_pct / 100

    st.markdown(
        """
        <div class="terminal-header">
          <div>
            <div class="terminal-kicker">Stock Trading Simulator · Robustness</div>
            <div class="terminal-title">Strategy comparison</div>
            <div class="terminal-sub">Same trading days · $10,000 start · $1.00 commission ·
              same execution for every strategy</div>
          </div>
          <div class="sim-badge">Simulated · Historical data · No real money</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_dir = Path(config.PRICE_DATA_DIR)
    try:
        results = load_comparison(execution, risk_free_rate, str(data_dir), prices_modified(data_dir))
    except DataError as error:
        st.error(f"{error}. Download prices with `python -m simulator.download`.",
                 title="Price data unavailable")
        st.stop()

    st.info(
        "Each period is a fresh simulation that starts in cash on its first trading day. "
        "Indicators may look back at earlier prices to warm up, but never at later ones. "
        "Because every period starts fresh, a strategy's result can differ from the same "
        "dates inside a longer run. Strong past results on one market's history don't show "
        "that a strategy is profitable or robust.",
        title="How to read this", icon=":material/info:",
    )

    chart_data = data.comparison_chart_data(results)
    left, right = st.columns(2, gap="large")
    with left:
        section("Annualised return by period")
        st.altair_chart(period_bar_chart(chart_data, "annualised_return", "Annualised return"),
                        width="stretch", theme=None, alt="Annualised return of each strategy by period")
    with right:
        section("Max drawdown by period")
        st.altair_chart(period_bar_chart(chart_data, "max_drawdown", "Max drawdown"),
                        width="stretch", theme=None, alt="Maximum drawdown of each strategy by period")

    for pr in results:
        section(f"{pr.period.name} · {pr.first_day} → {pr.last_day} · {pr.trading_days:,} trading days · "
                f"{pr.history_days_before:,} earlier days for warm-up")
        st.dataframe(data.comparison_period_table(pr), hide_index=True, width="stretch",
                     alt=f"Strategy metrics for {pr.period.name}")
        for r in pr.results:
            if r.start_note:
                st.caption(f":material/info: {data.strategy_label(r.strategy)}: {r.start_note}")

    st.download_button("Download as CSV", results_table(results).to_csv(index=False),
                       file_name=f"strategy_comparison_{execution}.csv", mime="text/csv")
    st.divider()
    st.caption(DISCLAIMER)


def risk_free_input() -> float:
    return st.number_input("Risk-free rate (% a year)", min_value=0.0, max_value=20.0,
                           value=config.RISK_FREE_RATE * 100, step=0.25,
                           help="Used only for the Sharpe ratio.")


# ----- the page ----------------------------------------------------------

def main():
    st.set_page_config(page_title="Backtest Terminal", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        view = st.segmented_control("View", VIEWS, default=VIEWS[0], required=True)
    if view == COMPARISON_VIEW:
        comparison_view()
    else:
        saved_run_view()


def saved_run_view():
    db_path = Path(config.DATABASE_PATH)
    try:
        runs = list_runs(db_path)
    except StorageError:
        stop_with_no_runs_message()
    if not runs:
        stop_with_no_runs_message()

    with st.sidebar:
        st.markdown("### Simulation run")
        name = st.selectbox("Saved run", runs, index=len(runs) - 1,
                            help="Runs saved with simulator.storage.save_simulation().")
        rate_pct = risk_free_input()
        st.caption(f"Database: `{db_path.name}` · Benchmark: {config.BENCHMARK} buy-and-hold")
        st.caption(DISCLAIMER)
    risk_free_rate = rate_pct / 100

    try:
        run = load_run(name, risk_free_rate, str(db_path), str(config.PRICE_DATA_DIR),
                       db_path.stat().st_mtime)
    except (StorageError, ValueError) as error:
        st.error(f"**Could not load run {name!r}:** {error}")
        st.stop()

    header(run)
    description = data.strategy_description(run.name)
    if description:
        st.caption(f":material/insights: {description}")
    if run.benchmark_error:
        st.warning(
            f"**S&P 500 benchmark unavailable:** {run.benchmark_error}. "
            "Showing the strategy only. Download prices with `python -m simulator.download`."
        )
    if run.strategy.years < 1:
        st.info("Less than a year of data: annualised figures can be misleading.")

    overview(run)

    section("Equity curve · portfolio value at each day's close")
    st.altair_chart(
        time_series_chart(data.equity_curve(run), "value", "Portfolio value", "$,.0f", "$,.2f"),
        width="stretch", theme=None,
    )

    section("Risk & return")
    risk_metrics(run, risk_free_rate)

    left, right = st.columns([3, 2], gap="large")
    with left:
        section("Drawdown · decline from previous peak")
        st.altair_chart(
            time_series_chart(data.drawdown_curve(run), "drawdown", "Below peak", ".0%", ".2%", area=True),
            width="stretch", theme=None,
        )
    with right:
        section(f"Strategy vs {data.BENCHMARK_LABEL}")
        st.dataframe(data.metrics_table(run), hide_index=True, width="stretch")
        for label, perf in (("Strategy", run.strategy), (data.BENCHMARK_LABEL, run.benchmark)):
            if perf and perf.max_drawdown.peak_day:
                st.caption(f"{label} worst drawdown: peak {perf.max_drawdown.peak_day}, "
                           f"low {perf.max_drawdown.trough_day}")

    section("Records")
    tables(run)

    st.divider()
    st.caption(DISCLAIMER)


main()
