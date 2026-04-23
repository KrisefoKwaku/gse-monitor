# -*- coding: utf-8 -*-
"""
GSE Stock Monitor — Hugging Face Spaces deployment
"""
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from datetime import date, timedelta

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model

# ── Data loading ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'GSE_Stock_Data.xlsx')

xls    = pd.ExcelFile(DATA_PATH, engine='openpyxl')
frames = []
for sheet in xls.sheet_names:
    df  = pd.read_excel(xls, sheet_name=sheet, engine='openpyxl')
    req = {
        'Daily Date', 'Share Code',
        'Opening Price (GH¢)', 'Year High (GH¢)',
        'Year Low (GH¢)', 'Closing Price - VWAP (GH¢)'
    }
    cols = list(req)
    if 'Total Shares Traded' in df.columns:
        cols.append('Total Shares Traded')
    if req.issubset(df.columns):
        frames.append(df[cols])

stock_data = pd.concat(frames, ignore_index=True)
stock_data.dropna(subset=[
    'Daily Date', 'Share Code',
    'Opening Price (GH¢)', 'Year High (GH¢)',
    'Year Low (GH¢)', 'Closing Price - VWAP (GH¢)'
], inplace=True)
stock_data['Daily Date'] = pd.to_datetime(
    stock_data['Daily Date'], dayfirst=True, format='mixed'
)
has_volume = 'Total Shares Traded' in stock_data.columns
stocks     = sorted(stock_data['Share Code'].unique())

PRESETS = {'1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, 'All': None}

# ── Scoring engine ────────────────────────────────────────────────────────────
def score_stock(stock):
    """
    Score a stock across its full history using multiple indicators.
    Returns a dict with composite buy_score, sell_score, and signal breakdown.
    """
    raw = (stock_data[stock_data['Share Code'] == stock]
           .sort_values('Daily Date')
           .drop_duplicates(subset='Daily Date', keep='last'))

    if len(raw) < 30:
        return None

    d = compute_indicators(raw)
    c = d['Closing Price - VWAP (GH¢)']

    scores = {}

    # ── 1. RSI score ──────────────────────────────────────────────────────────
    latest_rsi = d['RSI'].iloc[-1]
    if pd.isna(latest_rsi):
        scores['rsi'] = 0
    elif latest_rsi < 30:
        scores['rsi'] = 3        # strongly oversold → buy
    elif latest_rsi < 45:
        scores['rsi'] = 1
    elif latest_rsi > 70:
        scores['rsi'] = -3       # strongly overbought → sell
    elif latest_rsi > 55:
        scores['rsi'] = -1
    else:
        scores['rsi'] = 0

    # ── 2. MACD score ─────────────────────────────────────────────────────────
    latest_macd = d['MACD'].iloc[-1]
    latest_sig  = d['Signal'].iloc[-1]
    prev_macd   = d['MACD'].iloc[-2] if len(d) > 1 else latest_macd
    prev_sig    = d['Signal'].iloc[-2] if len(d) > 1 else latest_sig
    if pd.isna(latest_macd) or pd.isna(latest_sig):
        scores['macd'] = 0
    elif latest_macd > latest_sig and prev_macd <= prev_sig:
        scores['macd'] = 3       # fresh bullish crossover
    elif latest_macd > latest_sig:
        scores['macd'] = 1       # bullish but not fresh
    elif latest_macd < latest_sig and prev_macd >= prev_sig:
        scores['macd'] = -3      # fresh bearish crossover
    elif latest_macd < latest_sig:
        scores['macd'] = -1
    else:
        scores['macd'] = 0

    # ── 3. MA trend score ─────────────────────────────────────────────────────
    ma20 = d['MA20'].iloc[-1]
    ma50 = d['MA50'].iloc[-1]
    if pd.isna(ma20) or pd.isna(ma50):
        scores['ma'] = 0
    elif ma20 > ma50:
        scores['ma'] = 2         # uptrend
    else:
        scores['ma'] = -2        # downtrend

    # ── 4. Bollinger position score ───────────────────────────────────────────
    bb_upper = d['BB_upper'].iloc[-1]
    bb_lower = d['BB_lower'].iloc[-1]
    price    = c.iloc[-1]
    if pd.isna(bb_upper) or pd.isna(bb_lower):
        scores['bb'] = 0
    elif price <= bb_lower:
        scores['bb'] = 2         # at lower band → potential bounce
    elif price >= bb_upper:
        scores['bb'] = -2        # at upper band → potential reversal
    else:
        bb_mid = (bb_upper + bb_lower) / 2
        scores['bb'] = 1 if price < bb_mid else -1

    # ── 5. Price momentum score (5-day return) ────────────────────────────────
    if len(c) >= 6:
        ret_5d = (c.iloc[-1] - c.iloc[-6]) / c.iloc[-6] * 100
        if ret_5d > 5:
            scores['momentum'] = 2
        elif ret_5d > 1:
            scores['momentum'] = 1
        elif ret_5d < -5:
            scores['momentum'] = -2
        elif ret_5d < -1:
            scores['momentum'] = -1
        else:
            scores['momentum'] = 0
    else:
        scores['momentum'] = 0

    # ── 6. Buy/Sell signal frequency (last 20 days) ───────────────────────────
    recent      = d.tail(20)
    buy_count   = recent['Buy_Signal'].sum()
    sell_count  = recent['Sell_Signal'].sum()
    scores['signals'] = int(buy_count) - int(sell_count)

    # ── 7. 52W position score ─────────────────────────────────────────────────
    w52_high = d['52W_High'].iloc[-1]
    w52_low  = d['52W_Low'].iloc[-1]
    if w52_high != w52_low:
        pct_range = (price - w52_low) / (w52_high - w52_low)
        if pct_range < 0.25:
            scores['w52'] = 2    # near 52W low → potential value
        elif pct_range > 0.85:
            scores['w52'] = -2   # near 52W high → stretched
        else:
            scores['w52'] = 0
    else:
        scores['w52'] = 0

    total      = sum(scores.values())
    max_score  = 15
    buy_score  = max(0, total)
    sell_score = max(0, -total)
    pct_buy    = min(100, round(buy_score  / max_score * 100))
    pct_sell   = min(100, round(sell_score / max_score * 100))

    latest_close = c.iloc[-1]
    prev_close   = c.iloc[-2] if len(c) > 1 else latest_close
    chg          = latest_close - prev_close
    chg_pct      = (chg / prev_close * 100) if prev_close != 0 else 0

    return {
        'stock':       stock,
        'total':       total,
        'buy_score':   pct_buy,
        'sell_score':  pct_sell,
        'scores':      scores,
        'rsi':         round(latest_rsi, 1) if not pd.isna(latest_rsi) else None,
        'price':       latest_close,
        'chg':         chg,
        'chg_pct':     chg_pct,
        'ma_trend':    'Up' if scores.get('ma', 0) > 0 else 'Down',
        'macd_signal': ('Bullish Cross' if scores.get('macd', 0) == 3
                        else 'Bearish Cross' if scores.get('macd', 0) == -3
                        else 'Bullish'      if scores.get('macd', 0) > 0
                        else 'Bearish'),
    }


def get_recommendations():
    """Score all stocks and return top 2 buys and top 2 sells."""
    results = []
    for s in stocks:
        r = score_stock(s)
        if r:
            results.append(r)

    results.sort(key=lambda x: x['total'], reverse=True)
    top_buys  = [r for r in results if r['total'] > 0][:2]
    top_sells = sorted(
        [r for r in results if r['total'] < 0],
        key=lambda x: x['total']
    )[:2]
    return top_buys, top_sells


# ── Indicators ────────────────────────────────────────────────────────────────
def compute_indicators(df):
    c  = df['Closing Price - VWAP (GH¢)'].copy()
    o  = df['Opening Price (GH¢)'].copy()
    df = df.copy()
    df['MA20']      = c.rolling(20).mean()
    df['MA50']      = c.rolling(50).mean()
    df['BB_mid']    = c.rolling(20).mean()
    df['BB_upper']  = df['BB_mid'] + 2 * c.rolling(20).std()
    df['BB_lower']  = df['BB_mid'] - 2 * c.rolling(20).std()
    delta           = c.diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = (-delta.clip(upper=0)).rolling(14).mean()
    rs              = gain / loss.replace(0, np.nan)
    df['RSI']       = 100 - (100 / (1 + rs))
    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df['MACD']      = ema12 - ema26
    df['Signal']    = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['Signal']
    df['PctChange'] = ((c - o) / o * 100).round(2)
    df['52W_High']  = c.rolling(252, min_periods=1).max()
    df['52W_Low']   = c.rolling(252, min_periods=1).min()
    df['Gap']       = o - c.shift(1)
    ma20, ma50      = df['MA20'], df['MA50']
    cross_up        = (ma20 > ma50) & (ma20.shift(1) <= ma50.shift(1))
    cross_down      = (ma20 < ma50) & (ma20.shift(1) >= ma50.shift(1))
    df['Buy_Signal']  = cross_up   & (df['RSI'] < 60)
    df['Sell_Signal'] = cross_down & (df['RSI'] > 40)
    return df


# ── ARIMA + GARCH forecast ────────────────────────────────────────────────────
def _best_arima_order(series):
    d = 0
    s = series.copy()
    for _ in range(2):
        if adfuller(s.dropna(), autolag='AIC')[1] < 0.05:
            break
        s = s.diff().dropna()
        d += 1
    best_aic, best_order = np.inf, (1, d, 0)
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_order = m.aic, (p, d, q)
            except Exception:
                continue
    return best_order


def forecast_prices(series, horizon=5):
    series  = series.dropna()
    log_px  = np.log(series)
    returns = log_px.diff().dropna() * 100
    try:
        order       = _best_arima_order(log_px)
        arima_fc    = ARIMA(log_px, order=order).fit().forecast(steps=horizon)
        model_label = f"ARIMA{order}"
    except Exception:
        arima_fc    = pd.Series([log_px.iloc[-1]] * horizon,
                                index=range(horizon))
        model_label = "Random Walk"
    try:
        garch_fit    = arch_model(returns, vol='Garch', p=1, q=1,
                                  dist='normal', rescale=False).fit(
                                      disp='off', show_warning=False)
        vol_forecast = np.sqrt(
            garch_fit.forecast(horizon=horizon,
                               reindex=False).variance.values[-1]
        ) / 100
    except Exception:
        vol_forecast = np.full(horizon, returns.std() / 100)

    cum_vol  = np.array([np.sqrt(np.sum(vol_forecast[:i+1]**2))
                         for i in range(horizon)])
    fc_vals  = np.array(arima_fc)
    scale    = series.iloc[-1] / np.exp(log_px.iloc[-1])
    point    = np.exp(fc_vals) * scale
    lower_80 = np.exp(fc_vals - 1.282 * cum_vol) * scale
    upper_80 = np.exp(fc_vals + 1.282 * cum_vol) * scale
    lower_95 = np.exp(fc_vals - 1.960 * cum_vol) * scale
    upper_95 = np.exp(fc_vals + 1.960 * cum_vol) * scale
    last_date      = pd.to_datetime(series.index[-1])
    forecast_dates = pd.bdate_range(
        start=last_date, periods=horizon + 1, freq='C')[1:]
    return (forecast_dates, point, lower_80, upper_80,
            lower_95, upper_95, model_label)


def add_forecast_traces(fig, series, row=1, label='', color='#E0E0E0'):
    try:
        fc_dates, pt, lo80, hi80, lo95, hi95, lbl = \
            forecast_prices(series, horizon=5)
    except Exception as e:
        print(f'Forecast failed for {label}: {e}')
        return ''
    last_date  = series.index[-1]
    last_price = series.iloc[-1]
    x_line  = [last_date] + list(fc_dates)
    y_point = [last_price] + list(pt)
    y_lo95  = [last_price] + list(lo95)
    y_hi95  = [last_price] + list(hi95)
    y_lo80  = [last_price] + list(lo80)
    y_hi80  = [last_price] + list(hi80)
    for y_h, y_l, fc, name_ in [
        (y_hi95, y_lo95, 'rgba(224,224,224,0.10)', f'{label} 95% CI'),
        (y_hi80, y_lo80, 'rgba(224,224,224,0.20)', f'{label} 80% CI'),
    ]:
        fig.add_trace(go.Scatter(
            x=list(x_line) + list(reversed(x_line)),
            y=list(y_h) + list(reversed(y_l)),
            fill='toself', fillcolor=fc,
            line=dict(color='rgba(0,0,0,0)'),
            name=name_, hoverinfo='skip', showlegend=True
        ), row=row, col=1)
    for y_vals, dash_, color_, name_ in [
        (y_hi95, 'dot',  'rgba(224,224,224,0.35)', f'{label} 95% Upper'),
        (y_lo95, 'dot',  'rgba(224,224,224,0.35)', f'{label} 95% Lower'),
        (y_hi80, 'dash', 'rgba(224,224,224,0.55)', f'{label} 80% Upper'),
        (y_lo80, 'dash', 'rgba(224,224,224,0.55)', f'{label} 80% Lower'),
    ]:
        fig.add_trace(go.Scatter(
            x=x_line, y=y_vals, mode='lines',
            line=dict(color=color_, width=1, dash=dash_),
            name=name_,
            hovertemplate=f"{name_}: GH¢%{{y:.4f}}<extra></extra>",
            showlegend=False
        ), row=row, col=1)
    fig.add_trace(go.Scatter(
        x=x_line, y=y_point, mode='lines+markers',
        line=dict(color=color, width=2, dash='dash'),
        marker=dict(size=6, color=color, symbol='circle-open'),
        name=f'{label} Forecast ({lbl})',
        hovertemplate=(f"<b>{label} Forecast</b><br>"
                       "Date: %{x}<br>Price: GH¢%{y:.4f}<extra></extra>")
    ), row=row, col=1)
    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line=dict(color='rgba(255,255,255,0.2)', dash='dot', width=1),
        row=row, col=1
    )
    return lbl


# ── Layout helper ─────────────────────────────────────────────────────────────
def _apply_layout(fig, title, rsi_row, macd_row, has_vol,
                  yaxis1_title='Price (GH¢)', yaxis2_title='Volume'):
    spike = dict(
        showspikes=True, spikemode='across+marker', spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)', spikethickness=1,
        spikedash='dot', gridcolor='#1e2130'
    )
    fig.update_layout(
        template='plotly_dark',
        title=dict(text=title, font=dict(size=15, color='white')),
        font=dict(family='Inter, Arial', color='#cccccc'),
        paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        hovermode='x unified', spikedistance=-1, hoverdistance=50,
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7,  label='1W', step='day',
                         stepmode='backward'),
                    dict(count=1,  label='1M', step='month',
                         stepmode='backward'),
                    dict(count=3,  label='3M', step='month',
                         stepmode='backward'),
                    dict(count=6,  label='6M', step='month',
                         stepmode='backward'),
                    dict(count=1,  label='YTD', step='year',
                         stepmode='todate'),
                    dict(count=1,  label='1Y', step='year',
                         stepmode='backward'),
                    dict(step='all', label='All')
                ],
                bgcolor='#1e2130', activecolor='#26a69a',
                font=dict(color='white')
            ),
            rangeslider=dict(visible=True, thickness=0.04,
                             bgcolor='#1e2130'),
            type='date', **spike
        ),
        yaxis=dict(title=yaxis1_title, side='right', **spike),
        legend=dict(orientation='h', y=-0.14,
                    bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
        margin=dict(l=10, r=60, t=80, b=50),
        barmode='relative'
    )
    if has_vol:
        fig.update_yaxes(title_text=yaxis2_title,
                         row=2, col=1, **spike)
    fig.update_yaxes(title_text='RSI (14)', row=rsi_row, col=1,
                     range=[0, 100], **spike)
    fig.update_yaxes(title_text='MACD', row=macd_row, col=1, **spike)
    fig.update_xaxes(
        showgrid=True, gridcolor='#1e2130', showspikes=True,
        spikemode='across+marker', spikesnap='cursor',
        spikecolor='rgba(255,255,255,0.5)',
        spikethickness=1, spikedash='dot'
    )


# ── Single stock figure ───────────────────────────────────────────────────────
def build_single_stock_fig(stock, show_forecast=False,
                            date_start=None, date_end=None):
    raw = (stock_data[stock_data['Share Code'] == stock]
           .sort_values('Daily Date')
           .drop_duplicates(subset='Daily Date', keep='last'))
    if date_start:
        raw = raw[raw['Daily Date'] >= pd.to_datetime(date_start)]
    if date_end:
        raw = raw[raw['Daily Date'] <= pd.to_datetime(date_end)]
    if raw.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                          title=f'No data for {stock} in selected range.')
        return fig

    d            = compute_indicators(raw)
    w52_high     = d['52W_High'].iloc[-1]
    w52_low      = d['52W_Low'].iloc[-1]
    gap_mask     = d['Gap'].abs() > 0.01
    latest_close = d['Closing Price - VWAP (GH¢)'].iloc[-1]
    latest_date  = d['Daily Date'].iloc[-1]
    prev_close   = d['Closing Price - VWAP (GH¢)'].iloc[-2] \
                   if len(d) > 1 else latest_close
    price_change = latest_close - prev_close
    price_pct    = (price_change / prev_close * 100) if prev_close != 0 else 0
    badge_color  = '#26a69a' if price_change >= 0 else '#ef5350'
    badge_arrow  = '▲' if price_change >= 0 else '▼'
    badge_text   = (f"<b>{stock}</b>  {badge_arrow} GH¢{latest_close:.4f}  "
                    f"({price_change:+.4f} | {price_pct:+.2f}%)")

    if has_volume:
        n_rows, row_heights = 4, [0.50, 0.15, 0.175, 0.175]
        rsi_row, macd_row   = 3, 4
    else:
        n_rows, row_heights = 3, [0.55, 0.225, 0.225]
        rsi_row, macd_row   = 2, 3

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=row_heights,
        specs=[[{}]] * n_rows
    )
    fig.add_trace(go.Candlestick(
        x=d['Daily Date'],
        open=d['Opening Price (GH¢)'], high=d['Year High (GH¢)'],
        low=d['Year Low (GH¢)'],
        close=d['Closing Price - VWAP (GH¢)'],
        customdata=d['PctChange'].values, name=stock,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
        hovertemplate=(
            f"<b>{stock}</b><br>Date: %{{x}}<br>"
            "Open: GH¢%{open:.4f}<br>High: GH¢%{high:.4f}<br>"
            "Low: GH¢%{low:.4f}<br>Close: GH¢%{close:.4f}<br>"
            "% Chg from Open: %{customdata:.2f}%<extra></extra>"
        )
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['MA20'], mode='lines',
        name='MA 20', line=dict(color='#FFA726', width=1.2),
        hovertemplate="MA20: GH¢%{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['MA50'], mode='lines',
        name='MA 50', line=dict(color='#AB47BC', width=1.2),
        hovertemplate="MA50: GH¢%{y:.4f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['BB_upper'], mode='lines',
        name='BB Upper',
        line=dict(color='rgba(100,181,246,0.6)', width=1, dash='dot'),
        hovertemplate="BB Upper: GH¢%{y:.4f}<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['BB_lower'], mode='lines',
        name='BB Lower', fill='tonexty',
        fillcolor='rgba(100,181,246,0.07)',
        line=dict(color='rgba(100,181,246,0.6)', width=1, dash='dot'),
        hovertemplate="BB Lower: GH¢%{y:.4f}<extra></extra>"),
        row=1, col=1)
    for val, lbl_, clr in [
        (w52_high, '52W High', 'rgba(38,166,154,0.5)'),
        (w52_low,  '52W Low',  'rgba(239,83,80,0.5)')
    ]:
        fig.add_trace(go.Scatter(
            x=[d['Daily Date'].iloc[0], d['Daily Date'].iloc[-1]],
            y=[val, val], mode='lines', name=lbl_,
            line=dict(color=clr, width=1, dash='dash'),
            hovertemplate=f"{lbl_}: GH¢{val:.4f}<extra></extra>"
        ), row=1, col=1)
    gap_days = d[gap_mask]
    fig.add_trace(go.Scatter(
        x=gap_days['Daily Date'], y=gap_days['Opening Price (GH¢)'],
        mode='markers', name='Gap',
        marker=dict(symbol='triangle-up', color='yellow', size=7),
        hovertemplate="Gap: GH¢%{y:.4f}<extra></extra>"
    ), row=1, col=1)
    buys  = d[d['Buy_Signal']]
    sells = d[d['Sell_Signal']]
    fig.add_trace(go.Scatter(
        x=buys['Daily Date'], y=buys['Year Low (GH¢)'] * 0.995,
        mode='markers+text', name='Buy Signal',
        marker=dict(symbol='triangle-up', color='#00E676', size=14,
                    line=dict(color='white', width=1)),
        text=['BUY'] * len(buys), textposition='bottom center',
        textfont=dict(color='#00E676', size=9),
        hovertemplate="BUY — GH¢%{y:.4f}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=sells['Daily Date'], y=sells['Year High (GH¢)'] * 1.005,
        mode='markers+text', name='Sell Signal',
        marker=dict(symbol='triangle-down', color='#FF1744', size=14,
                    line=dict(color='white', width=1)),
        text=['SELL'] * len(sells), textposition='top center',
        textfont=dict(color='#FF1744', size=9),
        hovertemplate="SELL — GH¢%{y:.4f}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[latest_date], y=[latest_close], mode='markers+text',
        name='Latest Price',
        marker=dict(symbol='circle', color=badge_color, size=10,
                    line=dict(color='white', width=1.5)),
        text=[badge_text], textposition='middle left',
        textfont=dict(color=badge_color, size=11),
        hovertemplate=f"Latest: GH¢{latest_close:.4f}<extra></extra>"
    ), row=1, col=1)
    if has_volume:
        vol_colors = [
            '#26a69a' if c >= o else '#ef5350'
            for c, o in zip(d['Closing Price - VWAP (GH¢)'],
                            d['Opening Price (GH¢)'])
        ]
        fig.add_trace(go.Bar(
            x=d['Daily Date'],
            y=d['Total Shares Traded'], name='Volume',
            marker_color=vol_colors, opacity=0.6,
            hovertemplate="Volume: %{y:,.0f}<extra></extra>"
        ), row=2, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['RSI'], mode='lines',
        name='RSI 14', line=dict(color='#80CBC4', width=1.2),
        hovertemplate="RSI: %{y:.1f}<extra></extra>"),
        row=rsi_row, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['MACD'], mode='lines',
        name='MACD', line=dict(color='#42A5F5', width=1.3),
        hovertemplate="MACD: %{y:.4f}<extra></extra>"),
        row=macd_row, col=1)
    fig.add_trace(go.Scatter(x=d['Daily Date'], y=d['Signal'], mode='lines',
        name='Signal', line=dict(color='#FF7043', width=1.3),
        hovertemplate="Signal: %{y:.4f}<extra></extra>"),
        row=macd_row, col=1)
    hist_colors = ['#26a69a' if v >= 0 else '#ef5350'
                   for v in d['MACD_hist'].fillna(0)]
    fig.add_trace(go.Bar(x=d['Daily Date'], y=d['MACD_hist'],
        name='MACD Hist', marker_color=hist_colors, opacity=0.6,
        hovertemplate="Hist: %{y:.4f}<extra></extra>"),
        row=macd_row, col=1)
    for y0, y1, fc, lw, lc in [
        (70, 100, 'rgba(239,83,80,0.08)',   0, None),
        (0,  30,  'rgba(38,166,154,0.08)',  0, None),
    ]:
        fig.add_hrect(y0=y0, y1=y1, row=rsi_row, col=1,
                      fillcolor=fc, line_width=lw)
    fig.add_hline(y=70, row=rsi_row, col=1,
                  line=dict(color='rgba(239,83,80,0.4)',
                            dash='dash', width=1))
    fig.add_hline(y=30, row=rsi_row, col=1,
                  line=dict(color='rgba(38,166,154,0.4)',
                            dash='dash', width=1))
    fig.add_hline(y=0, row=macd_row, col=1,
                  line=dict(color='rgba(255,255,255,0.2)',
                            dash='dot', width=1))
    model_lbl = ''
    if show_forecast:
        price_series = (d[['Daily Date', 'Closing Price - VWAP (GH¢)']]
                        .set_index('Daily Date')
                        ['Closing Price - VWAP (GH¢)'])
        model_lbl = add_forecast_traces(fig, price_series, row=1,
                                        label=stock, color='#E0E0E0')
    fc_note = f'  ·  Forecast: {model_lbl}' if model_lbl else ''
    _apply_layout(fig, f'GSE Candlestick Chart – {stock}{fc_note}',
                  rsi_row, macd_row, has_volume)
    return fig


# ── Portfolio figure ──────────────────────────────────────────────────────────
def build_portfolio_fig(selected_stocks, weights, show_forecast=False,
                        date_start=None, date_end=None):
    COLORS = ['#42A5F5', '#FFA726', '#AB47BC', '#26a69a',
              '#FF7043', '#80CBC4', '#FFD54F', '#EF5350']
    aligned = {}
    for s in selected_stocks:
        d = (stock_data[stock_data['Share Code'] == s]
             .sort_values('Daily Date')
             .drop_duplicates(subset='Daily Date', keep='last'))
        if date_start:
            d = d[d['Daily Date'] >= pd.to_datetime(date_start)]
        if date_end:
            d = d[d['Daily Date'] <= pd.to_datetime(date_end)]
        d = d.set_index('Daily Date')['Closing Price - VWAP (GH¢)']
        aligned[s] = d
    prices = pd.DataFrame(aligned).dropna()
    if prices.empty:
        fig = go.Figure()
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0e1117',
                          title='No overlapping dates found.')
        return fig

    norm       = prices / prices.iloc[0] * 100
    port_idx   = sum(norm[s] * weights.get(s, 1 / len(selected_stocks))
                     for s in selected_stocks)
    port_close = sum(prices[s] * weights.get(s, 1 / len(selected_stocks))
                     for s in selected_stocks)
    c             = port_close
    ma20          = c.rolling(20).mean()
    ma50          = c.rolling(50).mean()
    bb_mid        = c.rolling(20).mean()
    bb_upper      = bb_mid + 2 * c.rolling(20).std()
    bb_lower      = bb_mid - 2 * c.rolling(20).std()
    delta         = c.diff()
    gain          = delta.clip(lower=0).rolling(14).mean()
    loss          = (-delta.clip(upper=0)).rolling(14).mean()
    rs            = gain / loss.replace(0, np.nan)
    rsi           = 100 - (100 / (1 + rs))
    ema12         = c.ewm(span=12, adjust=False).mean()
    ema26         = c.ewm(span=26, adjust=False).mean()
    macd          = ema12 - ema26
    signal_line   = macd.ewm(span=9, adjust=False).mean()
    macd_hist     = macd - signal_line
    port_returns  = c.pct_change()
    cum_return    = (1 + port_returns).cumprod() - 1
    total_ret     = cum_return.iloc[-1] * 100
    volatility    = port_returns.std() * np.sqrt(252) * 100
    sharpe        = (port_returns.mean() / port_returns.std()
                     * np.sqrt(252)) if port_returns.std() != 0 else 0
    max_dd        = ((c / c.cummax()) - 1).min() * 100
    weight_str    = '  |  '.join(
        f"{s}: {weights.get(s, 1/len(selected_stocks))*100:.1f}%"
        for s in selected_stocks
    )
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.20, 0.175, 0.175],
        specs=[[{}]] * 4
    )
    for idx, s in enumerate(selected_stocks):
        color = COLORS[idx % len(COLORS)]
        w_pct = weights.get(s, 1 / len(selected_stocks)) * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[s], mode='lines',
            name=f'{s} ({w_pct:.1f}%)',
            line=dict(color=color, width=1.2, dash='dot'),
            hovertemplate=f"{s}: %{{y:.2f}}<extra></extra>"
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=port_idx.index, y=port_idx, mode='lines',
        name='Portfolio Index', line=dict(color='white', width=2.5),
        hovertemplate="Portfolio: %{y:.2f}<extra></extra>"
    ), row=1, col=1)
    for y_ma, name_, clr_ in [
        (port_idx.rolling(20).mean(), 'MA 20', '#FFA726'),
        (port_idx.rolling(50).mean(), 'MA 50', '#AB47BC'),
    ]:
        fig.add_trace(go.Scatter(x=port_idx.index, y=y_ma, mode='lines',
            name=name_, line=dict(color=clr_, width=1.2),
            hovertemplate=f"{name_}: %{{y:.2f}}<extra></extra>"),
            row=1, col=1)
    bb_mid_n   = port_idx.rolling(20).mean()
    bb_upper_n = bb_mid_n + 2 * port_idx.rolling(20).std()
    bb_lower_n = bb_mid_n - 2 * port_idx.rolling(20).std()
    fig.add_trace(go.Scatter(x=port_idx.index, y=bb_upper_n, mode='lines',
        name='BB Upper',
        line=dict(color='rgba(100,181,246,0.5)', width=1, dash='dot'),
        hovertemplate="BB Upper: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=port_idx.index, y=bb_lower_n, mode='lines',
        name='BB Lower', fill='tonexty',
        fillcolor='rgba(100,181,246,0.06)',
        line=dict(color='rgba(100,181,246,0.5)', width=1, dash='dot'),
        hovertemplate="BB Lower: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=cum_return.index, y=cum_return * 100, mode='lines',
        name='Cum. Return (%)', fill='tozeroy',
        fillcolor='rgba(38,166,154,0.12)',
        line=dict(color='#26a69a', width=1.5),
        hovertemplate="Cum. Return: %{y:.2f}%<extra></extra>"
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, mode='lines',
        name='RSI 14', line=dict(color='#80CBC4', width=1.2),
        hovertemplate="RSI: %{y:.1f}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Scatter(x=macd.index, y=macd, mode='lines',
        name='MACD', line=dict(color='#42A5F5', width=1.3),
        hovertemplate="MACD: %{y:.4f}<extra></extra>"), row=4, col=1)
    fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode='lines',
        name='Signal', line=dict(color='#FF7043', width=1.3),
        hovertemplate="Signal: %{y:.4f}<extra></extra>"), row=4, col=1)
    hist_colors = ['#26a69a' if v >= 0 else '#ef5350'
                   for v in macd_hist.fillna(0)]
    fig.add_trace(go.Bar(x=macd_hist.index, y=macd_hist, name='MACD Hist',
        marker_color=hist_colors, opacity=0.6,
        hovertemplate="Hist: %{y:.4f}<extra></extra>"), row=4, col=1)
    for y0, y1, fc_ in [
        (70, 100, 'rgba(239,83,80,0.08)'),
        (0,  30,  'rgba(38,166,154,0.08)')
    ]:
        fig.add_hrect(y0=y0, y1=y1, row=3, col=1,
                      fillcolor=fc_, line_width=0)
    for y_val, clr_ in [(70, 'rgba(239,83,80,0.4)'),
                        (30, 'rgba(38,166,154,0.4)')]:
        fig.add_hline(y=y_val, row=3, col=1,
                      line=dict(color=clr_, dash='dash', width=1))
    fig.add_hline(y=0, row=4, col=1,
                  line=dict(color='rgba(255,255,255,0.2)',
                            dash='dot', width=1))
    model_lbl = ''
    if show_forecast:
        model_lbl = add_forecast_traces(fig, port_idx, row=1,
                                        label='Portfolio', color='#FFD54F')
    fc_note  = f'  ·  Forecast: {model_lbl}' if model_lbl else ''
    subtitle = (
        f"Weights → {weight_str}<br>"
        f"<sup>Total Return: {total_ret:+.2f}%  |  "
        f"Ann. Vol: {volatility:.2f}%  |  "
        f"Sharpe: {sharpe:.2f}  |  "
        f"Max DD: {max_dd:.2f}%{fc_note}</sup>"
    )
    _apply_layout(
        fig,
        f'GSE Portfolio — {", ".join(selected_stocks)}'
        f'<br><sup>{subtitle}</sup>',
        rsi_row=3, macd_row=4, has_vol=False,
        yaxis1_title='Normalised Index (base=100)',
        yaxis2_title='Cum. Return (%)'
    )
    return fig


# ── Advice card builder ───────────────────────────────────────────────────────
def make_advice_card(r, action):
    is_buy   = action == 'buy'
    css_cls  = 'advice-buy' if is_buy else 'advice-sell'
    icon     = '🟢' if is_buy else '🔴'
    action_  = 'BUY' if is_buy else 'SELL'
    score    = r['buy_score'] if is_buy else r['sell_score']
    bar_cls  = 'score-bar-fill-buy' if is_buy else 'score-bar-fill-sell'
    chg_clr  = '#26a69a' if r['chg'] >= 0 else '#ef5350'
    chg_arr  = '▲' if r['chg'] >= 0 else '▼'

    # Reason sentences
    reasons = []
    s = r['scores']
    if is_buy:
        if s.get('rsi', 0) > 0:
            reasons.append(f"RSI {r['rsi']} — oversold territory")
        if s.get('macd', 0) > 0:
            reasons.append(f"MACD: {r['macd_signal']}")
        if s.get('ma', 0) > 0:
            reasons.append('MA20 above MA50 — uptrend confirmed')
        if s.get('bb', 0) > 0:
            reasons.append('Price near lower Bollinger band — potential bounce')
        if s.get('w52', 0) > 0:
            reasons.append('Near 52W low — potential value entry')
        if s.get('momentum', 0) > 0:
            reasons.append('Positive 5-day price momentum')
    else:
        if s.get('rsi', 0) < 0:
            reasons.append(f"RSI {r['rsi']} — overbought territory")
        if s.get('macd', 0) < 0:
            reasons.append(f"MACD: {r['macd_signal']}")
        if s.get('ma', 0) < 0:
            reasons.append('MA20 below MA50 — downtrend in place')
        if s.get('bb', 0) < 0:
            reasons.append('Price near upper Bollinger band — potential reversal')
        if s.get('w52', 0) < 0:
            reasons.append('Near 52W high — stretched valuation')
        if s.get('momentum', 0) < 0:
            reasons.append('Negative 5-day price momentum')

    return html.Div([
        # Header
        html.Div([
            html.Span(f'{icon} {action_}  ',
                      style={'fontWeight': 'bold', 'fontSize': '13px',
                             'color': '#26a69a' if is_buy else '#ef5350'}),
            html.Span(r['stock'],
                      style={'fontWeight': 'bold', 'fontSize': '15px',
                             'color': 'white'}),
        ]),
        # Price
        html.Div([
            html.Span(f"GH¢{r['price']:.4f}  ",
                      style={'fontSize': '13px', 'color': '#ccc'}),
            html.Span(f"{chg_arr} {r['chg']:+.4f} ({r['chg_pct']:+.2f}%)",
                      style={'fontSize': '11px', 'color': chg_clr}),
        ], style={'marginTop': '3px'}),
        # Score bar
        html.Div([
            html.Div(style={'width': f'{score}%'},
                     className=bar_cls),
        ], className='score-bar-bg', style={'marginTop': '6px'}),
        html.Div(f'Conviction: {score}%',
                 style={'fontSize': '10px', 'color': '#888',
                        'marginTop': '2px'}),
        # Reasons
        html.Ul([
            html.Li(reason,
                    style={'fontSize': '11px', 'color': '#bbb',
                           'marginBottom': '2px'})
            for reason in reasons[:4]
        ], style={'paddingLeft': '14px', 'marginTop': '6px',
                  'marginBottom': '0'}),
    ], className=css_cls)


def build_advice_sidebar():
    top_buys, top_sells = get_recommendations()

    buy_cards = ([make_advice_card(r, 'buy') for r in top_buys]
                 if top_buys
                 else [html.P('No strong buy signals found.',
                              style={'color': '#888', 'fontSize': '12px'})])

    sell_cards = ([make_advice_card(r, 'sell') for r in top_sells]
                  if top_sells
                  else [html.P('No strong sell signals found.',
                               style={'color': '#888', 'fontSize': '12px'})])

    return html.Div([
        html.Div('📊 Market Advice',
                 style={'color': 'white', 'fontWeight': 'bold',
                        'fontSize': '14px', 'marginBottom': '10px',
                        'borderBottom': '1px solid #333',
                        'paddingBottom': '8px'}),
        html.Div('🟢 Top Buys',
                 style={'color': '#26a69a', 'fontSize': '12px',
                        'fontWeight': 'bold', 'marginBottom': '8px'}),
        *buy_cards,
        html.Hr(style={'borderColor': '#333', 'margin': '12px 0'}),
        html.Div('🔴 Top Sells',
                 style={'color': '#ef5350', 'fontSize': '12px',
                        'fontWeight': 'bold', 'marginBottom': '8px'}),
        *sell_cards,
        html.Hr(style={'borderColor': '#333', 'margin': '12px 0'}),
        html.Div('⚠ Disclaimer',
                 style={'color': '#FFA726', 'fontSize': '11px',
                        'fontWeight': 'bold'}),
        html.P(
            'These recommendations are generated purely from technical '
            'indicators and historical price data. They do not constitute '
            'financial advice. Always conduct your own research before '
            'making investment decisions.',
            style={'color': '#666', 'fontSize': '10px',
                   'marginTop': '4px', 'lineHeight': '1.4'}
        ),
        html.Div(f'Last updated: {date.today().strftime("%d %b %Y")}',
                 style={'color': '#555', 'fontSize': '10px',
                        'marginTop': '8px'}),
    ], style={
        'backgroundColor': '#0e1117',
        'border': '1px solid #1e2130',
        'borderRadius': '8px',
        'padding': '14px',
        'height': '100%',
        'overflowY': 'auto'
    })


# ── Indicator groups & tooltips ───────────────────────────────────────────────
INDICATOR_GROUPS = {
    'MAs':       ['MA 20', 'MA 50'],
    'Bollinger': ['BB Upper', 'BB Lower'],
    '52W':       ['52W High', '52W Low'],
    'Gaps':      ['Gap'],
    'Signals':   ['Buy Signal', 'Sell Signal'],
    'Latest':    ['Latest Price'],
    'Volume':    ['Volume'],
    'RSI':       ['RSI 14'],
    'MACD':      ['MACD', 'Signal', 'MACD Hist'],
    'Forecast':  ['Forecast', '95% CI', '80% CI',
                  '95% Upper', '95% Lower', '80% Upper', '80% Lower'],
}

INDICATOR_META = {
    'MAs': {'tooltip': html.Div([
        html.Strong('📈 Moving Averages (MA20 & MA50)'),
        html.Br(), html.Br(),
        html.Span('MA20: Average closing price over the last 20 trading '
                  'days. Tracks short-term momentum.'),
        html.Br(), html.Br(),
        html.Span('MA50: Average over 50 days. Tracks medium-term trend.'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Golden Cross (MA20 > MA50) = potential uptrend. '
                  'Death Cross (MA20 < MA50) = potential downtrend.')
    ], style={'textAlign': 'left'})},
    'Bollinger': {'tooltip': html.Div([
        html.Strong('📊 Bollinger Bands (20-period, 2 std dev)'),
        html.Br(), html.Br(),
        html.Span('Upper band (+2σ), middle MA20, lower band (−2σ).'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Price near upper = overbought. Near lower = oversold. '
                  'Bands squeezing = big move incoming.')
    ], style={'textAlign': 'left'})},
    '52W': {'tooltip': html.Div([
        html.Strong('📅 52-Week High & Low'),
        html.Br(), html.Br(),
        html.Span('Highest and lowest price over the past 252 trading days.'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Breaking above 52W High = strong bullish breakout. '
                  'These levels act as strong support/resistance.')
    ], style={'textAlign': 'left'})},
    'Gaps': {'tooltip': html.Div([
        html.Strong('⚡ Price Gaps'),
        html.Br(), html.Br(),
        html.Span('Days where open ≠ previous close, signalling news '
                  'events or order imbalances.'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Gaps are often filled later — useful reference levels.')
    ], style={'textAlign': 'left'})},
    'Signals': {'tooltip': html.Div([
        html.Strong('🟢🔴 Buy / Sell Signals'),
        html.Br(), html.Br(),
        html.Span('🟢 BUY: MA20 crosses above MA50 AND RSI < 60'),
        html.Br(),
        html.Span('🔴 SELL: MA20 crosses below MA50 AND RSI > 40'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Best in trending markets. May give false signals '
                  'in low-volume GSE conditions.')
    ], style={'textAlign': 'left'})},
    'Latest': {'tooltip': html.Div([
        html.Strong('🏷 Latest Price Badge'),
        html.Br(), html.Br(),
        html.Span('Shows current price, absolute change, and % change '
                  'from the previous close.')
    ], style={'textAlign': 'left'})},
    'Volume': {'tooltip': html.Div([
        html.Strong('📦 Trading Volume'),
        html.Br(), html.Br(),
        html.Span('Green = up day, Red = down day.'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Breakouts on high volume are more reliable. '
                  'Declining volume = weakening trend.')
    ], style={'textAlign': 'left'})},
    'RSI': {'tooltip': html.Div([
        html.Strong('⚡ RSI (14)'),
        html.Br(), html.Br(),
        html.Span('Momentum oscillator 0–100.'),
        html.Br(),
        html.Span('• > 70 = Overbought'), html.Br(),
        html.Span('• < 30 = Oversold'), html.Br(),
        html.Span('• 50 = Neutral'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('Divergence between RSI and price = powerful '
                  'reversal signal.')
    ], style={'textAlign': 'left'})},
    'MACD': {'tooltip': html.Div([
        html.Strong('📉 MACD'),
        html.Br(), html.Br(),
        html.Span('• MACD (blue): EMA12 − EMA26'), html.Br(),
        html.Span('• Signal (orange): 9-EMA of MACD'), html.Br(),
        html.Span('• Histogram: MACD − Signal'),
        html.Br(), html.Br(),
        html.Strong('🔑 Use: '),
        html.Span('MACD crossing above Signal = bullish. '
                  'Histogram shrinking = fading momentum.')
    ], style={'textAlign': 'left'})},
    'Forecast': {'tooltip': html.Div([
        html.Strong('🔮 5-Day Forecast (ARIMA + GARCH)'),
        html.Br(), html.Br(),
        html.Span('• 80% CI: price expected here 4/5 times'), html.Br(),
        html.Span('• 95% CI: price expected here 19/20 times'),
        html.Br(), html.Br(),
        html.Strong('⚠ Note: '),
        html.Span('High uncertainty on GSE due to low liquidity. '
                  'Use as directional guide only.')
    ], style={'textAlign': 'left'})},
}

# ── Dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server   # expose for gunicorn

DARK = {'backgroundColor': '#0e1117', 'color': '#cccccc'}

app.layout = dbc.Container(fluid=True, style=DARK, children=[

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.H4(
        '📈 The GSE Stock Monitor',
        style={'color': 'white', 'padding': '14px 0 4px',
               'fontSize': '20px'}
    ))),

    # ── Mode + forecast ───────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(dbc.RadioItems(
            id='mode-toggle',
            options=[{'label': '  Single Stock', 'value': 'single'},
                     {'label': '  Portfolio',    'value': 'portfolio'}],
            value='single', inline=True,
            inputStyle={'marginRight': '6px'},
            labelStyle={'marginRight': '24px', 'color': '#cccccc'}
        ), width=6),
        dbc.Col(dbc.Checklist(
            id='forecast-toggle',
            options=[{'label': '  Show 5-Day Forecast (ARIMA + GARCH)',
                      'value': 'show'}],
            value=[], inline=True,
            inputStyle={'marginRight': '6px'},
            labelStyle={'color': '#FFD54F', 'fontWeight': 'bold'}
        ), width=6)
    ], style={'marginBottom': '8px'}),

    # ── Indicator toggles ─────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        dbc.Button('👁 Show All', id='btn-show-all', color='success',
                   size='sm', style={'marginRight': '6px',
                                      'fontSize': '11px'}),
        dbc.Button('✕ Hide All', id='btn-hide-all', color='danger',
                   size='sm', style={'marginRight': '12px',
                                      'fontSize': '11px'}),
        *[html.Span([
            dbc.Button(grp,
                id={'type': 'grp-btn', 'index': grp},
                color='info', outline=False, size='sm',
                style={'marginRight': '4px', 'fontSize': '11px'}),
            dbc.Tooltip(
                info['tooltip'],
                target={'type': 'grp-btn', 'index': grp},
                placement='bottom',
                delay={'show': 300, 'hide': 100},
                style={'backgroundColor': '#1e2130', 'color': '#e0e0e0',
                       'border': '1px solid #444', 'borderRadius': '6px',
                       'fontSize': '12px', 'maxWidth': '280px',
                       'padding': '8px 10px', 'lineHeight': '1.5'}
            )
        ]) for grp, info in INDICATOR_META.items()]
    ], style={'display': 'flex', 'flexWrap': 'wrap',
              'alignItems': 'center', 'gap': '2px',
              'padding': '6px 0 8px'}))),

    # ── Date range bar ────────────────────────────────────────────────────────
    dbc.Row(dbc.Col(html.Div([
        *[dbc.Button(lbl,
              id={'type': 'preset-btn', 'index': lbl},
              color='secondary', outline=True, size='sm',
              style={'marginRight': '4px', 'fontSize': '11px'})
          for lbl in PRESETS],
        html.Span('|', style={'color': '#555', 'margin': '0 8px',
                               'fontSize': '18px'}),
        html.Label('From:', style={'color': '#aaa', 'fontSize': '12px',
                                    'marginRight': '4px'}),
        dcc.DatePickerSingle(id='date-start',
                              display_format='DD MMM YYYY',
                              placeholder='Start date',
                              style={'marginRight': '8px'}),
        html.Label('To:', style={'color': '#aaa', 'fontSize': '12px',
                                  'marginRight': '4px'}),
        dcc.DatePickerSingle(id='date-end',
                              display_format='DD MMM YYYY',
                              placeholder='End date',
                              style={'marginRight': '8px'}),
        dbc.Button('Apply', id='btn-apply-dates', color='primary',
                   size='sm', style={'fontSize': '11px',
                                      'marginRight': '6px'}),
        dbc.Button('Reset', id='btn-reset-dates', color='secondary',
                   size='sm', style={'fontSize': '11px'}),
        html.Span(id='date-range-badge',
                  style={'color': '#26a69a', 'fontSize': '11px',
                          'marginLeft': '10px', 'fontStyle': 'italic'})
    ], style={'display': 'flex', 'flexWrap': 'wrap',
              'alignItems': 'center', 'gap': '2px',
              'padding': '2px 0 10px'}))),

    # ── Stock / portfolio controls ────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.Div(id='single-controls', children=[
            html.Label('Select Stock',
                       style={'color': '#aaa', 'fontSize': '12px'}),
            dcc.Dropdown(
                id='single-stock-dd',
                options=[{'label': s, 'value': s} for s in stocks],
                value=stocks[0], clearable=False,
                style={'backgroundColor': '#1e2130', 'color': 'white'}
            )
        ]), width=3),
        dbc.Col(html.Div(id='portfolio-controls',
                         style={'display': 'none'}, children=[
            dbc.Row([
                dbc.Col([
                    html.Label('Select Stocks for Portfolio',
                               style={'color': '#aaa', 'fontSize': '12px'}),
                    dcc.Dropdown(
                        id='portfolio-stock-dd',
                        options=[{'label': s, 'value': s} for s in stocks],
                        value=([stocks[0], stocks[1]]
                               if len(stocks) > 1 else stocks),
                        multi=True,
                        placeholder='Select 2 or more stocks...',
                        style={'backgroundColor': '#1e2130',
                               'color': 'white'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label('Custom Weights (must sum to 100%)',
                               style={'color': '#aaa', 'fontSize': '12px'}),
                    html.Div(id='weight-inputs')
                ], width=6)
            ]),
            dbc.Row(dbc.Col(
                dbc.Button('Build Portfolio', id='build-btn',
                           color='success', size='sm',
                           style={'marginTop': '10px'}),
                width=12
            )),
            html.Div(id='weight-error',
                     style={'color': '#ef5350', 'fontSize': '12px',
                            'marginTop': '4px'})
        ]), width=9)
    ], style={'marginBottom': '12px'}),

    # ── Main content: chart + advice sidebar ──────────────────────────────────
    dbc.Row([
        # Chart (9 cols)
        dbc.Col(dcc.Graph(
            id='main-chart',
            figure=build_single_stock_fig(stocks[0]),
            style={'height': '82vh'},
            config={'scrollZoom': True, 'displayModeBar': True}
        ), width=9),

        # Advice sidebar (3 cols)
        dbc.Col(
            html.Div(
                id='advice-sidebar',
                children=build_advice_sidebar(),
                style={'height': '82vh', 'overflowY': 'auto'}
            ),
            width=3
        )
    ]),

    # ── Stores ────────────────────────────────────────────────────────────────
    dcc.Store(id='grp-visibility',
              data={grp: True for grp in INDICATOR_GROUPS}),
    dcc.Store(id='active-date-range',
              data={'start': None, 'end': None}),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('single-controls',    'style'),
    Output('portfolio-controls', 'style'),
    Input('mode-toggle', 'value')
)
def toggle_controls(mode):
    if mode == 'single':
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('weight-inputs', 'children'),
    Input('portfolio-stock-dd', 'value')
)
def make_weight_inputs(selected):
    if not selected:
        return []
    default = round(100 / len(selected), 1)
    return [
        dbc.Row([
            dbc.Col(html.Label(s, style={'color': '#ccc',
                                          'fontSize': '12px',
                                          'marginTop': '6px'}), width=4),
            dbc.Col(dbc.Input(
                id={'type': 'weight-input', 'index': s},
                type='number', min=0, max=100, step=0.1, value=default,
                style={'backgroundColor': '#0e1117', 'color': 'white',
                       'border': '1px solid #444', 'height': '28px',
                       'fontSize': '12px'}
            ), width=5),
            dbc.Col(html.Span('%', style={'color': '#aaa',
                                           'fontSize': '12px',
                                           'marginTop': '6px'}), width=1)
        ], style={'marginBottom': '4px'})
        for s in selected
    ]


@app.callback(
    Output('date-start', 'date'),
    Output('date-end',   'date'),
    Input({'type': 'preset-btn', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def apply_preset(_):
    triggered = ctx.triggered_id
    if not triggered:
        return None, None
    label = triggered['index']
    days  = PRESETS.get(label)
    if days is None:
        return None, None
    end   = date.today()
    start = end - timedelta(days=days)
    return str(start), str(end)


@app.callback(
    Output('active-date-range', 'data'),
    Output('date-range-badge',  'children'),
    *[Output({'type': 'preset-btn', 'index': lbl}, 'color')
      for lbl in PRESETS],
    *[Output({'type': 'preset-btn', 'index': lbl}, 'outline')
      for lbl in PRESETS],
    Input('btn-apply-dates',  'n_clicks'),
    Input('btn-reset-dates',  'n_clicks'),
    Input({'type': 'preset-btn', 'index': ALL}, 'n_clicks'),
    State('date-start', 'date'),
    State('date-end',   'date'),
    prevent_initial_call=True
)
def commit_date_range(apply_clicks, reset_clicks, preset_clicks,
                      start_date, end_date):
    triggered     = ctx.triggered_id
    active_preset = None

    if triggered == 'btn-reset-dates' or (
            isinstance(triggered, dict) and
            triggered.get('index') == 'All'):
        data, badge   = {'start': None, 'end': None}, '📅 Showing all data'
        active_preset = 'All'
    elif isinstance(triggered, dict) and \
            triggered.get('type') == 'preset-btn':
        label  = triggered['index']
        days   = PRESETS.get(label)
        end    = date.today()
        start  = end - timedelta(days=days) if days else None
        data   = {'start': str(start) if start else None, 'end': str(end)}
        badge  = (f'📅 {label}: '
                  f'{start.strftime("%d %b %Y") if start else "All"}'
                  f' → {end.strftime("%d %b %Y")}')
        active_preset = label
    elif triggered == 'btn-apply-dates':
        if not start_date and not end_date:
            data, badge = {'start': None, 'end': None}, '📅 Showing all data'
        else:
            data  = {'start': start_date, 'end': end_date}
            s_fmt = (pd.to_datetime(start_date).strftime('%d %b %Y')
                     if start_date else 'Start')
            e_fmt = (pd.to_datetime(end_date).strftime('%d %b %Y')
                     if end_date else 'Today')
            badge = f'📅 Custom: {s_fmt} → {e_fmt}'
    else:
        data, badge = {'start': None, 'end': None}, ''

    colors   = ['primary' if lbl == active_preset else 'secondary'
                for lbl in PRESETS]
    outlines = [False     if lbl == active_preset else True
                for lbl in PRESETS]
    return data, badge, *colors, *outlines


@app.callback(
    Output('grp-visibility', 'data'),
    *[Output({'type': 'grp-btn', 'index': grp}, 'color')
      for grp in INDICATOR_META],
    *[Output({'type': 'grp-btn', 'index': grp}, 'outline')
      for grp in INDICATOR_META],
    Input('btn-show-all',  'n_clicks'),
    Input('btn-hide-all',  'n_clicks'),
    Input({'type': 'grp-btn', 'index': ALL}, 'n_clicks'),
    State('grp-visibility', 'data'),
    prevent_initial_call=True
)
def update_visibility_store(show_all, hide_all, grp_clicks, current):
    triggered = ctx.triggered_id
    if triggered == 'btn-show-all':
        new = {grp: True for grp in INDICATOR_GROUPS}
    elif triggered == 'btn-hide-all':
        new = {grp: False for grp in INDICATOR_GROUPS}
    elif isinstance(triggered, dict) and \
            triggered.get('type') == 'grp-btn':
        grp = triggered['index']
        new = dict(current)
        new[grp] = not current.get(grp, True)
    else:
        new = current
    colors   = ['info'      if new.get(grp, True) else 'secondary'
                for grp in INDICATOR_META]
    outlines = [False       if new.get(grp, True) else True
                for grp in INDICATOR_META]
    return new, *colors, *outlines


@app.callback(
    Output('main-chart',   'figure'),
    Output('weight-error', 'children'),
    Input('single-stock-dd',   'value'),
    Input('forecast-toggle',   'value'),
    Input('build-btn',         'n_clicks'),
    Input('grp-visibility',    'data'),
    Input('active-date-range', 'data'),
    State('mode-toggle',         'value'),
    State('portfolio-stock-dd',  'value'),
    State('weight-inputs',       'children'),
    prevent_initial_call=False
)
def update_chart(single_stock, fc_toggle, n_clicks, grp_vis,
                 date_range, mode, port_stocks, weight_children):
    show_forecast = 'show' in (fc_toggle or [])
    d_start = date_range.get('start') if date_range else None
    d_end   = date_range.get('end')   if date_range else None

    if mode == 'single' or ctx.triggered_id in (
            'single-stock-dd', 'forecast-toggle',
            'grp-visibility',  'active-date-range'):
        fig, err = build_single_stock_fig(
            single_stock, show_forecast,
            date_start=d_start, date_end=d_end), ''
    elif not port_stocks or len(port_stocks) < 2:
        fig, err = build_single_stock_fig(
            single_stock, show_forecast,
            date_start=d_start, date_end=d_end), \
            '⚠ Select at least 2 stocks for a portfolio.'
    else:
        weights_raw = {}
        if weight_children:
            for row in weight_children:
                try:
                    stock_label = (row['props']['children'][0]
                                   ['props']['children']
                                   ['props']['children'])
                    val = (row['props']['children'][1]
                           ['props']['children']['props']
                           .get('value', 0))
                    weights_raw[stock_label] = (float(val)
                                                if val is not None
                                                else 0.0)
                except Exception:
                    pass
        total = sum(weights_raw.values())
        if abs(total - 100) > 0.5:
            return (build_single_stock_fig(single_stock, show_forecast,
                                           date_start=d_start,
                                           date_end=d_end),
                    f'⚠ Weights sum to {total:.1f}% — must equal 100%.')
        weights = {s: weights_raw.get(s, 0) / 100 for s in port_stocks}
        fig, err = build_portfolio_fig(
            port_stocks, weights, show_forecast,
            date_start=d_start, date_end=d_end), ''

    if grp_vis:
        for trace in fig.data:
            name = trace.name or ''
            for grp, fragments in INDICATOR_GROUPS.items():
                if any(frag in name for frag in fragments):
                    trace.visible = (True if grp_vis.get(grp, True)
                                     else 'legendonly')
                    break
    return fig, err


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)
