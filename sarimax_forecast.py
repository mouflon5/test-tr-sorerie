"""
Module SARIMAX — Prévision de trésorerie avec variables exogènes
Intégration : ajouter comme onglet dans app.py existant
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings("ignore")


def generate_historical_data(n_weeks: int = 104, seed: int = 42):
    """
    Génère 2 ans d'historique hebdomadaire simulé pour une PME manufacturière.
    Retourne un DataFrame avec cashflows + variables exogènes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n_weeks, freq="W-MON")

    # --- Variables exogènes ---
    # DSO (Days Sales Outstanding) — entre 30 et 65 jours, tendance légère + bruit
    dso_base = 45 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_weeks))  # saisonnalité
    dso = dso_base + rng.normal(0, 3, n_weeks)
    dso = np.clip(dso, 28, 70)

    # Carnet de commandes (backlog) — en k$, corrélé aux encaissements futurs
    backlog_base = 300 + 50 * np.sin(np.linspace(0, 3 * np.pi, n_weeks))
    backlog_trend = np.linspace(0, 40, n_weeks)  # croissance légère
    backlog = backlog_base + backlog_trend + rng.normal(0, 25, n_weeks)
    backlog = np.clip(backlog, 150, 500)

    # PMI manufacturier Québec — proxy d'activité économique (50 = neutre)
    pmi = 52 + 3 * np.sin(np.linspace(0, 2 * np.pi, n_weeks)) + rng.normal(0, 1.5, n_weeks)
    pmi = np.clip(pmi, 45, 60)

    # Semaine avec paie (aux 2 semaines)
    sem_paie = np.array([1 if i % 2 == 0 else 0 for i in range(n_weeks)])

    # --- Cashflows (flux net hebdomadaire) ---
    # Modèle génératif: flux = f(backlog, DSO, PMI, saisonnalité, paie) + bruit
    saisonnalite = 8 * np.sin(np.linspace(0, 8 * np.pi, n_weeks))  # ~mensuelle
    effet_backlog = 0.08 * (backlog - 300)  # Plus de backlog = plus d'encaissements
    effet_dso = -0.6 * (dso - 45)  # DSO élevé = encaissements retardés
    effet_pmi = 1.2 * (pmi - 52)  # PMI > 52 = bon signe
    effet_paie = -18 * sem_paie  # Semaines de paie = décaissement important
    tendance = np.linspace(0, -15, n_weeks)  # légère dégradation

    flux_net = (
        -5  # baseline légèrement négatif
        + saisonnalite
        + effet_backlog
        + effet_dso
        + effet_pmi
        + effet_paie
        + tendance
        + rng.normal(0, 8, n_weeks)  # bruit
    )

    # Solde cumulé
    solde_initial = 180.0
    solde = np.zeros(n_weeks)
    solde[0] = solde_initial + flux_net[0]
    for i in range(1, n_weeks):
        solde[i] = solde[i - 1] + flux_net[i]

    df = pd.DataFrame({
        "Date": dates,
        "Semaine": [f"S{i+1}" for i in range(n_weeks)],
        "Flux_Net_k$": np.round(flux_net, 1),
        "Solde_k$": np.round(solde, 1),
        "DSO_jours": np.round(dso, 1),
        "Backlog_k$": np.round(backlog, 1),
        "PMI_Quebec": np.round(pmi, 1),
        "Sem_Paie": sem_paie.astype(int),
    })

    return df


def render_sarimax_tab(df_dashboard: pd.DataFrame, solde_initial: float, seuil_critique: float):
    """
    Render the SARIMAX forecasting tab.
    """

    st.markdown("#### 📐 Prévision SARIMAX — Modèle économétrique avec variables exogènes")
    st.markdown(
        "Ce modèle utilise l'historique des flux nets et des variables exogènes "
        "(DSO, carnet de commandes, PMI, cycle de paie) pour projeter la trésorerie "
        "sur 13 semaines avec intervalles de confiance."
    )

    # ------------------------------------------
    # Parameters
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### ⚙️ Paramètres du modèle")

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        n_hist = st.select_slider(
            "Historique (semaines)",
            options=[52, 78, 104, 130, 156],
            value=104,
            help="Nombre de semaines d'historique simulé pour entraîner le modèle"
        )
    with p2:
        horizon = st.select_slider(
            "Horizon de prévision",
            options=[4, 8, 13, 18, 26],
            value=13
        )
    with p3:
        ci_level = st.select_slider(
            "Intervalle de confiance (%)",
            options=[80, 90, 95, 99],
            value=90
        )
    with p4:
        n_mc = st.select_slider(
            "Simulations Monte Carlo",
            options=[500, 1000, 2000, 5000],
            value=1000,
            help="Simulations sur les résidus pour affiner les intervalles"
        )

    st.markdown("##### 📊 Hypothèses exogènes pour l'horizon de prévision")
    st.caption("Valeurs projetées des variables exogènes sur l'horizon. Ajustez selon vos attentes.")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        dso_forecast = st.slider("DSO projeté (jours)", 30, 65, 47, 1,
                                 help="Délai moyen de paiement client anticipé")
    with e2:
        backlog_forecast = st.slider("Backlog projeté (k$)", 150, 500, 320, 10,
                                     help="Carnet de commandes anticipé")
    with e3:
        pmi_forecast = st.slider("PMI projeté", 45.0, 58.0, 51.5, 0.5,
                                 help="Indice PMI manufacturier anticipé")
    with e4:
        trend_dso = st.slider("Tendance DSO (/sem.)", -1.0, 1.0, 0.2, 0.1,
                              help="Drift hebdomadaire du DSO (+ = détérioration)")

    st.markdown("---")

    # ------------------------------------------
    # Generate historical data
    # ------------------------------------------
    with st.spinner("Génération de l'historique simulé..."):
        df_hist = generate_historical_data(n_weeks=n_hist)

    # ------------------------------------------
    # Fit SARIMAX
    # ------------------------------------------
    with st.spinner("Ajustement du modèle SARIMAX..."):
        y = df_hist["Flux_Net_k$"].values
        exog = df_hist[["DSO_jours", "Backlog_k$", "PMI_Quebec", "Sem_Paie"]].values

        # SARIMAX(1,1,1)(1,0,1,4) — ordre choisi pour données hebdo avec cycle ~mensuel
        # p=1, d=1, q=1 pour la partie ARIMA
        # P=1, D=0, Q=1, s=4 pour la saisonnalité ~mensuelle (4 semaines)
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 4),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False, maxiter=300)
            model_success = True
        except Exception as e:
            st.error(f"Erreur d'ajustement : {e}")
            model_success = False

    if not model_success:
        return

    # ------------------------------------------
    # Diagnostics
    # ------------------------------------------
    residuals = results.resid[2:]  # skip initial NaN
    resid_std = np.std(residuals)
    aic = results.aic
    bic = results.bic

    # Ljung-Box test
    try:
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].values[0]
    except:
        lb_pvalue = np.nan

    # ADF on residuals
    try:
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(residuals.dropna() if hasattr(residuals, 'dropna') else residuals[~np.isnan(residuals)])
    except:
        adf_pvalue = np.nan

    # ------------------------------------------
    # Build exogenous forecast matrix
    # ------------------------------------------
    exog_forecast = np.zeros((horizon, 4))
    for h in range(horizon):
        exog_forecast[h, 0] = dso_forecast + trend_dso * h  # DSO with trend
        exog_forecast[h, 1] = backlog_forecast  # Stable backlog
        exog_forecast[h, 2] = pmi_forecast  # Stable PMI
        exog_forecast[h, 3] = 1 if (n_hist + h) % 2 == 0 else 0  # Cycle paie

    # ------------------------------------------
    # Forecast
    # ------------------------------------------
    alpha = 1 - ci_level / 100
    forecast_obj = results.get_forecast(steps=horizon, exog=exog_forecast, alpha=alpha)
    forecast_mean_raw = forecast_obj.predicted_mean
    forecast_ci_raw = forecast_obj.conf_int(alpha=alpha)

    # Robust extraction — works whether statsmodels returns Series/DataFrame or ndarray
    forecast_mean_vals = np.array(forecast_mean_raw).flatten()

    if hasattr(forecast_ci_raw, 'values'):
        # DataFrame — extract columns by position
        ci_array = forecast_ci_raw.values
    else:
        # ndarray
        ci_array = np.array(forecast_ci_raw)
    ci_low_vals = ci_array[:, 0]
    ci_high_vals = ci_array[:, 1]

    # Cumulative solde from forecast
    last_solde = float(df_hist["Solde_k$"].iloc[-1])
    solde_forecast_mean = last_solde + np.cumsum(forecast_mean_vals)
    solde_forecast_low = last_solde + np.cumsum(ci_low_vals)
    solde_forecast_high = last_solde + np.cumsum(ci_high_vals)

    forecast_dates = pd.date_range(df_hist["Date"].iloc[-1] + pd.Timedelta(weeks=1),
                                   periods=horizon, freq="W-MON")
    forecast_labels = [f"S+{i+1}" for i in range(horizon)]

    # ------------------------------------------
    # Monte Carlo on residuals
    # ------------------------------------------
    rng = np.random.default_rng(42)
    resid_clean = residuals[~np.isnan(residuals)] if hasattr(residuals, '__iter__') else residuals
    mc_flux = np.zeros((n_mc, horizon))
    mc_solde = np.zeros((n_mc, horizon))

    for sim in range(n_mc):
        noise = rng.normal(0, resid_std, horizon)
        mc_flux[sim] = forecast_mean_vals + noise
        mc_solde[sim] = last_solde + np.cumsum(mc_flux[sim])

    mc_median = np.median(mc_solde, axis=0)
    mc_p5 = np.percentile(mc_solde, 5, axis=0)
    mc_p95 = np.percentile(mc_solde, 95, axis=0)
    mc_p_low = np.percentile(mc_solde, (100 - ci_level) / 2, axis=0)
    mc_p_high = np.percentile(mc_solde, 100 - (100 - ci_level) / 2, axis=0)
    prob_sous_seuil = ((mc_solde < seuil_critique).mean(axis=0) * 100)
    prob_negatif = ((mc_solde < 0).mean(axis=0) * 100)

    # ------------------------------------------
    # KPIs
    # ------------------------------------------
    st.markdown("##### 📈 Résultats du modèle")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("AIC", f"{aic:.0f}", help="Critère d'information d'Akaike — plus bas = meilleur")
    k2.metric("BIC", f"{bic:.0f}", help="Critère bayésien — plus bas = meilleur")
    k3.metric("Écart-type résidus", f"{resid_std:.1f} k$")
    k4.metric("Ljung-Box p-value", f"{lb_pvalue:.3f}" if not np.isnan(lb_pvalue) else "N/A",
              delta="Résidus OK" if lb_pvalue > 0.05 else "Autocorrélation détectée",
              delta_color="normal" if lb_pvalue > 0.05 else "inverse")
    k5.metric("ADF résidus p-value", f"{adf_pvalue:.3f}" if not np.isnan(adf_pvalue) else "N/A",
              delta="Stationnaire" if adf_pvalue < 0.05 else "Non-stationnaire",
              delta_color="normal" if adf_pvalue < 0.05 else "inverse")

    # ------------------------------------------
    # Model coefficients
    # ------------------------------------------
    with st.expander("📋 Coefficients du modèle et interprétation"):
        # Robust extraction — params can be Series (with .index) or ndarray
        param_values = np.array(results.params).flatten()
        pvalue_values = np.array(results.pvalues).flatten()

        if hasattr(results.params, 'index'):
            param_names = list(results.params.index)
        else:
            # Fallback: generate generic names
            exog_names = ["DSO_jours", "Backlog_k$", "PMI_Quebec", "Sem_Paie"]
            param_names = []
            # SARIMAX(1,1,1)(1,0,1,4) typical param order:
            # intercept (if any), ar.L1, ma.L1, ar.S.L4, ma.S.L4, x1..x4, sigma2
            n_params = len(param_values)
            base_names = ["intercept", "ar.L1", "ma.L1", "ar.S.L4", "ma.S.L4"]
            for i in range(n_params):
                if i < len(base_names):
                    param_names.append(base_names[i])
                elif i < len(base_names) + len(exog_names):
                    param_names.append(exog_names[i - len(base_names)])
                elif i == n_params - 1:
                    param_names.append("sigma2")
                else:
                    param_names.append(f"param_{i}")

        coef_df = pd.DataFrame({
            "Paramètre": param_names[:len(param_values)],
            "Coefficient": np.round(param_values, 4),
            "P-value": np.round(pvalue_values, 4),
            "Significatif (5%)": ["✅" if p < 0.05 else "❌" for p in pvalue_values],
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown("""
        **Interprétation des exogènes :**
        - **DSO_jours** : coefficient négatif = un DSO plus élevé réduit les flux nets (retards de paiement)
        - **Backlog_k$** : coefficient positif = plus de commandes = plus d'encaissements futurs
        - **PMI_Quebec** : coefficient positif = conjoncture favorable soutient les flux
        - **Sem_Paie** : coefficient négatif = les semaines de paie pèsent sur la trésorerie
        """)

    # ------------------------------------------
    # Chart 1: Historical fit + forecast
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 🔮 Prévision du solde de trésorerie")

    fig_fc = go.Figure()

    # Historical solde (last 26 weeks for readability)
    show_last = min(26, n_hist)
    hist_tail = df_hist.iloc[-show_last:]

    fig_fc.add_trace(go.Scatter(
        x=hist_tail["Date"], y=hist_tail["Solde_k$"],
        mode="lines+markers", name="Historique",
        line=dict(color="#2E75B6", width=2),
        marker=dict(size=4),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Solde: %{y:.0f} k$<extra></extra>"
    ))

    # MC confidence bands
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=list(mc_p95) + list(mc_p5)[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.1)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P5-P95 (MC)", showlegend=True, hoverinfo="skip"
    ))

    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=list(mc_p_high) + list(mc_p_low)[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.2)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"IC {ci_level}% (MC)", showlegend=True, hoverinfo="skip"
    ))

    # SARIMAX CI band
    fig_fc.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates)[::-1],
        y=list(solde_forecast_high) + list(solde_forecast_low)[::-1],
        fill="toself", fillcolor="rgba(118,147,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"IC {ci_level}% (SARIMAX)", showlegend=True, hoverinfo="skip"
    ))

    # Forecast mean
    fig_fc.add_trace(go.Scatter(
        x=forecast_dates, y=solde_forecast_mean,
        mode="lines+markers", name="Prévision SARIMAX",
        line=dict(color="#76933C", width=3),
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Prévision: %{y:.0f} k$<extra></extra>"
    ))

    # MC median
    fig_fc.add_trace(go.Scatter(
        x=forecast_dates, y=mc_median,
        mode="lines", name="Médiane MC",
        line=dict(color="#E46C0A", width=2, dash="dot"),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Médiane MC: %{y:.0f} k$<extra></extra>"
    ))

    fig_fc.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Seuil critique ({seuil_critique} k$)",
                     annotation_position="top right",
                     annotation_font=dict(color="red", size=10))
    fig_fc.add_hline(y=0, line_color="#999", line_width=1)

    # Vertical line at forecast start — use shape instead of add_vline for date compatibility
    fc_start = forecast_dates[0].isoformat()
    fig_fc.add_shape(
        type="line", x0=fc_start, x1=fc_start, y0=0, y1=1,
        yref="paper", line=dict(color="#999", width=1, dash="dot"),
    )
    fig_fc.add_annotation(
        x=fc_start, y=1, yref="paper",
        text="Début prévision", showarrow=False,
        font=dict(size=9, color="#666"), yshift=10,
    )

    fig_fc.update_layout(
        height=520,
        yaxis_title="Solde de trésorerie (k$)",
        plot_bgcolor="#fafbfc",
        yaxis=dict(gridcolor="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(t=40, b=40, l=60, r=40),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ------------------------------------------
    # Chart 2: Risk heatmap + exogenous variables
    # ------------------------------------------
    col_risk, col_exog = st.columns(2)

    with col_risk:
        st.markdown("##### ⚠️ Probabilité hebdo sous le seuil")
        fig_risk = go.Figure()
        colors = ["#C0504D" if p > 25 else "#E46C0A" if p > 10 else "#76933C" for p in prob_sous_seuil]
        fig_risk.add_trace(go.Bar(
            x=forecast_labels, y=prob_sous_seuil,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in prob_sous_seuil],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Prob: %{y:.1f}%<extra></extra>"
        ))
        fig_risk.add_hline(y=25, line_dash="dot", line_color="#C0504D",
                           annotation_text="Risque élevé", annotation_font=dict(size=8, color="#C0504D"))
        fig_risk.add_hline(y=10, line_dash="dot", line_color="#E46C0A",
                           annotation_text="Risque modéré", annotation_font=dict(size=8, color="#E46C0A"))
        fig_risk.update_layout(
            height=380, yaxis_title="Probabilité (%)",
            yaxis=dict(range=[0, max(prob_sous_seuil.max() * 1.3, 30)]),
            plot_bgcolor="#fafbfc", margin=dict(t=20, b=40, l=60, r=20),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_exog:
        st.markdown("##### 📊 Variables exogènes projetées")
        fig_exog = go.Figure()
        fig_exog.add_trace(go.Scatter(
            x=forecast_labels, y=exog_forecast[:, 0],
            mode="lines+markers", name="DSO (jours)",
            line=dict(color="#2E75B6"), yaxis="y1"
        ))
        fig_exog.add_trace(go.Scatter(
            x=forecast_labels, y=exog_forecast[:, 1],
            mode="lines+markers", name="Backlog (k$)",
            line=dict(color="#76933C"), yaxis="y2"
        ))
        fig_exog.add_trace(go.Bar(
            x=forecast_labels, y=exog_forecast[:, 3] * 5,
            name="Semaine de paie",
            marker_color="rgba(228,108,10,0.3)",
            yaxis="y1"
        ))
        fig_exog.update_layout(
            height=380, plot_bgcolor="#fafbfc",
            yaxis=dict(title="DSO (jours)", side="left", gridcolor="#e8e8e8"),
            yaxis2=dict(title="Backlog (k$)", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
            margin=dict(t=20, b=40, l=60, r=60),
        )
        st.plotly_chart(fig_exog, use_container_width=True)

    # ------------------------------------------
    # Decomposition & Diagnostics section
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 🔬 Analyse de la série temporelle — Décomposition, cycles et résidus")

    # === Seasonal decomposition ===
    st.markdown("###### 📉 Décomposition saisonnière (STL)")
    st.caption("Décomposition additive de la série des flux nets en tendance, saisonnalité et résidus.")

    try:
        series_for_decomp = pd.Series(
            df_hist["Flux_Net_k$"].values,
            index=pd.date_range(df_hist["Date"].iloc[0], periods=len(df_hist), freq="W-MON")
        )
        decomp = seasonal_decompose(series_for_decomp, model="additive", period=4)

        from plotly.subplots import make_subplots
        fig_decomp = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=("Série observée (flux nets)", "Tendance", "Saisonnalité (~4 sem.)", "Résidus / Bruit")
        )

        decomp_dates = series_for_decomp.index

        fig_decomp.add_trace(go.Scatter(
            x=decomp_dates, y=decomp.observed,
            mode="lines", line=dict(color="#2E75B6", width=1.2), name="Observé",
            hovertemplate="%{x|%Y-%m-%d}<br>Flux: %{y:.1f} k$<extra></extra>"
        ), row=1, col=1)

        fig_decomp.add_trace(go.Scatter(
            x=decomp_dates, y=decomp.trend,
            mode="lines", line=dict(color="#76933C", width=2.5), name="Tendance",
            hovertemplate="%{x|%Y-%m-%d}<br>Tendance: %{y:.1f} k$<extra></extra>"
        ), row=2, col=1)
        # Add zero line on trend
        fig_decomp.add_hline(y=0, row=2, col=1, line_color="#ccc", line_width=0.5)

        fig_decomp.add_trace(go.Scatter(
            x=decomp_dates, y=decomp.seasonal,
            mode="lines", line=dict(color="#E46C0A", width=1.5), name="Saisonnalité",
            hovertemplate="%{x|%Y-%m-%d}<br>Saisonnier: %{y:.1f} k$<extra></extra>"
        ), row=3, col=1)
        fig_decomp.add_hline(y=0, row=3, col=1, line_color="#ccc", line_width=0.5)

        # Residuals with color coding
        resid_decomp = decomp.resid.dropna()
        colors_resid = ["#C0504D" if abs(v) > 2 * resid_decomp.std() else "#8DB4E2" for v in resid_decomp]
        fig_decomp.add_trace(go.Bar(
            x=resid_decomp.index, y=resid_decomp.values,
            marker_color=colors_resid, name="Résidus",
            hovertemplate="%{x|%Y-%m-%d}<br>Résidu: %{y:.1f} k$<extra></extra>"
        ), row=4, col=1)
        fig_decomp.add_hline(y=0, row=4, col=1, line_color="#999", line_width=0.5)
        fig_decomp.add_hline(y=2*resid_decomp.std(), row=4, col=1,
                             line_dash="dot", line_color="#E46C0A", line_width=1)
        fig_decomp.add_hline(y=-2*resid_decomp.std(), row=4, col=1,
                             line_dash="dot", line_color="#E46C0A", line_width=1)

        fig_decomp.update_layout(
            height=700, showlegend=False, plot_bgcolor="#fafbfc",
            margin=dict(t=40, b=30, l=60, r=40),
        )
        for i in range(1, 5):
            fig_decomp.update_yaxes(gridcolor="#e8e8e8", row=i, col=1)

        st.plotly_chart(fig_decomp, use_container_width=True)

        # Decomposition stats
        dc1, dc2, dc3, dc4 = st.columns(4)
        trend_vals = decomp.trend.dropna()
        seasonal_amplitude = decomp.seasonal.max() - decomp.seasonal.min()
        dc1.metric("Tendance début → fin",
                   f"{trend_vals.iloc[-1] - trend_vals.iloc[0]:.1f} k$",
                   delta="Dégradation" if trend_vals.iloc[-1] < trend_vals.iloc[0] else "Amélioration",
                   delta_color="inverse" if trend_vals.iloc[-1] < trend_vals.iloc[0] else "normal")
        dc2.metric("Amplitude saisonnière", f"{seasonal_amplitude:.1f} k$",
                   help="Écart entre le pic et le creux du cycle saisonnier")
        dc3.metric("Écart-type résidus (décomp.)", f"{resid_decomp.std():.1f} k$")
        dc4.metric("% variance expliquée par tendance+saison.",
                   f"{(1 - resid_decomp.var() / series_for_decomp.var()) * 100:.1f}%",
                   help="Part de la variance totale expliquée par la tendance et la saisonnalité")

    except Exception as e:
        st.warning(f"Décomposition saisonnière non disponible : {e}")

    # === ACF / PACF ===
    st.markdown("---")
    st.markdown("###### 📊 Autocorrélations (ACF / PACF)")
    st.caption(
        "L'ACF montre les corrélations entre la série et ses retards. "
        "La PACF isole la corrélation directe à chaque retard. "
        "Les pics significatifs indiquent les ordres AR et MA du modèle."
    )

    try:
        flux_clean = df_hist["Flux_Net_k$"].dropna().values
        max_lags = min(30, len(flux_clean) // 3)
        acf_vals = acf(flux_clean, nlags=max_lags, fft=True)
        pacf_vals = pacf(flux_clean, nlags=max_lags)
        ci_bound = 1.96 / np.sqrt(len(flux_clean))

        col_acf, col_pacf = st.columns(2)

        with col_acf:
            fig_acf = go.Figure()
            lags = list(range(len(acf_vals)))
            colors_acf = ["#C0504D" if abs(v) > ci_bound and i > 0 else "#2E75B6" for i, v in enumerate(acf_vals)]
            fig_acf.add_trace(go.Bar(
                x=lags, y=acf_vals, marker_color=colors_acf, name="ACF",
                hovertemplate="Retard %{x}<br>ACF: %{y:.3f}<extra></extra>"
            ))
            fig_acf.add_hline(y=ci_bound, line_dash="dash", line_color="#E46C0A", line_width=1)
            fig_acf.add_hline(y=-ci_bound, line_dash="dash", line_color="#E46C0A", line_width=1)
            fig_acf.add_hline(y=0, line_color="#999", line_width=0.5)

            # Annotate seasonal lags
            for lag in [4, 8, 12, 16, 20, 24]:
                if lag < len(acf_vals) and abs(acf_vals[lag]) > ci_bound:
                    fig_acf.add_annotation(x=lag, y=acf_vals[lag], text=f"S={lag}",
                                          showarrow=True, arrowhead=2, arrowsize=0.8,
                                          font=dict(size=8, color="#E46C0A"))

            fig_acf.update_layout(
                title="ACF — Autocorrélation", height=320,
                xaxis_title="Retard (semaines)",
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=40, l=50, r=20),
                yaxis=dict(gridcolor="#e8e8e8"),
            )
            st.plotly_chart(fig_acf, use_container_width=True)

        with col_pacf:
            fig_pacf = go.Figure()
            colors_pacf = ["#C0504D" if abs(v) > ci_bound and i > 0 else "#76933C" for i, v in enumerate(pacf_vals)]
            fig_pacf.add_trace(go.Bar(
                x=lags, y=pacf_vals, marker_color=colors_pacf, name="PACF",
                hovertemplate="Retard %{x}<br>PACF: %{y:.3f}<extra></extra>"
            ))
            fig_pacf.add_hline(y=ci_bound, line_dash="dash", line_color="#E46C0A", line_width=1)
            fig_pacf.add_hline(y=-ci_bound, line_dash="dash", line_color="#E46C0A", line_width=1)
            fig_pacf.add_hline(y=0, line_color="#999", line_width=0.5)

            for lag in [4, 8, 12]:
                if lag < len(pacf_vals) and abs(pacf_vals[lag]) > ci_bound:
                    fig_pacf.add_annotation(x=lag, y=pacf_vals[lag], text=f"S={lag}",
                                           showarrow=True, arrowhead=2, arrowsize=0.8,
                                           font=dict(size=8, color="#E46C0A"))

            fig_pacf.update_layout(
                title="PACF — Autocorrélation partielle", height=320,
                xaxis_title="Retard (semaines)",
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=40, l=50, r=20),
                yaxis=dict(gridcolor="#e8e8e8"),
            )
            st.plotly_chart(fig_pacf, use_container_width=True)

        st.caption(
            "🔴 Barres rouges = significatives (hors bande de confiance 95%). "
            "Les pics aux retards 4, 8, 12... confirment la saisonnalité ~mensuelle. "
            "ACF → ordre MA (q). PACF → ordre AR (p)."
        )

    except Exception as e:
        st.warning(f"Autocorrélations non disponibles : {e}")

    # === Spectral / Periodogram ===
    st.markdown("---")
    st.markdown("###### 🌊 Analyse spectrale — Périodogramme")
    st.caption("Identifie les fréquences dominantes (cycles récurrents) dans les flux de trésorerie.")

    try:
        flux_centered = flux_clean - flux_clean.mean()
        n = len(flux_centered)
        fft_vals = np.fft.rfft(flux_centered)
        power = np.abs(fft_vals) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=1)  # en cycles/semaine
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # Skip DC component (index 0)
        mask = (freqs > 0) & (periods <= n / 2) & (periods >= 2)
        plot_periods = periods[mask]
        plot_power = power[mask]

        fig_spectral = go.Figure()
        fig_spectral.add_trace(go.Scatter(
            x=plot_periods, y=plot_power,
            mode="lines", fill="tozeroy",
            line=dict(color="#2E75B6", width=1.5),
            fillcolor="rgba(46,117,182,0.15)",
            hovertemplate="Période: %{x:.1f} sem.<br>Puissance: %{y:.1f}<extra></extra>"
        ))

        # Annotate top peaks
        top_k = 4
        top_idx = np.argsort(plot_power)[-top_k:]
        for idx in top_idx:
            p = plot_periods[idx]
            pw = plot_power[idx]
            label = ""
            if 3.5 <= p <= 4.5:
                label = f"~Mensuel ({p:.1f} sem.)"
            elif 1.8 <= p <= 2.2:
                label = f"~Bi-hebdo ({p:.1f} sem.)"
            elif 12 <= p <= 14:
                label = f"~Trimestriel ({p:.1f} sem.)"
            elif 6 <= p <= 7:
                label = f"~6 sem. ({p:.1f})"
            else:
                label = f"{p:.1f} sem."
            fig_spectral.add_annotation(
                x=p, y=pw, text=label,
                showarrow=True, arrowhead=2, arrowsize=0.8, ay=-30,
                font=dict(size=9, color="#C0504D")
            )

        fig_spectral.update_layout(
            height=350,
            xaxis_title="Période (semaines)",
            yaxis_title="Puissance spectrale",
            plot_bgcolor="#fafbfc",
            xaxis=dict(gridcolor="#e8e8e8", type="log",
                       tickvals=[2, 4, 6, 8, 13, 26, 52],
                       ticktext=["2s", "4s\n(mois)", "6s", "8s", "13s\n(trim)", "26s\n(sem)", "52s\n(an)"]),
            yaxis=dict(gridcolor="#e8e8e8"),
            margin=dict(t=20, b=50, l=60, r=40),
        )
        st.plotly_chart(fig_spectral, use_container_width=True)

        st.caption(
            "Les pics identifient les cycles dominants. Un pic à ~4 semaines confirme le cycle mensuel "
            "(facturation/paie). Un pic à ~2 semaines reflète le cycle bi-hebdomadaire de paie. "
            "Un pic à ~13 semaines indique un effet trimestriel (loyer, acomptes fiscaux)."
        )

    except Exception as e:
        st.warning(f"Analyse spectrale non disponible : {e}")

    # === Enhanced residuals diagnostics ===
    st.markdown("---")
    st.markdown("###### 🔍 Diagnostics des résidus du modèle SARIMAX")

    with st.expander("Voir les diagnostics détaillés des résidus", expanded=False):
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_res = go.Figure()
            resid_colors = ["#C0504D" if abs(r) > 2 * resid_std else "#2E75B6" for r in residuals]
            fig_res.add_trace(go.Bar(
                y=residuals, marker_color=resid_colors, name="Résidus",
                hovertemplate="Obs %{x}<br>Résidu: %{y:.1f} k$<extra></extra>"
            ))
            fig_res.add_hline(y=0, line_color="#999")
            fig_res.add_hline(y=2*resid_std, line_dash="dot", line_color="#E46C0A",
                              annotation_text=f"+2σ ({2*resid_std:.1f})", annotation_font=dict(size=9))
            fig_res.add_hline(y=-2*resid_std, line_dash="dot", line_color="#E46C0A",
                              annotation_text=f"-2σ ({-2*resid_std:.1f})", annotation_font=dict(size=9))
            fig_res.update_layout(
                title="Résidus SARIMAX dans le temps", height=320,
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=30),
                xaxis_title="Observation", yaxis_title="Résidu (k$)",
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col_d2:
            fig_hist_res = go.Figure()
            fig_hist_res.add_trace(go.Histogram(
                x=residuals, nbinsx=40,
                marker_color="#2E75B6", opacity=0.7, name="Résidus"
            ))
            # Overlay normal curve
            x_norm = np.linspace(residuals.min(), residuals.max(), 100)
            y_norm = (1 / (resid_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - np.mean(residuals)) / resid_std) ** 2)
            # Scale to match histogram
            bin_width = (residuals.max() - residuals.min()) / 40
            y_norm_scaled = y_norm * len(residuals) * bin_width
            fig_hist_res.add_trace(go.Scatter(
                x=x_norm, y=y_norm_scaled,
                mode="lines", line=dict(color="#C0504D", width=2, dash="dash"),
                name="Normale théorique"
            ))
            fig_hist_res.add_vline(x=0, line_color="red", line_width=1)
            fig_hist_res.update_layout(
                title="Distribution des résidus vs Normale", height=320,
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=30),
                xaxis_title="Résidu (k$)", yaxis_title="Fréquence",
            )
            st.plotly_chart(fig_hist_res, use_container_width=True)

        # QQ-ish plot and residual ACF
        col_d3, col_d4 = st.columns(2)

        with col_d3:
            # Residual ACF
            resid_clean_arr = residuals[~np.isnan(residuals)] if hasattr(residuals, '__iter__') else np.array([residuals])
            if len(resid_clean_arr) > 10:
                resid_acf = acf(resid_clean_arr, nlags=min(20, len(resid_clean_arr) // 3), fft=True)
                ci_r = 1.96 / np.sqrt(len(resid_clean_arr))
                fig_racf = go.Figure()
                r_colors = ["#C0504D" if abs(v) > ci_r and i > 0 else "#8DB4E2" for i, v in enumerate(resid_acf)]
                fig_racf.add_trace(go.Bar(x=list(range(len(resid_acf))), y=resid_acf, marker_color=r_colors))
                fig_racf.add_hline(y=ci_r, line_dash="dash", line_color="#E46C0A", line_width=1)
                fig_racf.add_hline(y=-ci_r, line_dash="dash", line_color="#E46C0A", line_width=1)
                fig_racf.add_hline(y=0, line_color="#999", line_width=0.5)
                fig_racf.update_layout(
                    title="ACF des résidus", height=300,
                    xaxis_title="Retard", plot_bgcolor="#fafbfc",
                    margin=dict(t=40, b=40, l=50, r=20),
                )
                st.plotly_chart(fig_racf, use_container_width=True)
                st.caption("Si les résidus sont bien du bruit blanc, aucune barre ne devrait dépasser les bandes orange.")

        with col_d4:
            # Residuals vs fitted
            fitted = results.fittedvalues
            fitted_arr = np.array(fitted).flatten()
            resid_arr = np.array(residuals).flatten()
            if len(fitted_arr) == len(resid_arr):
                fig_rvf = go.Figure()
                fig_rvf.add_trace(go.Scatter(
                    x=fitted_arr, y=resid_arr,
                    mode="markers", marker=dict(color="#2E75B6", size=4, opacity=0.6),
                    hovertemplate="Ajusté: %{x:.1f}<br>Résidu: %{y:.1f}<extra></extra>"
                ))
                fig_rvf.add_hline(y=0, line_color="#999")
                fig_rvf.add_hline(y=2*resid_std, line_dash="dot", line_color="#E46C0A", line_width=1)
                fig_rvf.add_hline(y=-2*resid_std, line_dash="dot", line_color="#E46C0A", line_width=1)
                fig_rvf.update_layout(
                    title="Résidus vs Valeurs ajustées", height=300,
                    xaxis_title="Valeur ajustée (k$)", yaxis_title="Résidu (k$)",
                    plot_bgcolor="#fafbfc", margin=dict(t=40, b=40, l=50, r=20),
                )
                st.plotly_chart(fig_rvf, use_container_width=True)
                st.caption("Un bon modèle montre des résidus dispersés sans pattern. Un entonnoir indiquerait de l'hétéroscédasticité.")

        # Summary stats
        n_outliers = int(np.sum(np.abs(residuals) > 2 * resid_std))
        skew = float(pd.Series(residuals).skew())
        kurt = float(pd.Series(residuals).kurtosis())

        rs1, rs2, rs3, rs4 = st.columns(4)
        rs1.metric("Moyenne résidus", f"{np.mean(residuals):.2f} k$",
                   help="Devrait être proche de 0 (modèle non biaisé)")
        rs2.metric("Outliers (>2σ)", f"{n_outliers} / {len(residuals)}",
                   help="Observations avec résidu supérieur à 2 écarts-types")
        rs3.metric("Asymétrie (skew)", f"{skew:.2f}",
                   delta="OK" if abs(skew) < 0.5 else "Asymétrique",
                   delta_color="normal" if abs(skew) < 0.5 else "inverse",
                   help="0 = symétrique. >0.5 ou <-0.5 = distribution asymétrique")
        rs4.metric("Kurtosis", f"{kurt:.2f}",
                   delta="OK" if abs(kurt) < 1 else "Queues épaisses" if kurt > 1 else "Queues fines",
                   delta_color="normal" if abs(kurt) < 1 else "inverse",
                   help="0 = normal. >0 = queues plus épaisses que la normale")

    # ------------------------------------------
    # Forecast table
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 📋 Tableau de prévision détaillé")

    fc_table = pd.DataFrame({
        "Semaine": forecast_labels,
        "Date": forecast_dates.strftime("%Y-%m-%d"),
        "Flux net prévu (k$)": np.round(forecast_mean_vals, 0).astype(int),
        "Solde prévu (k$)": np.round(solde_forecast_mean, 0).astype(int),
        f"Solde IC bas {ci_level}%": mc_p_low.round(0).astype(int),
        f"Solde IC haut {ci_level}%": mc_p_high.round(0).astype(int),
        "Médiane MC (k$)": mc_median.round(0).astype(int),
        "Prob. < seuil (%)": prob_sous_seuil.round(1),
        "Prob. < 0 (%)": prob_negatif.round(1),
        "DSO projeté": exog_forecast[:, 0].round(1),
        "Backlog projeté": exog_forecast[:, 1].round(0).astype(int),
    })

    def style_risk(val):
        if isinstance(val, (int, float)):
            if val > 25: return "background-color: #f8d7da; color: #C0504D; font-weight: bold"
            elif val > 10: return "background-color: #fff3cd; color: #E46C0A; font-weight: bold"
        return ""

    styled_fc = fc_table.style\
        .applymap(style_risk, subset=["Prob. < seuil (%)", "Prob. < 0 (%)"])\
        .format("{:,.0f}", subset=["Flux net prévu (k$)", "Solde prévu (k$)",
                                    f"Solde IC bas {ci_level}%", f"Solde IC haut {ci_level}%",
                                    "Médiane MC (k$)", "Backlog projeté"])\
        .format("{:.1f}", subset=["Prob. < seuil (%)", "Prob. < 0 (%)", "DSO projeté"])

    st.dataframe(styled_fc, use_container_width=True, height=520)

    csv_fc = fc_table.to_csv(index=False, sep=";", decimal=",")
    st.download_button(
        "📥 Télécharger les prévisions SARIMAX (CSV)",
        csv_fc,
        file_name="sarimax_forecast.csv",
        mime="text/csv"
    )

    # ------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 🎛️ Analyse de sensibilité — Impact des exogènes sur le solde final")

    # Vary each exogenous variable and observe impact on final solde
    base_final = float(solde_forecast_mean[-1])

    sensitivities = []
    for var_name, var_idx, var_range, var_unit in [
        ("DSO", 0, np.arange(30, 66, 5), "jours"),
        ("Backlog", 1, np.arange(150, 501, 50), "k$"),
        ("PMI", 2, np.arange(46, 59, 2), ""),
    ]:
        for val in var_range:
            exog_test = exog_forecast.copy()
            exog_test[:, var_idx] = val
            try:
                fc_test = results.get_forecast(steps=horizon, exog=exog_test)
                test_final = last_solde + np.cumsum(np.array(fc_test.predicted_mean))[-1]
                sensitivities.append({
                    "Variable": f"{var_name} ({var_unit})" if var_unit else var_name,
                    "Valeur": val,
                    "Solde_Final_k$": round(test_final, 0),
                })
            except:
                pass

    if sensitivities:
        df_sens = pd.DataFrame(sensitivities)

        fig_sens = go.Figure()
        for var in df_sens["Variable"].unique():
            mask = df_sens["Variable"] == var
            fig_sens.add_trace(go.Scatter(
                x=df_sens[mask]["Valeur"],
                y=df_sens[mask]["Solde_Final_k$"],
                mode="lines+markers",
                name=var,
                hovertemplate=f"<b>{var}</b>: %{{x}}<br>Solde final: %{{y:.0f}} k$<extra></extra>"
            ))

        fig_sens.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=1.5,
                           annotation_text="Seuil critique")
        fig_sens.add_hline(y=base_final, line_dash="dot", line_color="#999",
                           annotation_text=f"Base ({base_final:.0f} k$)")

        fig_sens.update_layout(
            height=400,
            xaxis_title="Valeur de la variable exogène",
            yaxis_title="Solde final projeté (k$)",
            plot_bgcolor="#fafbfc",
            yaxis=dict(gridcolor="#e8e8e8"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=40, b=40, l=60, r=40),
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.caption(
            "Ce graphique montre comment le solde final varie quand on change une variable "
            "exogène en gardant les autres constantes (ceteris paribus). "
            "Les pentes reflètent les coefficients du modèle SARIMAX."
        )
