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
from statsmodels.tsa.stattools import adfuller
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
        coef_df = pd.DataFrame({
            "Paramètre": results.params.index,
            "Coefficient": results.params.values.round(4),
            "P-value": results.pvalues.values.round(4),
            "Significatif (5%)": ["✅" if p < 0.05 else "❌" for p in results.pvalues.values],
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

    # Vertical line at forecast start
    fig_fc.add_vline(x=forecast_dates[0], line_dash="dot", line_color="#999",
                     annotation_text="Début prévision", annotation_position="top left",
                     annotation_font=dict(size=9, color="#666"))

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
    # Chart 3: Residuals diagnostics
    # ------------------------------------------
    with st.expander("🔍 Diagnostics des résidus"):
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                y=residuals, mode="lines",
                line=dict(color="#2E75B6", width=1),
                name="Résidus"
            ))
            fig_res.add_hline(y=0, line_color="#999")
            fig_res.add_hline(y=2*resid_std, line_dash="dot", line_color="#E46C0A",
                              annotation_text="+2σ", annotation_font=dict(size=9))
            fig_res.add_hline(y=-2*resid_std, line_dash="dot", line_color="#E46C0A",
                              annotation_text="-2σ", annotation_font=dict(size=9))
            fig_res.update_layout(
                title="Résidus dans le temps", height=300,
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_res, use_container_width=True)

        with col_d2:
            fig_hist_res = go.Figure()
            fig_hist_res.add_trace(go.Histogram(
                x=residuals, nbinsx=40,
                marker_color="#2E75B6", opacity=0.7, name="Résidus"
            ))
            fig_hist_res.add_vline(x=0, line_color="red", line_width=1)
            fig_hist_res.update_layout(
                title="Distribution des résidus", height=300,
                plot_bgcolor="#fafbfc", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_hist_res, use_container_width=True)

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
