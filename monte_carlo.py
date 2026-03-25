"""
Module Monte Carlo — Simulation de scénarios de trésorerie
Intégration : ajouter comme onglet dans app.py existant
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def render_monte_carlo_tab(df: pd.DataFrame, solde_initial: float, seuil_critique: float):
    """
    Render the Monte Carlo simulation tab.
    
    Args:
        df: DataFrame from main app with columns Clients, Autres_Entrees, Subventions,
            Fournisseurs, Masse_Sal, Loyer, Remb_Dette, Capex, Taxes, Semaine
        solde_initial: Starting cash balance (k$)
        seuil_critique: Critical threshold (k$)
    """

    st.markdown("#### 🎲 Simulation Monte Carlo — Scénarios de trésorerie")
    st.markdown(
        "La simulation génère des milliers de trajectoires possibles en faisant varier "
        "les encaissements et décaissements autour des hypothèses de base, selon les "
        "paramètres de volatilité ci-dessous."
    )

    # ------------------------------------------
    # Simulation parameters (sidebar-like columns)
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### ⚙️ Paramètres de simulation")

    p1, p2, p3 = st.columns(3)
    with p1:
        n_sims = st.select_slider(
            "Nombre de simulations",
            options=[500, 1000, 2000, 5000, 10000],
            value=2000,
            help="Plus de simulations = résultats plus stables mais calcul plus long"
        )
    with p2:
        seed = st.number_input("Seed aléatoire", value=42, step=1,
                               help="Fixe le générateur pour reproduire les mêmes résultats")
    with p3:
        ci_level = st.select_slider(
            "Intervalle de confiance",
            options=[80, 90, 95, 99],
            value=90,
            help="Pourcentage des trajectoires contenues dans la bande"
        )

    st.markdown("##### 📊 Volatilité par catégorie")
    st.caption("Coefficient de variation (écart-type / moyenne). Plus c'est élevé, plus la dispersion est grande.")

    v1, v2, v3, v4 = st.columns(4)
    with v1:
        vol_clients = st.slider("Clients A/R", 0.05, 0.50, 0.15, 0.01,
                                help="Variabilité des encaissements clients")
    with v2:
        vol_autres = st.slider("Autres entrées", 0.05, 0.80, 0.30, 0.01,
                               help="Variabilité des autres encaissements")
    with v3:
        vol_fournisseurs = st.slider("Fournisseurs A/P", 0.05, 0.50, 0.15, 0.01,
                                     help="Variabilité des paiements fournisseurs")
    with v4:
        prob_retard = st.slider("Prob. retard client (%)", 0, 30, 10, 1,
                                help="Probabilité qu'un client paie en retard (décalage d'une semaine)")

    st.markdown("---")

    # ------------------------------------------
    # Run simulation
    # ------------------------------------------
    rng = np.random.default_rng(int(seed))
    n_weeks = len(df)

    # Base values from the main scenario
    base_clients = df["Clients"].values.astype(float)
    base_autres = df["Autres_Entrees"].values.astype(float)
    base_subventions = df["Subventions"].values.astype(float)
    base_fournisseurs = df["Fournisseurs"].values.astype(float)
    base_masse_sal = df["Masse_Sal"].values.astype(float)
    base_loyer = df["Loyer"].values.astype(float)
    base_remb = df["Remb_Dette"].values.astype(float)
    base_capex = df["Capex"].values.astype(float)
    base_taxes = df["Taxes"].values.astype(float)

    # Matrices (n_sims x n_weeks)
    # Encaissements: lognormal to ensure positivity
    sim_clients = rng.lognormal(
        mean=np.log(base_clients.clip(min=1)) - 0.5 * vol_clients**2,
        sigma=vol_clients,
        size=(n_sims, n_weeks)
    )

    sim_autres = rng.lognormal(
        mean=np.log(base_autres.clip(min=1)) - 0.5 * vol_autres**2,
        sigma=vol_autres,
        size=(n_sims, n_weeks)
    )

    # Subventions: binary (received or not, with 90% probability of base)
    prob_sub = 0.9
    sim_subventions = np.where(
        rng.random((n_sims, n_weeks)) < prob_sub,
        np.tile(base_subventions, (n_sims, 1)),
        0
    )

    # Simulate client payment delays
    delay_mask = rng.random((n_sims, n_weeks)) < (prob_retard / 100.0)
    for sim in range(n_sims):
        for w in range(n_weeks - 1):
            if delay_mask[sim, w]:
                delayed = sim_clients[sim, w] * 0.5
                sim_clients[sim, w] -= delayed
                sim_clients[sim, w + 1] += delayed

    # Décaissements
    sim_fourn = rng.lognormal(
        mean=np.log(base_fournisseurs.clip(min=1)) - 0.5 * vol_fournisseurs**2,
        sigma=vol_fournisseurs,
        size=(n_sims, n_weeks)
    )

    # Fixed costs: small variation (2%)
    sim_masse_sal = np.tile(base_masse_sal, (n_sims, 1)) * rng.normal(1, 0.02, (n_sims, n_weeks))
    sim_loyer = np.tile(base_loyer, (n_sims, 1))  # Fixed
    sim_remb = np.tile(base_remb, (n_sims, 1))    # Fixed
    sim_capex = np.tile(base_capex, (n_sims, 1)) * np.where(
        rng.random((n_sims, n_weeks)) < 0.85, 1.0,
        rng.uniform(1.0, 1.5, (n_sims, n_weeks))  # 15% chance of cost overrun
    )
    sim_taxes = np.tile(base_taxes, (n_sims, 1))  # Fixed

    # Total flows
    sim_enc = sim_clients + sim_autres + sim_subventions
    sim_dec = sim_fourn + sim_masse_sal + sim_loyer + sim_remb + sim_capex + sim_taxes
    sim_flux_net = sim_enc - sim_dec

    # Cumulative balance
    sim_solde = np.zeros((n_sims, n_weeks))
    sim_solde[:, 0] = solde_initial + sim_flux_net[:, 0]
    for w in range(1, n_weeks):
        sim_solde[:, w] = sim_solde[:, w - 1] + sim_flux_net[:, w]

    # ------------------------------------------
    # Statistics
    # ------------------------------------------
    alpha_low = (100 - ci_level) / 2
    alpha_high = 100 - alpha_low

    solde_mean = sim_solde.mean(axis=0)
    solde_median = np.median(sim_solde, axis=0)
    solde_p_low = np.percentile(sim_solde, alpha_low, axis=0)
    solde_p_high = np.percentile(sim_solde, alpha_high, axis=0)
    solde_p5 = np.percentile(sim_solde, 5, axis=0)
    solde_p95 = np.percentile(sim_solde, 95, axis=0)
    solde_min = sim_solde.min(axis=0)
    solde_max = sim_solde.max(axis=0)

    # Final solde distribution
    solde_final_all = sim_solde[:, -1]
    prob_sous_seuil = (solde_final_all < seuil_critique).mean() * 100
    prob_negatif = (solde_final_all < 0).mean() * 100

    # Week-by-week probability of being under threshold
    prob_sous_seuil_hebdo = (sim_solde < seuil_critique).mean(axis=0) * 100

    # Minimum solde across all weeks per simulation
    sim_solde_min = sim_solde.min(axis=1)
    prob_toucher_seuil = (sim_solde_min < seuil_critique).mean() * 100

    semaines = df["Semaine"].values

    # ------------------------------------------
    # KPIs
    # ------------------------------------------
    st.markdown("##### 📈 Résultats clés")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Prob. solde < seuil en S13",
        f"{prob_sous_seuil:.1f} %",
        delta="Risque élevé" if prob_sous_seuil > 25 else "Risque modéré" if prob_sous_seuil > 10 else "Risque faible",
        delta_color="inverse" if prob_sous_seuil > 10 else "normal"
    )
    k2.metric(
        "Prob. solde négatif en S13",
        f"{prob_negatif:.1f} %",
        delta="Critique" if prob_negatif > 10 else "Acceptable",
        delta_color="inverse" if prob_negatif > 5 else "normal"
    )
    k3.metric(
        "Prob. toucher le seuil (sur 13 sem.)",
        f"{prob_toucher_seuil:.1f} %",
        help="Probabilité que le solde passe sous le seuil au moins une fois sur l'horizon"
    )
    k4.metric(
        "Solde final médian",
        f"{solde_median[-1]:.0f} k$",
        delta=f"IC {ci_level}% : [{solde_p_low[-1]:.0f}, {solde_p_high[-1]:.0f}]"
    )

    # ------------------------------------------
    # Chart 1: Fan chart (cone of trajectories)
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 🌀 Cône de trajectoires — Solde projeté")

    fig_fan = go.Figure()

    # Min-Max band (very light)
    fig_fan.add_trace(go.Scatter(
        x=list(semaines) + list(semaines)[::-1],
        y=list(solde_max) + list(solde_min)[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.05)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Min-Max", showlegend=True, hoverinfo="skip"
    ))

    # CI band
    fig_fan.add_trace(go.Scatter(
        x=list(semaines) + list(semaines)[::-1],
        y=list(solde_p_high) + list(solde_p_low)[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"IC {ci_level}%", showlegend=True, hoverinfo="skip"
    ))

    # P5-P95 band
    fig_fan.add_trace(go.Scatter(
        x=list(semaines) + list(semaines)[::-1],
        y=list(solde_p95) + list(solde_p5)[::-1],
        fill="toself", fillcolor="rgba(46,117,182,0.25)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P5-P95", showlegend=True, hoverinfo="skip"
    ))

    # Median line
    fig_fan.add_trace(go.Scatter(
        x=semaines, y=solde_median,
        mode="lines+markers", name="Médiane",
        line=dict(color="#2E75B6", width=3),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>Médiane: %{y:.0f} k$<extra></extra>"
    ))

    # Mean line
    fig_fan.add_trace(go.Scatter(
        x=semaines, y=solde_mean,
        mode="lines", name="Moyenne",
        line=dict(color="#76933C", width=2, dash="dot"),
        hovertemplate="<b>%{x}</b><br>Moyenne: %{y:.0f} k$<extra></extra>"
    ))

    # Base scenario
    base_solde = df["Solde_Fin"].values
    fig_fan.add_trace(go.Scatter(
        x=semaines, y=base_solde,
        mode="lines+markers", name="Scénario de base",
        line=dict(color="#E46C0A", width=2, dash="dash"),
        marker=dict(size=5, symbol="diamond"),
        hovertemplate="<b>%{x}</b><br>Base: %{y:.0f} k$<extra></extra>"
    ))

    # Seuil
    fig_fan.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Seuil critique ({seuil_critique} k$)",
                      annotation_position="top right",
                      annotation_font=dict(color="red", size=10))

    # Zero line
    fig_fan.add_hline(y=0, line_color="#999", line_width=1)

    fig_fan.update_layout(
        height=500,
        yaxis_title="Solde de trésorerie (k$)",
        plot_bgcolor="#fafbfc",
        yaxis=dict(gridcolor="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(t=40, b=40, l=60, r=40),
    )

    st.plotly_chart(fig_fan, use_container_width=True)

    # ------------------------------------------
    # Chart 2: Distribution du solde final
    # ------------------------------------------
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("##### 📊 Distribution du solde final (S13)")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=solde_final_all, nbinsx=60,
            marker_color="#2E75B6", opacity=0.75,
            name="Simulations",
            hovertemplate="Solde: %{x:.0f} k$<br>Fréquence: %{y}<extra></extra>"
        ))
        fig_hist.add_vline(x=seuil_critique, line_dash="dash", line_color="red", line_width=2,
                           annotation_text=f"Seuil ({seuil_critique} k$)")
        fig_hist.add_vline(x=np.median(solde_final_all), line_dash="solid", line_color="#76933C", line_width=2,
                           annotation_text=f"Médiane ({np.median(solde_final_all):.0f})")
        fig_hist.add_vline(x=0, line_color="#333", line_width=1, annotation_text="Zéro")
        fig_hist.update_layout(
            height=400, xaxis_title="Solde fin S13 (k$)", yaxis_title="Nombre de simulations",
            plot_bgcolor="#fafbfc", showlegend=False,
            margin=dict(t=20, b=40, l=60, r=40),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        st.markdown("##### ⚠️ Probabilité hebdo de passer sous le seuil")
        fig_prob = go.Figure()
        colors = ["#C0504D" if p > 25 else "#E46C0A" if p > 10 else "#76933C" for p in prob_sous_seuil_hebdo]
        fig_prob.add_trace(go.Bar(
            x=semaines, y=prob_sous_seuil_hebdo,
            marker_color=colors,
            text=[f"{p:.1f}%" for p in prob_sous_seuil_hebdo],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="<b>%{x}</b><br>Prob. sous seuil: %{y:.1f}%<extra></extra>"
        ))
        fig_prob.add_hline(y=25, line_dash="dot", line_color="#C0504D",
                           annotation_text="Risque élevé (25%)", annotation_font=dict(size=9, color="#C0504D"))
        fig_prob.add_hline(y=10, line_dash="dot", line_color="#E46C0A",
                           annotation_text="Risque modéré (10%)", annotation_font=dict(size=9, color="#E46C0A"))
        fig_prob.update_layout(
            height=400, yaxis_title="Probabilité (%)", yaxis=dict(range=[0, max(prob_sous_seuil_hebdo) * 1.3 + 5]),
            plot_bgcolor="#fafbfc",
            margin=dict(t=20, b=40, l=60, r=40),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ------------------------------------------
    # Chart 3: Sample trajectories
    # ------------------------------------------
    st.markdown("##### 🔀 Échantillon de trajectoires individuelles")
    n_sample = st.slider("Nombre de trajectoires à afficher", 10, 100, 30, 5)

    fig_traj = go.Figure()
    sample_idx = rng.choice(n_sims, size=min(n_sample, n_sims), replace=False)
    for idx in sample_idx:
        fig_traj.add_trace(go.Scatter(
            x=semaines, y=sim_solde[idx],
            mode="lines", line=dict(width=0.8, color="rgba(46,117,182,0.2)"),
            showlegend=False, hoverinfo="skip"
        ))
    fig_traj.add_trace(go.Scatter(
        x=semaines, y=solde_median,
        mode="lines+markers", name="Médiane",
        line=dict(color="#2E75B6", width=3), marker=dict(size=5)
    ))
    fig_traj.add_trace(go.Scatter(
        x=semaines, y=base_solde,
        mode="lines", name="Scénario de base",
        line=dict(color="#E46C0A", width=2, dash="dash")
    ))
    fig_traj.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=2)
    fig_traj.add_hline(y=0, line_color="#999", line_width=1)
    fig_traj.update_layout(
        height=420, yaxis_title="Solde (k$)",
        plot_bgcolor="#fafbfc", yaxis=dict(gridcolor="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=40, b=40, l=60, r=40),
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    # ------------------------------------------
    # Summary statistics table
    # ------------------------------------------
    st.markdown("---")
    st.markdown("##### 📋 Statistiques par semaine")

    stats_df = pd.DataFrame({
        "Semaine": semaines,
        "Moyenne": solde_mean.round(0).astype(int),
        "Médiane": solde_median.round(0).astype(int),
        f"P{alpha_low:.0f}": solde_p_low.round(0).astype(int),
        f"P{alpha_high:.0f}": solde_p_high.round(0).astype(int),
        "P5": solde_p5.round(0).astype(int),
        "P95": solde_p95.round(0).astype(int),
        "Min": solde_min.round(0).astype(int),
        "Max": solde_max.round(0).astype(int),
        "Prob. < seuil (%)": prob_sous_seuil_hebdo.round(1),
        "Scénario base": base_solde.astype(int),
    })

    def highlight_risk(val):
        if isinstance(val, (int, float)):
            if val > 25:
                return "background-color: #f8d7da; color: #C0504D; font-weight: bold"
            elif val > 10:
                return "background-color: #fff3cd; color: #E46C0A; font-weight: bold"
        return ""

    styled_stats = stats_df.style\
        .applymap(highlight_risk, subset=["Prob. < seuil (%)"])\
        .format("{:,.0f}", subset=[c for c in stats_df.columns if c not in ["Semaine", "Prob. < seuil (%)"]])\
        .format("{:.1f}", subset=["Prob. < seuil (%)"])

    st.dataframe(styled_stats, use_container_width=True, height=520)

    # Download stats
    csv_stats = stats_df.to_csv(index=False, sep=";", decimal=",")
    st.download_button(
        "📥 Télécharger les statistiques Monte Carlo (CSV)",
        csv_stats,
        file_name="monte_carlo_stats.csv",
        mime="text/csv"
    )
