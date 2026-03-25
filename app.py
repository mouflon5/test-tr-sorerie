import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from monte_carlo import render_monte_carlo_tab

# =============================================
# Page config
# =============================================
st.set_page_config(
    page_title="Trésorerie PME — Rolling Forecast 13 sem.",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================
# Custom CSS
# =============================================
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e8f0fe 100%);
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #2E75B6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { font-size: 0.78rem !important; font-weight: 600; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# Sidebar — Hypothèses modifiables
# =============================================
st.sidebar.image("https://img.icons8.com/fluency/96/cash-register.png", width=64)
st.sidebar.title("⚙️ Hypothèses")
st.sidebar.markdown("---")

solde_initial = st.sidebar.number_input("Solde initial (k$)", value=180, step=10, min_value=0)
seuil_critique = st.sidebar.number_input("Seuil critique (k$)", value=50, step=5, min_value=0)
date_debut = st.sidebar.date_input("Date début S1", value=datetime(2026, 3, 30))

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Encaissements hebdo")
clients_base = st.sidebar.slider("Clients A/R moyen (k$)", 50, 150, 88)
autres_base = st.sidebar.slider("Autres entrées moyen (k$)", 0, 30, 6)
sub_s4 = st.sidebar.number_input("Subvention S4 (k$)", value=15, step=5, min_value=0)
sub_s9 = st.sidebar.number_input("Subvention S9 (k$)", value=20, step=5, min_value=0)

st.sidebar.markdown("---")
st.sidebar.subheader("💸 Décaissements hebdo")
fourn_base = st.sidebar.slider("Fournisseurs A/P moyen (k$)", 20, 100, 49)
masse_sal = st.sidebar.slider("Masse salariale fixe (k$)", 20, 80, 38)
loyer_trim = st.sidebar.number_input("Loyer trimestriel (k$)", value=12, step=2, min_value=0)
remb_dette = st.sidebar.slider("Rembt. dette fixe (k$)", 0, 30, 8)
capex_s3 = st.sidebar.number_input("Capex S3 (k$)", value=15, step=5, min_value=0)
capex_s7 = st.sidebar.number_input("Capex S7 (k$)", value=25, step=5, min_value=0)
capex_s10 = st.sidebar.number_input("Capex S10 (k$)", value=10, step=5, min_value=0)
taxes_s6 = st.sidebar.number_input("Taxes S6 (k$)", value=20, step=5, min_value=0)
taxes_s12 = st.sidebar.number_input("Taxes S12 (k$)", value=15, step=5, min_value=0)

# =============================================
# Generate data from sidebar inputs
# =============================================
import random
random.seed(42)

weeks = list(range(1, 14))
sem_ids = [f"S{i}" for i in weeks]

# Simulate variation around base
var = [0.97, 1.05, 0.89, 1.00, 1.08, 0.91, 1.25, 1.19, 0.80, 1.02, 0.97, 1.00, 1.05]
var_f = [0.92, 1.02, 0.86, 0.98, 1.12, 0.82, 1.22, 1.18, 0.71, 1.06, 0.98, 1.02, 1.12]
var_a = [0.83, 1.33, 0.50, 1.00, 0.67, 2.00, 1.17, 0.83, 0.50, 1.33, 1.00, 0.67, 1.67]

data_clients = [round(clients_base * v) for v in var]
data_autres = [max(0, round(autres_base * v)) for v in var_a]
data_subventions = [0]*13
if sub_s4 > 0: data_subventions[3] = sub_s4
if sub_s9 > 0: data_subventions[8] = sub_s9

data_fournisseurs = [round(fourn_base * v) for v in var_f]
data_masse_sal = [masse_sal] * 13
data_loyer = [0]*13
for i in [0, 4, 8, 12]:
    if i < 13: data_loyer[i] = loyer_trim
data_remb = [remb_dette] * 13
data_capex = [0]*13
if capex_s3 > 0: data_capex[2] = capex_s3
if capex_s7 > 0: data_capex[6] = capex_s7
if capex_s10 > 0: data_capex[9] = capex_s10
data_taxes = [0]*13
if taxes_s6 > 0: data_taxes[5] = taxes_s6
if taxes_s12 > 0: data_taxes[11] = taxes_s12

# Build main dataframe
rows = []
cumul = solde_initial
for i in range(13):
    enc = data_clients[i] + data_autres[i] + data_subventions[i]
    dec = data_fournisseurs[i] + data_masse_sal[i] + data_loyer[i] + data_remb[i] + data_capex[i] + data_taxes[i]
    flux = enc - dec
    s_debut = cumul
    s_fin = cumul + flux
    cumul = s_fin
    d = datetime.combine(date_debut, datetime.min.time()) + timedelta(weeks=i)
    rows.append({
        "Semaine": sem_ids[i],
        "Date": d,
        "Clients": data_clients[i],
        "Autres_Entrees": data_autres[i],
        "Subventions": data_subventions[i],
        "Enc_Total": enc,
        "Fournisseurs": data_fournisseurs[i],
        "Masse_Sal": data_masse_sal[i],
        "Loyer": data_loyer[i],
        "Remb_Dette": data_remb[i],
        "Capex": data_capex[i],
        "Taxes": data_taxes[i],
        "Dec_Total": dec,
        "Flux_Net": flux,
        "Solde_Debut": s_debut,
        "Solde_Fin": s_fin,
    })

df = pd.DataFrame(rows)

# =============================================
# Title
# =============================================
st.markdown("""
<h1 style='text-align:center; color:#2E75B6; margin-bottom:0; font-size:1.8rem;'>
    💰 Dashboard Trésorerie PME — Rolling Forecast 13 semaines
</h1>
<p style='text-align:center; color:#888; font-size:0.9rem; margin-top:4px;'>
    Données simulées • Modifiez les hypothèses dans la barre latérale
</p>
""", unsafe_allow_html=True)

# =============================================
# KPIs
# =============================================
solde_final = df.iloc[-1]["Solde_Fin"]
solde_min = df["Solde_Fin"].min()
sem_min = df.loc[df["Solde_Fin"].idxmin(), "Semaine"]
tot_enc = df["Enc_Total"].sum()
tot_dec = df["Dec_Total"].sum()
flux_net_total = df["Flux_Net"].sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Solde initial", f"{solde_initial} k$")
c2.metric("Solde final S13", f"{solde_final} k$", delta=f"{solde_final - solde_initial} k$")
c3.metric("Entrées totales", f"{tot_enc} k$")
c4.metric("Sorties totales", f"{tot_dec} k$")
c5.metric("Solde minimum", f"{solde_min} k$ ({sem_min})",
          delta=f"{solde_min - seuil_critique} k$ vs seuil",
          delta_color="normal" if solde_min >= seuil_critique else "inverse")
c6.metric("Flux net total", f"{flux_net_total} k$",
          delta="Excédent" if flux_net_total >= 0 else "Déficit",
          delta_color="normal" if flux_net_total >= 0 else "inverse")

st.markdown("")

# =============================================
# Tabs
# =============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Waterfall", "📈 Solde projeté", "🍩 Répartitions", "📋 Données", "🎲 Monte Carlo"])

# =============================================
# TAB 1: Waterfall
# =============================================
with tab1:
    fig_wf = go.Figure()

    # Build waterfall data
    wf_x = ["Début"] + list(df["Semaine"]) + ["Fin S13"]
    wf_measure = ["absolute"] + ["relative"] * 13 + ["total"]
    wf_values = [solde_initial] + list(df["Flux_Net"]) + [0]
    wf_text = [f"{solde_initial}"] + [f"{'+' if v >= 0 else ''}{v}" for v in df["Flux_Net"]] + [f"{solde_final}"]

    # Custom hover text
    hover_texts = [f"<b>Solde initial</b><br>{solde_initial} k$"]
    for _, r in df.iterrows():
        hover_texts.append(
            f"<b>{r['Semaine']}</b><br>"
            f"<span style='color:#76933C'>Encaissements: +{r['Enc_Total']} k$</span><br>"
            f"  Clients: {r['Clients']} | Autres: {r['Autres_Entrees']}"
            + (f" | Subv: {r['Subventions']}" if r['Subventions'] > 0 else "") +
            f"<br><span style='color:#C0504D'>Décaissements: −{r['Dec_Total']} k$</span><br>"
            f"  Fourn: {r['Fournisseurs']} | Sal: {r['Masse_Sal']}"
            + (f" | Loyer: {r['Loyer']}" if r['Loyer'] > 0 else "")
            + f" | Dette: {r['Remb_Dette']}"
            + (f" | Capex: {r['Capex']}" if r['Capex'] > 0 else "")
            + (f" | Taxes: {r['Taxes']}" if r['Taxes'] > 0 else "") +
            f"<br><b>Flux net: {'+' if r['Flux_Net'] >= 0 else ''}{r['Flux_Net']} k$</b>"
            f"<br>Solde cumulé: {r['Solde_Fin']} k$"
        )
    hover_texts.append(f"<b>Solde final S13</b><br>{solde_final} k$")

    fig_wf.add_trace(go.Waterfall(
        x=wf_x,
        measure=wf_measure,
        y=wf_values,
        text=wf_text,
        textposition="outside",
        textfont=dict(size=10, color="#333"),
        hovertext=hover_texts,
        hoverinfo="text",
        connector=dict(line=dict(color="#ccc", width=1, dash="dot")),
        increasing=dict(marker=dict(color="#76933C", line=dict(color="#5a7a2a", width=1))),
        decreasing=dict(marker=dict(color="#C0504D", line=dict(color="#a03a3a", width=1))),
        totals=dict(marker=dict(color="#2E75B6", line=dict(color="#1a5a9e", width=1))),
    ))

    # Seuil critique line
    fig_wf.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"Seuil critique ({seuil_critique} k$)",
                     annotation_position="top right",
                     annotation_font=dict(color="red", size=10))

    fig_wf.update_layout(
        title=dict(text="Pont de trésorerie — Waterfall des flux nets hebdomadaires", font=dict(size=15)),
        yaxis_title="k$ CAD",
        showlegend=False,
        height=480,
        margin=dict(t=60, b=40, l=60, r=40),
        plot_bgcolor="#fafbfc",
        yaxis=dict(gridcolor="#e8e8e8", zeroline=True, zerolinecolor="#999"),
    )

    st.plotly_chart(fig_wf, use_container_width=True)

# =============================================
# TAB 2: Solde projeté
# =============================================
with tab2:
    fig_sol = go.Figure()

    fig_sol.add_trace(go.Bar(
        x=df["Semaine"], y=df["Solde_Debut"],
        name="Solde début semaine",
        marker_color="#8DB4E2",
        hovertemplate="<b>%{x}</b><br>Solde début: %{y} k$<extra></extra>",
    ))
    fig_sol.add_trace(go.Bar(
        x=df["Semaine"], y=df["Solde_Fin"],
        name="Solde fin semaine",
        marker_color="#2E75B6",
        text=[f"{v}" for v in df["Solde_Fin"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>Solde fin: %{y} k$<extra></extra>",
    ))
    fig_sol.add_trace(go.Scatter(
        x=df["Semaine"], y=df["Solde_Fin"],
        mode="lines+markers",
        name="Tendance",
        line=dict(color="#1a3a5c", width=2),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>Tendance: %{y} k$<extra></extra>",
    ))

    fig_sol.add_hline(y=seuil_critique, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Seuil critique ({seuil_critique} k$)",
                      annotation_position="top right",
                      annotation_font=dict(color="red", size=10))

    fig_sol.update_layout(
        title=dict(text="Solde de trésorerie projeté — Début vs Fin de semaine", font=dict(size=15)),
        yaxis_title="k$ CAD",
        barmode="group",
        height=460,
        margin=dict(t=60, b=40, l=60, r=40),
        plot_bgcolor="#fafbfc",
        yaxis=dict(gridcolor="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    st.plotly_chart(fig_sol, use_container_width=True)

    # Alert zone
    semaines_alerte = df[df["Solde_Fin"] < seuil_critique]["Semaine"].tolist()
    if semaines_alerte:
        st.error(f"⚠️ **Alerte trésorerie** — Le solde passe sous le seuil critique ({seuil_critique} k$) aux semaines : {', '.join(semaines_alerte)}")
    else:
        st.success(f"✅ Le solde reste au-dessus du seuil critique ({seuil_critique} k$) sur tout l'horizon.")

# =============================================
# TAB 3: Pie charts
# =============================================
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        pie_enc = pd.DataFrame({
            "Catégorie": ["Clients (A/R)", "Autres entrées", "Subventions / Aides"],
            "Montant": [df["Clients"].sum(), df["Autres_Entrees"].sum(), df["Subventions"].sum()],
        })
        fig_pe = px.pie(pie_enc, values="Montant", names="Catégorie",
                        title="Répartition encaissements (13 sem.)",
                        color_discrete_sequence=["#2E75B6", "#8DB4E2", "#A9D18E"],
                        hole=0.0)
        fig_pe.update_traces(textposition="inside", textinfo="percent+label",
                            hovertemplate="<b>%{label}</b><br>%{value} k$ (%{percent})<extra></extra>")
        fig_pe.update_layout(height=400, margin=dict(t=50, b=20), showlegend=True,
                            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
        st.plotly_chart(fig_pe, use_container_width=True)

    with col_b:
        pie_dec = pd.DataFrame({
            "Catégorie": ["Fournisseurs (A/P)", "Masse salariale", "Loyer & charges",
                          "Rembt. dette", "Capex", "Taxes"],
            "Montant": [df["Fournisseurs"].sum(), df["Masse_Sal"].sum(), df["Loyer"].sum(),
                        df["Remb_Dette"].sum(), df["Capex"].sum(), df["Taxes"].sum()],
        })
        fig_pd = px.pie(pie_dec, values="Montant", names="Catégorie",
                        title="Répartition décaissements (13 sem.)",
                        color_discrete_sequence=["#C0504D", "#E46C0A", "#808080", "#BF8F00", "#595959", "#A52714"],
                        hole=0.0)
        fig_pd.update_traces(textposition="inside", textinfo="percent+label",
                            hovertemplate="<b>%{label}</b><br>%{value} k$ (%{percent})<extra></extra>")
        fig_pd.update_layout(height=400, margin=dict(t=50, b=20), showlegend=True,
                            legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5))
        st.plotly_chart(fig_pd, use_container_width=True)

    # Stacked bar breakdown
    st.markdown("#### Détail hebdomadaire — Encaissements vs Décaissements")
    fig_stack = go.Figure()
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=df["Clients"], name="Clients (A/R)", marker_color="#2E75B6"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=df["Autres_Entrees"], name="Autres entrées", marker_color="#8DB4E2"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=df["Subventions"], name="Subventions", marker_color="#A9D18E"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=[-v for v in df["Fournisseurs"]], name="Fournisseurs", marker_color="#C0504D"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=[-v for v in df["Masse_Sal"]], name="Masse salariale", marker_color="#E46C0A"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=[-v for v in df["Loyer"]], name="Loyer & charges", marker_color="#808080"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=[-v for v in df["Remb_Dette"]], name="Rembt. dette", marker_color="#BF8F00"))
    fig_stack.add_trace(go.Bar(x=df["Semaine"], y=[-(df["Capex"].iloc[i] + df["Taxes"].iloc[i]) for i in range(13)], name="Capex / Taxes", marker_color="#595959"))
    fig_stack.add_trace(go.Scatter(x=df["Semaine"], y=df["Flux_Net"], mode="lines+markers", name="Flux net",
                                   line=dict(color="black", width=2), marker=dict(size=5)))
    fig_stack.update_layout(
        barmode="relative", height=420, yaxis_title="k$ CAD",
        plot_bgcolor="#fafbfc", yaxis=dict(gridcolor="#e8e8e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(t=40, b=30),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

# =============================================
# TAB 4: Data table
# =============================================
with tab4:
    st.markdown("#### Tableau de données complet")

    display_df = df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    display_df.columns = [
        "Semaine", "Date", "Clients (A/R)", "Autres entrées", "Subventions",
        "Total Encaiss.", "Fournisseurs (A/P)", "Masse salariale", "Loyer & charges",
        "Rembt. dette", "Capex", "Taxes", "Total Décaiss.", "Flux net",
        "Solde début", "Solde fin"
    ]

    # Color flux net
    def color_flux(val):
        color = "#76933C" if val > 0 else "#C0504D" if val < 0 else "#333"
        return f"color: {color}; font-weight: bold"

    def color_solde(val):
        color = "#C0504D" if val < seuil_critique else "#2E75B6"
        return f"color: {color}; font-weight: bold"

    styled = display_df.style\
        .applymap(color_flux, subset=["Flux net"])\
        .applymap(color_solde, subset=["Solde fin"])\
        .format("{:,.0f}", subset=[c for c in display_df.columns if c not in ["Semaine", "Date"]])

    st.dataframe(styled, use_container_width=True, height=520)

    # Totals
    st.markdown("---")
    tc1, tc2, tc3, tc4 = st.columns(4)
    tc1.metric("Total Encaissements", f"{tot_enc} k$")
    tc2.metric("Total Décaissements", f"{tot_dec} k$")
    tc3.metric("Flux Net Cumulé", f"{flux_net_total} k$")
    tc4.metric("Solde Final", f"{solde_final} k$")

    # Download
    st.markdown("---")
    csv = df.to_csv(index=False, sep=";", decimal=",")
    st.download_button("📥 Télécharger les données (CSV)", csv,
                       file_name="tresorerie_pme_13sem.csv", mime="text/csv")

# =============================================
# TAB 5: Monte Carlo
# =============================================
with tab5:
    render_monte_carlo_tab(df, solde_initial, seuil_critique)

# =============================================
# Footer
# =============================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#999; font-size:0.75rem;'>
    Dashboard Trésorerie PME • Données simulées • Construit avec Streamlit + Plotly<br>
    Modifiez les hypothèses dans la barre latérale pour simuler différents scénarios
</div>
""", unsafe_allow_html=True)
