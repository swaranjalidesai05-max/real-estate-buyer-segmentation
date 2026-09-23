import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Real Estate Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Professional Font Stack */
    .stApp {
        font-family: "Inter", "Segoe UI", Roboto, system-ui, sans-serif;
    }
    
    /* Typography Hierarchy */
    h1 { font-size: 2.2rem !important; font-weight: 700 !important; margin-bottom: 0px !important; padding-bottom: 0px !important; letter-spacing: -0.01em; }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 1.5rem !important; margin-bottom: 1rem !important; opacity: 0.8 !important; }
    h3 { font-size: 1rem !important; font-weight: 500 !important; opacity: 0.7 !important; margin-top: 0.2rem !important; padding-top: 0px !important; }
    
    /* Metric Cards Styling (Native theme adaptation) */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 1.25rem;
    }
    
    div[data-testid="stMetricValue"] > div {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        line-height: 1.2;
    }
    
    div[data-testid="stMetricLabel"] > div {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
        margin-bottom: 0.25rem;
    }

    /* Layout & Separators */
    hr { border-top: 1px solid var(--border-color); opacity: 0.3; margin: 2.5rem 0; }
    
    /* Subtle Footer */
    .footer {
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.6;
        text-align: center;
        margin-top: 4rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    try:
        bs = pd.read_csv(BASE_DIR / "data" / "buyer_segments.csv")
        cp = pd.read_csv(BASE_DIR / "data" / "cluster_profile.csv")
        return bs, cp
    except FileNotFoundError:
        return None, None

buyer_segments_raw, cluster_profile = load_data()

if buyer_segments_raw is None or cluster_profile is None:
    st.error("Dataset not found in 'data/' directory. Run preprocessing notebooks first.")
    st.stop()

# Definitive cluster map mapping ML indices to human concepts
segment_names = {
    0: "Mixed Property Buyers",
    1: "Frequent Apartment Buyers",
    2: "No-Loan Buyers",
    3: "Corporate Buyers",
    4: "Apartment Home Buyers",
    5: "Large-Area Apartment Buyers",
    6: "Investment-Focused Buyers",
    7: "Loan-Financed Mixed Buyers"
}
buyer_segments_raw["segment_name"] = buyer_segments_raw["cluster"].map(segment_names)
cluster_profile["segment_name"] = cluster_profile["cluster"].map(segment_names)

# Base Color Palette (Professional/Restrained)
COLOR_PRIMARY = "#3182ce"
COLOR_MUTED = "#718096"
COLOR_HIGHLIGHT = "#2b6cb0"
COLOR_TEAL = "#319795"

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def format_lakhs(val):
    if val >= 100000:
        return f"₹{val/100000:.2f}L"
    return f"₹{val:,.0f}"

def format_perc(val):
    return f"{val:.1f}%" if pd.notnull(val) else "0.0%"

def get_plotly_layout(fig):
    """Applies a universal transparent theme layout that perfectly inherits Streamlit Light/Dark modes."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=80, t=30, b=10),
        font=dict(family="Inter, Segoe UI, Roboto, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=dict(showgrid=False)
    )
    fig.update_traces(cliponaxis=False)
    return fig

def render_metric_card(title, value, subtitle=""):
    return f"""
    <div style="background-color: var(--secondary-background-color); border: 1px solid var(--border-color); border-radius: 6px; padding: 1.25rem; height: 100%;">
        <div style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; margin-bottom: 0.25rem; color: var(--text-color);">{title}</div>
        <div style="font-size: 1.8rem; font-weight: 600; color: var(--text-color); line-height: 1.2; margin-bottom: 0.4rem;">{value}</div>
        <div style="font-size: 0.8rem; font-weight: 500; color: var(--text-color); opacity: 0.6;">{subtitle}</div>
    </div>
    """

# ==========================================
# 4. SIDEBAR NAVIGATION & FILTERS
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='margin-top: 0 !important;'>REAL ESTATE<br>INTELLIGENCE</h2>", unsafe_allow_html=True)
    st.markdown("Buyer Segmentation<br>& Investment Profiling", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<h3>NAVIGATION</h3>", unsafe_allow_html=True)
    view_mode = st.radio(
        "Navigation",
        ["Overview", "Segmentation", "Investment", "Behavior", "Cluster Explorer"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>FILTERS</h3>", unsafe_allow_html=True)
    
    acq_purpose = st.selectbox("Acquisition Purpose", ["All", "Investment", "Home"])
    client_type = st.selectbox("Client Type", ["All", "Corporate", "Individual"])
    
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.8rem; opacity: 0.7;'>
        <strong>DATASET</strong><br>
        2,000 buyer profiles<br>
        7,305 sold transactions<br>
        8 ML segments
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 5. DYNAMIC DATA AGGREGATION
# ==========================================
buyer_segments = buyer_segments_raw.copy()

if acq_purpose == "Investment":
    buyer_segments = buyer_segments[buyer_segments['investment_flag'] == 1]
elif acq_purpose == "Home":
    buyer_segments = buyer_segments[buyer_segments['investment_flag'] == 0]

if client_type == "Corporate":
    buyer_segments = buyer_segments[buyer_segments['corporate_flag'] == 1]
elif client_type == "Individual":
    buyer_segments = buyer_segments[buyer_segments['corporate_flag'] == 0]

# Dynamic profile recalcs based on filtered subset
dynamic_profile = buyer_segments.groupby('cluster').agg(
    buyers=('client_id', 'count'),
    avg_age=('age', 'mean'),
    avg_purchases=('purchase_count', 'mean'),
    avg_total_value=('total_purchase_value', 'mean'),
    avg_purchase_value=('avg_purchase_value', 'mean'),
    avg_floor_area=('avg_floor_area', 'mean'),
    investment_rate=('investment_flag', lambda x: x.mean() * 100),
    loan_rate=('loan_flag', lambda x: x.mean() * 100),
    apartment_share=('apartment_share', lambda x: x.mean() * 100),
    office_share=('office_share', lambda x: x.mean() * 100),
    corporate_rate=('corporate_flag', lambda x: x.mean() * 100)
).reset_index()

if dynamic_profile.empty:
    st.warning("No data matches the selected filters. Please adjust the sidebar controls.")
    st.stop()
else:
    dynamic_profile['segment_name'] = dynamic_profile['cluster'].map(segment_names)

# Global KPI vars
global_buyers = len(buyer_segments)
global_segments = buyer_segments["cluster"].nunique()
global_inv_buyers = int(buyer_segments["investment_flag"].sum())
global_inv_rate = (global_inv_buyers / global_buyers * 100) if global_buyers > 0 else 0

# ==========================================
# 6. ROUTER VIEWS
# ==========================================

# ----------------- OVERVIEW -----------------
if view_mode == "Overview":
    st.markdown("<h1>REAL ESTATE BUYER INTELLIGENCE</h1>", unsafe_allow_html=True)
    st.markdown("<h3>Analyzing buyer behavior, property preferences and investment patterns using machine-learning-based segmentation.</h3>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(render_metric_card("TOTAL BUYERS", f"{global_buyers:,}", "Buyer profiles analyzed"), unsafe_allow_html=True)
    with k2: st.markdown(render_metric_card("BUYER SEGMENTS", global_segments, "Machine-learning segments"), unsafe_allow_html=True)
    with k3: st.markdown(render_metric_card("INVESTMENT BUYERS", f"{global_inv_buyers:,}", "Investment acquisition profiles"), unsafe_allow_html=True)
    with k4: st.markdown(render_metric_card("INVESTMENT RATE", f"{global_inv_rate:.2f}%", "Investment-purpose buyers"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2>KEY DATA INSIGHTS</h2>", unsafe_allow_html=True)
    
    largest_b = dynamic_profile.loc[dynamic_profile['buyers'].idxmax()]
    highest_val = dynamic_profile.loc[dynamic_profile['avg_total_value'].idxmax()]
    highest_inv = dynamic_profile.loc[dynamic_profile['investment_rate'].idxmax()]
    highest_freq = dynamic_profile.loc[dynamic_profile['avg_purchases'].idxmax()]
    highest_corp = dynamic_profile.loc[dynamic_profile['corporate_rate'].idxmax()]
    
    i1, i2 = st.columns(2)
    with i1:
        st.info(f"**Largest buyer segment**\n\n{largest_b['segment_name']} — {largest_b['buyers']} buyers")
        st.info(f"**Highest investment rate**\n\n{highest_inv['segment_name']} — {format_perc(highest_inv['investment_rate'])}")
        st.info(f"**Highest purchase frequency**\n\n{highest_freq['segment_name']} — {highest_freq['avg_purchases']:.2f} transactions")
    with i2:
        st.info(f"**Highest average total acquisition value**\n\n{highest_val['segment_name']} — {format_lakhs(highest_val['avg_total_value'])}")
        st.info(f"**Highest corporate rate**\n\n{highest_corp['segment_name']} — {format_perc(highest_corp['corporate_rate'])}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2>HOW THE SEGMENTATION WORKS</h2>", unsafe_allow_html=True)
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        st.markdown("<div style='text-align:center; padding:10px; border:1px solid var(--border-color); border-radius:6px; opacity:0.85;'><span style='font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Data</span><br><b>7,305</b> transactions</div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div style='text-align:center; padding:10px; border:1px solid var(--border-color); border-radius:6px; opacity:0.85;'><span style='font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Buyer Features</span><br><b>11</b> behavioral stats</div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div style='text-align:center; padding:10px; border:1px solid var(--border-color); border-radius:6px; opacity:0.85;'><span style='font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Feature Scaling</span><br>StandardScaler</div>", unsafe_allow_html=True)
    with f4:
        st.markdown("<div style='text-align:center; padding:10px; border:1px solid var(--border-color); border-radius:6px; opacity:0.85;'><span style='font-size:0.75rem; font-weight:600; text-transform:uppercase;'>K-Means</span><br><b>8</b> buyer segments</div>", unsafe_allow_html=True)
    with f5:
        st.markdown("<div style='text-align:center; padding:10px; border:1px solid var(--border-color); border-radius:6px; opacity:0.85;'><span style='font-size:0.75rem; font-weight:600; text-transform:uppercase;'>Profiling</span><br>Yield & property insights</div>", unsafe_allow_html=True)


# ----------------- SEGMENTATION -----------------
elif view_mode == "Segmentation":
    st.markdown("<h2>BUYER SEGMENTATION OVERVIEW</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Buyer Distribution</p>", unsafe_allow_html=True)
        dist_df = dynamic_profile[['segment_name', 'buyers']].copy()
        dist_df['Percentage'] = (dist_df['buyers'] / global_buyers) * 100
        dist_df = dist_df.sort_values('buyers', ascending=True)
        
        fig_dist = px.bar(
            dist_df, x='buyers', y='segment_name', 
            text=dist_df.apply(lambda row: f"{row['buyers']} ({row['Percentage']:.1f}%)", axis=1)
        )
        fig_dist.update_traces(marker_color=COLOR_PRIMARY, textposition='outside')
        fig_dist.update_layout(xaxis_title="Buyer Count", yaxis_title="", xaxis_range=[0, dist_df['buyers'].max() * 1.3])
        st.plotly_chart(get_plotly_layout(fig_dist), use_container_width=True, theme="streamlit")
        
    with c2:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Property Preference By Segment</p>", unsafe_allow_html=True)
        prop_df = dynamic_profile[['segment_name', 'apartment_share', 'office_share']].sort_values('apartment_share', ascending=True)
        
        fig_prop = go.Figure()
        fig_prop.add_trace(go.Bar(y=prop_df['segment_name'], x=prop_df['apartment_share'], name='Apartment Share', orientation='h', marker_color=COLOR_PRIMARY))
        fig_prop.add_trace(go.Bar(y=prop_df['segment_name'], x=prop_df['office_share'], name='Office Share', orientation='h', marker_color=COLOR_MUTED))
        fig_prop.update_layout(
            barmode='stack', yaxis_title="", xaxis_title="Percentage (%)", 
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(get_plotly_layout(fig_prop), use_container_width=True, theme="streamlit")


# ----------------- INVESTMENT -----------------
elif view_mode == "Investment":
    st.markdown("<h2>INVESTMENT INTELLIGENCE</h2>", unsafe_allow_html=True)
    
    i1, i2, i3 = st.columns(3)
    with i1: st.metric("INVESTMENT BUYERS", f"{global_inv_buyers:,}")
    with i2: st.metric("NON-INVESTMENT BUYERS", f"{(global_buyers - global_inv_buyers):,}")
    with i3: st.metric("OVERALL INVESTMENT RATE", f"{global_inv_rate:.2f}%")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Investment Rate By Segment</p>", unsafe_allow_html=True)
        ir_df = dynamic_profile[['segment_name', 'investment_rate']].sort_values('investment_rate', ascending=True)
        fig_ir = px.bar(ir_df, x='investment_rate', y='segment_name', text='investment_rate')
        fig_ir.update_traces(marker_color=COLOR_TEAL, texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
        fig_ir.update_layout(xaxis_title="Investment Rate (%)", yaxis_title="", xaxis_range=[0, 125])
        st.plotly_chart(get_plotly_layout(fig_ir), use_container_width=True, theme="streamlit")
        
    with c2:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Financing Behavior (Loan Rate)</p>", unsafe_allow_html=True)
        lr_df = dynamic_profile[['segment_name', 'loan_rate']].sort_values('loan_rate', ascending=True)
        fig_lr = px.bar(lr_df, x='loan_rate', y='segment_name', text='loan_rate')
        fig_lr.update_traces(marker_color=COLOR_MUTED, texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
        fig_lr.update_layout(xaxis_title="Loan Rate (%)", yaxis_title="", xaxis_range=[0, 125])
        st.plotly_chart(get_plotly_layout(fig_lr), use_container_width=True, theme="streamlit")

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    
    with c3:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Buyer Type (Corporate vs Individual)</p>", unsafe_allow_html=True)
        cr_df = dynamic_profile[['segment_name', 'corporate_rate']].sort_values('corporate_rate', ascending=True)
        fig_cr = px.bar(cr_df, x='corporate_rate', y='segment_name', text='corporate_rate')
        fig_cr.update_traces(marker_color=COLOR_PRIMARY, texttemplate='%{text:.1f}%', textposition='outside', cliponaxis=False)
        fig_cr.update_layout(xaxis_title="Corporate Rate (%)", yaxis_title="", xaxis_range=[0, 125])
        st.plotly_chart(get_plotly_layout(fig_cr), use_container_width=True, theme="streamlit")


# ----------------- BEHAVIOR -----------------
elif view_mode == "Behavior":
    st.markdown("<h2>BEHAVIORAL TRENDS</h2>", unsafe_allow_html=True)
    
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Average Total Acquisition Value</p>", unsafe_allow_html=True)
        df_tot = dynamic_profile[['segment_name', 'avg_total_value']].sort_values('avg_total_value', ascending=True)
        fig_tot = px.bar(df_tot, x='avg_total_value', y='segment_name', text='avg_total_value')
        fig_tot.update_traces(marker_color=COLOR_PRIMARY, texttemplate='₹%{text:,.0f}', textposition='outside')
        fig_tot.update_layout(xaxis_title="Average Total Value (₹)", yaxis_title="", xaxis_range=[0, df_tot['avg_total_value'].max() * 1.3])
        st.plotly_chart(get_plotly_layout(fig_tot), use_container_width=True, theme="streamlit")

    with row1_c2:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Average Purchase Frequency</p>", unsafe_allow_html=True)
        df_freq = dynamic_profile[['segment_name', 'avg_purchases']].sort_values('avg_purchases', ascending=True)
        fig_freq = px.bar(df_freq, x='avg_purchases', y='segment_name', text='avg_purchases')
        fig_freq.update_traces(marker_color=COLOR_TEAL, texttemplate='%{text:.2f}', textposition='outside')
        fig_freq.update_layout(xaxis_title="Standard Purchases", yaxis_title="", xaxis_range=[0, df_freq['avg_purchases'].max() * 1.3])
        st.plotly_chart(get_plotly_layout(fig_freq), use_container_width=True, theme="streamlit")

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Average Purchase Value</p>", unsafe_allow_html=True)
        df_pv = dynamic_profile[['segment_name', 'avg_purchase_value']].sort_values('avg_purchase_value', ascending=True)
        fig_pv = px.bar(df_pv, x='avg_purchase_value', y='segment_name', text='avg_purchase_value')
        fig_pv.update_traces(marker_color=COLOR_MUTED, texttemplate='₹%{text:,.0f}', textposition='outside')
        fig_pv.update_layout(xaxis_title="Average Purchase Value (₹)", yaxis_title="", xaxis_range=[0, df_pv['avg_purchase_value'].max() * 1.3])
        st.plotly_chart(get_plotly_layout(fig_pv), use_container_width=True, theme="streamlit")

    with row2_c2:
        st.markdown("<p style='font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.8;'>Average Floor Area</p>", unsafe_allow_html=True)
        df_area = dynamic_profile[['segment_name', 'avg_floor_area']].sort_values('avg_floor_area', ascending=True)
        fig_area = px.bar(df_area, x='avg_floor_area', y='segment_name', text='avg_floor_area')
        fig_area.update_traces(marker_color=COLOR_HIGHLIGHT, texttemplate='%{text:,.0f} sq ft', textposition='outside')
        fig_area.update_layout(xaxis_title="Square Footage", yaxis_title="", xaxis_range=[0, df_area['avg_floor_area'].max() * 1.3])
        st.plotly_chart(get_plotly_layout(fig_area), use_container_width=True, theme="streamlit")


# ----------------- CLUSTER EXPLORER -----------------
elif view_mode == "Cluster Explorer":
    st.markdown("<h2>CLUSTER EXPLORER</h2>", unsafe_allow_html=True)
    
    mode_options = ["All Segments"] + [f"{name} (Cluster {c_id})" for c_id, name in segment_names.items() if c_id in dynamic_profile['cluster'].values]
    selected_explore = st.selectbox("Select View", mode_options, label_visibility="collapsed")
    
    if selected_explore == "All Segments":
        
        st.markdown("<br><h3>SEGMENT COMPARISON</h3>", unsafe_allow_html=True)
        comp_col, chart_col = st.columns([1, 3])
        with comp_col:
            metric_map = {
                "Average Total Purchase Value": "avg_total_value",
                "Average Purchase Value": "avg_purchase_value",
                "Average Purchases": "avg_purchases",
                "Average Floor Area": "avg_floor_area",
                "Investment Rate": "investment_rate",
                "Loan Rate": "loan_rate",
                "Corporate Rate": "corporate_rate"
            }
            selected_metric_lbl = st.radio("Interactive Comparison Metric", list(metric_map.keys()))
            selected_metric = metric_map[selected_metric_lbl]
        with chart_col:
            comp_df = dynamic_profile[['segment_name', selected_metric]].sort_values(selected_metric, ascending=True)
            figc = px.bar(comp_df, x=selected_metric, y='segment_name')
            figc.update_traces(marker_color=COLOR_PRIMARY)
            figc.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(get_plotly_layout(figc), use_container_width=True, theme="streamlit")
        
        st.markdown("---")
        st.markdown("<h3>FULL CLUSTER PROFILE TABLE</h3>", unsafe_allow_html=True)
        
        styled_df = dynamic_profile.rename(columns={
            'cluster': 'Cluster', 'segment_name': 'Segment', 'buyers': 'Buyers',
            'avg_age': 'Avg Age', 'avg_purchases': 'Avg Purchases',
            'avg_purchase_value': 'Avg Purchase Value', 'avg_total_value': 'Avg Total Value',
            'avg_floor_area': 'Avg Floor Area', 'investment_rate': 'Investment Rate',
            'loan_rate': 'Loan Rate', 'apartment_share': 'Apartment Share',
            'office_share': 'Office Share', 'corporate_rate': 'Corporate Rate'
        }).style.format({
            'Avg Purchase Value': format_lakhs, 'Avg Total Value': format_lakhs,
            'Avg Floor Area': '{:,.0f} sq ft', 'Investment Rate': format_perc,
            'Loan Rate': format_perc, 'Apartment Share': format_perc,
            'Office Share': format_perc, 'Corporate Rate': format_perc,
            'Avg Age': '{:.1f}', 'Avg Purchases': '{:.2f}'
        })
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    else:
        # Specific cluster selected
        c_id = int(selected_explore.split("(Cluster ")[1].replace(")", "").strip())
        profile = dynamic_profile[dynamic_profile['cluster'] == c_id].iloc[0]
        
        st.markdown(f"<br><h2>{profile['segment_name'].upper()}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='opacity: 0.7;'>Cluster {c_id} &nbsp;|&nbsp; {int(profile['buyers']):,} buyers</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("INVESTMENT RATE", format_perc(profile['investment_rate']))
        with col2: st.metric("LOAN RATE", format_perc(profile['loan_rate']))
        with col3: st.metric("APARTMENT SHARE", format_perc(profile['apartment_share']))
        with col4: st.metric("CORPORATE RATE", format_perc(profile['corporate_rate']))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col5, col6, col7 = st.columns(3)
        with col5: st.metric("AVERAGE PURCHASE VALUE", format_lakhs(profile['avg_purchase_value']))
        with col6: st.metric("AVERAGE TOTAL PURCHASE VALUE", format_lakhs(profile['avg_total_value']))
        with col7: st.metric("AVERAGE FLOOR AREA", f"{profile['avg_floor_area']:,.0f} sq ft")


# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    REAL ESTATE BUYER INTELLIGENCE<br>
    Machine Learning &bull; Python &bull; Pandas &bull; Scikit-learn &bull; Plotly &bull; Streamlit<br>
    Buyer Segmentation & Investment Profiling
</div>
""", unsafe_allow_html=True)