
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import requests
from datetime import datetime
import time

from utils import  load_model_formulaire

# Configuration de la page
st.set_page_config(
    page_title="🎓 AI Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CSS Styling Avancé ----
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .stApp {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh;
        }
        
        .main-container {
            background: rgba(15, 23, 42, 0.9);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem;
            box-shadow: 0 25px 50px rgba(0,0,0,0.3);
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .hero-title {
            background: linear-gradient(135deg, #06b6d4, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .hero-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #1e293b, #334155);
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(6, 182, 212, 0.5);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #06b6d4, #0891b2);
            color: white;
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            margin: 0.5rem 0;
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3);
        }
        
        .success-card {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            animation: pulse 2s infinite;
            box-shadow: 0 15px 35px rgba(16, 185, 129, 0.3);
        }
        
        .warning-card {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            animation: pulse 2s infinite;
            box-shadow: 0 15px 35px rgba(245, 158, 11, 0.3);
        }
        
        .danger-card {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            animation: pulse 2s infinite;
            box-shadow: 0 15px 35px rgba(239, 68, 68, 0.3);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .progress-container {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        .recommendation-box {
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            color: white;
            border-radius: 20px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 12px 30px rgba(139, 92, 246, 0.3);
            border: 1px solid rgba(139, 92, 246, 0.4);
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #f1f5f9;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Styling pour les éléments Streamlit */
        .stSelectbox > div > div {
            background-color: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 10px !important;
            color: #f1f5f9 !important;
        }
        
        .stSelectbox label {
            color: #e2e8f0 !important;
            font-weight: 500 !important;
        }
        
        .stSlider > div > div > div {
            background-color: rgba(6, 182, 212, 0.3) !important;
        }
        
        .stSlider label {
            color: #e2e8f0 !important;
            font-weight: 500 !important;
        }
        
        .stNumberInput > div > div > input {
            background-color: rgba(30, 41, 59, 0.8) !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 10px !important;
            color: #f1f5f9 !important;
        }
        
        .stNumberInput label {
            color: #e2e8f0 !important;
            font-weight: 500 !important;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background-color: rgba(30, 41, 59, 0.6) !important;
            border-radius: 15px !important;
            padding: 5px !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            border-radius: 10px !important;
            color: #94a3b8 !important;
            font-weight: 500 !important;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(6, 182, 212, 0.2) !important;
            color: #06b6d4 !important;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #06b6d4, #8b5cf6) !important;
            color: white !important;
            border: none !important;
            border-radius: 15px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 35px rgba(6, 182, 212, 0.4) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialisation de l'état de session
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Header avec animation
st.markdown("""
    <div class="main-container">
        <h1 class="hero-title">🎓 AI Student Success Predictor</h1>
        <p class="hero-subtitle">Prédiction intelligente de la réussite étudiante avec IA avancée</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar avec statistiques
with st.sidebar:
    st.header("📊 Dashboard")
    
    # Statistiques de session
    st.subheader("Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prédictions", len(st.session_state.prediction_history))
    with col2:
        if st.session_state.prediction_history:
            success_rate = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 1) / len(st.session_state.prediction_history) * 100
            st.metric("Taux Réussite", f"{success_rate:.1f}%")
    
    # Navigation
    
    page = "🏠 Prédiction"
    
    # Réinitialiser
    if st.button("🔄 Réinitialiser", use_container_width=True):
        st.session_state.prediction_history = []
        st.session_state.current_prediction = None
        st.rerun()

# Page principale selon la navigation
if page == "🏠 Prédiction":
    
    # Formulaire interactif
    with st.container():
        st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
        st.subheader("📝 Informations de l'étudiant")
        st.markdown("""
            <style>
                .stSubheader {
                    color: #e2e8f0 !important;
                    font-size: 1.5rem !important;
                    font-weight: 600 !important;
                    margin-bottom: 1.5rem !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
        # Progress bar pour le formulaire
        progress_container = st.container()
        
        # Onglets pour organiser les informations
        tab1, tab2, tab3 = st.tabs(["👤 Profil Personnel", "📚 Performance Académique", "💰 Situation Financière"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                marital_status = st.selectbox(
                    "👫 Statut marital", 
                    options=[0, 1, 2, 3, 4, 5],
                    format_func=lambda x: ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve", "Union libre", "Autre"][x],
                    help="Sélectionnez votre statut marital actuel"
                )
                
                gender = st.selectbox(
                    "👤 Genre", 
                    options=[0, 1],
                    format_func=lambda x: "Homme" if x == 0 else "Femme",
                    help="Sélectionnez votre genre"
                )
                
                displaced = st.selectbox(
                    "🏠 Statut de déplacement", 
                    options=[0, 1],
                    format_func=lambda x: "Non déplacé" if x == 0 else "Déplacé",
                    help="Êtes-vous un étudiant déplacé ?"
                )
            
            with col2:
                enrollment_age = st.slider(
                    "🎂 Âge à l'inscription", 
                    min_value=15, max_value=100, value=20,
                    help="Quel était votre âge lors de l'inscription ?"
                )
                
                previous_qualification = st.selectbox(
                    "🎓 Qualification précédente", 
                    options=[0, 1, 2, 3, 4],
                    format_func=lambda x: ["Aucune", "Certificat", "Diplôme", "Licence", "Master"][x],
                    help="Votre plus haut niveau d'études avant cette formation"
                )
        
        with tab2:
            st.markdown("""
                <h3 style='color: #e2e8f0; font-size: 1.3rem; margin-bottom: 1rem;'>
                    📚 Semestre 1
                </h3>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                course_sem1_enrolled = st.number_input("📚 Cours inscrits S1", min_value=0, max_value=20, value=6)
            with col2:
                course_sem1_passed = st.number_input("✅ Cours validés S1", min_value=0, max_value=20, value=4)
            with col3:
                course_sem1_grade = st.slider("📊 Note moyenne S1", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            
            # Progression S1
            if course_sem1_enrolled > 0:
                progress_s1 = (course_sem1_passed / course_sem1_enrolled) * 100
                st.progress(progress_s1 / 100)
                st.markdown(f"""
                    <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;'>
                        Taux de réussite S1: {progress_s1:.1f}%
                    </p>
                """, unsafe_allow_html=True)
            
            st.markdown("""
                <h3 style='color: #e2e8f0; font-size: 1.3rem; margin: 2rem 0 1rem 0;'>
                    📚 Semestre 2
                </h3>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                course_sem2_enrolled = st.number_input("📚 Cours inscrits S2", min_value=0, max_value=20, value=6)
            with col2:
                course_sem2_passed = st.number_input("✅ Cours validés S2", min_value=0, max_value=20, value=4)
            with col3:
                course_sem2_grade = st.slider("📊 Note moyenne S2", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
            
            # Progression S2
            if course_sem2_enrolled > 0:
                progress_s2 = (course_sem2_passed / course_sem2_enrolled) * 100
                st.progress(progress_s2 / 100)
                st.markdown(f"""
                    <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;'>
                        Taux de réussite S2: {progress_s2:.1f}%
                    </p>
                """, unsafe_allow_html=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                tuition_paid = st.selectbox(
                    "💳 Frais de scolarité payés", 
                    options=[0, 1],
                    format_func=lambda x: "❌ Non payés" if x == 0 else "✅ Payés",
                    help="Les frais de scolarité ont-ils été payés ?"
                )
                
                has_scholarship = st.selectbox(
                    "🎓 Bourse d'études", 
                    options=[0, 1],
                    format_func=lambda x: "❌ Pas de bourse" if x == 0 else "✅ Avec bourse",
                    help="Bénéficiez-vous d'une bourse d'études ?"
                )
            
            with col2:
                debtor = st.selectbox(
                    "💸 Statut d'endettement", 
                    options=[0, 1],
                    format_func=lambda x: "✅ Pas de dettes" if x == 0 else "⚠️ Endetté",
                    help="Avez-vous des dettes liées aux études ?"
                )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Validation et prédiction
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔮 Prédire la Réussite", 
                use_container_width=True,
                type="primary"
            )

# Section Résultats
if predict_button or st.session_state.current_prediction:
    if predict_button:
       with st.spinner("🤖 IA en cours d'analyse..."):
        # Préparation des données pour le modèle
        input_data = {
                "Marital_status" : [marital_status],
                "Previous_qualification": [previous_qualification],
                "Displaced": [displaced],
                "Debtor": [debtor],
                "tuition_paid": [tuition_paid],
                "Gender": [gender],
                "has_scholarship": [has_scholarship],
                "enrollment_age": [enrollment_age],
                "course_sem1_enrolled": [course_sem1_enrolled],
                "course_sem1_passed": [course_sem1_passed],
                "course_sem1_grade" : [course_sem1_grade],
                "course_sem2_enrolled" :[ course_sem2_enrolled ],
                "course_sem2_passed" :[course_sem2_passed ],
                "course_sem2_grade" : [course_sem2_grade],
        }
        model = load_model_formulaire()
        # Conversion en DataFrame
        input = pd.DataFrame(input_data)
        # input_df = scaler.transform(input)
        
       
            # Prédiction avec le modèle Random Forest
        prediction = model.predict(input)[0]
        proba = model.predict_proba(input)[0]
        print(prediction)

        confidence = max(proba) * 100  # Probabilité de la classe prédite
 
        prediction_data = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'data': input_data
        }
        
        st.session_state.current_prediction = prediction_data
        st.session_state.prediction_history.append(prediction_data)

    if st.session_state.current_prediction:
        pred = st.session_state.current_prediction
        
        st.markdown("---")
        st.subheader("🎯 Résultats de la Prédiction")
        
        # Résultat principal
        if pred['prediction'] == 1:
            st.markdown(f"""
                <div class="success-card">
                    <h2>🎓 RÉUSSITE PRÉDITE</h2>
                    <p style="font-size: 1.2rem;">L'étudiant a de fortes chances de réussir !</p>
                    <p style="font-size: 2rem; font-weight: bold;">{pred['confidence']}% de confiance</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if pred['confidence'] > 70:
                st.markdown(f"""
                    <div class="danger-card">
                        <h2>⚠️ ÉCHEC PROBABLE</h2>
                        <p style="font-size: 1.2rem;">L'étudiant risque d'échouer</p>
                        <p style="font-size: 2rem; font-weight: bold;">{pred['confidence']}% de confiance</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="warning-card">
                        <h2>⚡ RÉSULTATS INCERTAINS</h2>
                        <p style="font-size: 1.2rem;">Situation à surveiller de près</p>
                        <p style="font-size: 2rem; font-weight: bold;">{pred['confidence']}% de confiance</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Gauge de confiance
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = pred['confidence'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Niveau de Confiance"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Analyse détaillée
        st.subheader("🔍 Analyse Détaillée")
        
        col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown("#### ✅ Facteurs Positifs")
        #     positive_factors = []
        #     if pred['has_scholarship'] == 1:
        #         positive_factors.append("• Bénéficie d'une bourse d'études")
        #     if pred['tuition_paid'] == 1:
        #         positive_factors.append("• Frais de scolarité payés")
        #     if pred['Debtor'] == 0:
        #         positive_factors.append("• Aucune dette")
        #     if pred['course_sem1_grade'] >= 12:
        #         positive_factors.append(f"• Bonne note S1 ({pred['course_sem1_grade']:.1f}/20)")
        #     if pred['course_sem2_grade'] >= 12:
        #         positive_factors.append(f"• Bonne note S2 ({pred['course_sem2_grade']:.1f}/20)")
            
        #     for factor in positive_factors:
        #         st.markdown(factor)
        
        # with col2:
        #     st.markdown("#### ⚠️ Points d'Attention")
        #     warning_factors = []
        #     if pred['Debtor'] == 1:
        #         warning_factors.append("• Situation d'endettement")
        #     if pred['tuition_paid'] == 0:
        #         warning_factors.append("• Frais de scolarité impayés")
        #     if pred['course_sem1_grade'] < 10:
        #         warning_factors.append(f"• Note S1 faible ({pred['course_sem1_grade']:.1f}/20)")
        #     if pred['course_sem2_grade'] < 10:
        #         warning_factors.append(f"• Note S2 faible ({pred['course_sem2_grade']:.1f}/20)")
        with col1:
              st.markdown("#### ✅ Facteurs Positifs")
              positive_factors = []
              if pred['data']['has_scholarship'][0] == 1:  # Accès via pred['data']
                  positive_factors.append("• Bénéficie d'une bourse d'études")
              if pred['data']['tuition_paid'][0] == 1:  # Accès via pred['data']
                  positive_factors.append("• Frais de scolarité payés")
              if pred['data']['Debtor'][0] == 0:  # Accès via pred['data']
                  positive_factors.append("• Aucune dette")
              if pred['data']['course_sem1_grade'][0] >= 12:  # Accès via pred['data']
                  positive_factors.append(f"• Bonne note S1 ({pred['data']['course_sem1_grade'][0]:.1f}/20)")
              if pred['data']['course_sem2_grade'][0] >= 12:  # Accès via pred['data']
                  positive_factors.append(f"• Bonne note S2 ({pred['data']['course_sem2_grade'][0]:.1f}/20)")
    
              for factor in positive_factors:
                  st.markdown(factor)

        with col2:
              st.markdown("#### ⚠️ Points d'Attention")
              warning_factors = []
              if pred['data']['Debtor'][0] == 1:  # Accès via pred['data']
                  warning_factors.append("• Situation d'endettement")
              if pred['data']['tuition_paid'][0] == 0:  # Accès via pred['data']
                  warning_factors.append("• Frais de scolarité impayés")
              if pred['data']['course_sem1_grade'][0] < 10:  # Accès via pred['data']
                  warning_factors.append(f"• Note S1 faible ({pred['data']['course_sem1_grade'][0]:.1f}/20)")
              if pred['data']['course_sem2_grade'][0] < 10:  # Accès via pred['data']
                  warning_factors.append(f"• Note S2 faible ({pred['data']['course_sem2_grade'][0]:.1f}/20)")
              
              for factor in warning_factors:
                  st.markdown(factor)
              
              if not warning_factors:
                  st.success("Aucun point d'attention majeur détecté !")
        
        for factor in warning_factors:
            st.markdown(factor)
            
        if not warning_factors:
            st.success("Aucun point d'attention majeur détecté !")
        
        # Recommandations
        st.markdown("""
            <div class="recommendation-box">
                <h3>💡 Recommandations Personnalisées</h3>
        """, unsafe_allow_html=True)
        
        if pred['prediction'] == 1:
            st.markdown("""
                • Continuez sur cette excellente voie !<br>
                • Maintenez votre rythme de travail<br>
                • Participez à des activités extra-scolaires<br>
                • Préparez-vous pour les opportunités futures
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                • Consultez un conseiller pédagogique<br>
                • Organisez des séances de tutorat<br>
                • Établissez un planning de révision<br>
                • Cherchez un soutien financier si nécessaire
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# elif page == "📈 Analyse":
#     st.subheader("📈 Analyse des Tendances")
    
#     if st.session_state.prediction_history:
#         # Graphiques d'analyse
#         df_history = pd.DataFrame([{
#             'Date': p['timestamp'].strftime('%Y-%m-%d %H:%M'),
#             'Prédiction': 'Réussite' if p['prediction'] == 1 else 'Échec',
#             'Confiance': p['confidence'],
#             'Note_S1': p['data']['course_sem1_grade'],
#             'Note_S2': p['data']['course_sem2_grade'],
#             'Bourse': 'Oui' if p['data']['has_scholarship'] == 1 else 'Non'
#         } for p in st.session_state.prediction_history])
        
#         # Graphique des prédictions
#         fig_pred = px.pie(df_history, names='Prédiction', title="Répartition des Prédictions")
#         st.plotly_chart(fig_pred, use_container_width=True)
        
#         # Relation notes vs prédiction
#         fig_notes = px.scatter(df_history, x='Note_S1', y='Note_S2', 
#                               color='Prédiction', size='Confiance',
#                               title="Relation Notes S1/S2 vs Prédictions")
#         st.plotly_chart(fig_notes, use_container_width=True)
        
#     else:
#         st.info("Effectuez quelques prédictions pour voir les analyses !")

# elif page == "📋 Historique":
#     st.subheader("📋 Historique des Prédictions")
    
#     if st.session_state.prediction_history:
#         for i, pred in enumerate(reversed(st.session_state.prediction_history)):
#             with st.expander(f"Prédiction #{len(st.session_state.prediction_history)-i} - {pred['timestamp'].strftime('%d/%m/%Y %H:%M')}"):
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     result = "✅ Réussite" if pred['prediction'] == 1 else "❌ Échec"
#                     st.markdown(f"**Résultat:** {result}")
#                     st.markdown(f"**Confiance:** {pred['confidence']}%")
#                 with col2:
#                     st.markdown(f"**Notes:** S1={pred['data']['course_sem1_grade']:.1f}, S2={pred['data']['course_sem2_grade']:.1f}")
#                     st.markdown(f"**Bourse:** {'Oui' if pred['data']['has_scholarship'] else 'Non'}")
#     else:
#         st.info("Aucune prédiction dans l'historique.")

elif page == "ℹ️ À propos":
    st.subheader("ℹ️ À propos de l'Application")
    
    st.markdown("""
    ### 🎯 Objectif
    Cette application utilise l'intelligence artificielle pour prédire les chances de réussite d'un étudiant 
    basé sur différents facteurs académiques, financiers et personnels.
    
    ### 🧠 Fonctionnalités
    - **Prédiction IA**: Algorithme de machine learning avancé
    - **Analyse en temps réel**: Résultats instantanés
    - **Recommandations personnalisées**: Conseils adaptés
    - **Historique des prédictions**: Suivi des analyses
    - **Visualisations interactives**: Graphiques dynamiques
    
    ### 📊 Facteurs Analysés
    - Performance académique (notes, cours validés)
    - Situation financière (bourses, dettes, frais)
    - Profil personnel (âge, statut marital, genre)
    - Contexte social (déplacement, qualifications)
    
    ### 🔒 Confidentialité
    Toutes les données sont traitées de manière confidentielle et ne sont pas stockées 
    de façon permanente.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <p style='color: #6b7280; font-size: 0.9rem;'>
            🎓 AI Student Success Predictor v2.0 | 
            Développé avec ❤️ par votre équipe IA | 
            © 2025 Tous droits réservés
        </p>
    </div>
""", unsafe_allow_html=True)