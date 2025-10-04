import streamlit as st

st.set_page_config(
    page_title="Prédiction Réussite Étudiants", 
    layout="wide",
    page_icon="🎓"
)

# En-tête principal avec style
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f4e79; font-size: 3rem; margin-bottom: 0.5rem;">
            🎓 Prédiction de Réussite Étudiante
        </h1>
        <p style="font-size: 1.2rem; color: #666; font-style: italic;">
            Intelligence Artificielle au service de l'éducation
        </p>
    </div>
""", unsafe_allow_html=True)

# Section de présentation du projet
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## 🎯 À propos du projet")
    
    st.markdown("""
    ### Objectif Principal
    Notre application utilise des algorithmes d'apprentissage automatique pour **prédire la probabilité de réussite académique** 
    des étudiants en analysant différents facteurs socio-économiques, académiques et personnels.
    
    ### Pourquoi c'est important ?
    - **Détection précoce** des étudiants à risque d'échec
    - **Intervention personnalisée** pour améliorer les chances de réussite
    - **Optimisation des ressources** éducatives et d'accompagnement
    - **Aide à la prise de décision** pour les établissements d'enseignement
    
    ### Comment ça fonctionne ?
    Notre modèle d'IA analyse des variables telles que :
    - Les résultats académiques antérieurs
    - Le contexte socio-économique familial
    - Les habitudes d'étude et de participation
    - Les facteurs démographiques
    - L'engagement scolaire et les absences
    """)

with col2:
    st.markdown("## 📈 Statistiques du modèle")
    
    # Métriques fictives pour illustrer
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric("Précision", "90.3%", "2.1%")
        st.metric("Étudiants analysés", "3800", "156")
    
    with col_metric2:
        st.metric("Rappel", "89%", "1.8%")
        st.metric("Interventions réussies", "340", "23")
    


# Section navigation
st.markdown("---")
st.markdown("## 🧭 Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Dashboard Analytique
    - Visualisation des données étudiantes
    - Analyse statistique approfondie
    - Tendances et patterns de réussite
    - Comparaisons par groupes démographiques
    """)
    if st.button("🔍 Accéder au Dashboard", type="primary", use_container_width=True):
        st.switch_page("pages/1_📊_Dashboard.py")

with col2:
    st.markdown("""
    ### 📝 Prédiction Individuelle
    - Formulaire de saisie des données
    - Prédiction en temps réel
    - Recommandations personnalisées
    - Facteurs d'influence détaillés
    """)
    if st.button("🎯 Faire une prédiction", type="primary", use_container_width=True):
        st.switch_page("pages/2_📝_Formulaire.py")

with col3:
    st.markdown("""
    ### 📚 Documentation
    - Guide d'utilisation
    - Méthodologie du modèle
    - Interprétation des résultats
    - FAQ et support
    """)
    if st.button("📖 Consulter la doc", type="secondary", use_container_width=True):
        st.info("Documentation en cours de développement")

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem 0; color: #666;">
    <p>💡 <strong>Astuce :</strong> Utilisez la barre latérale pour naviguer rapidement entre les sections</p>
    <p style="font-size: 0.9rem;">Développé avec ❤️ pour améliorer la réussite étudiante</p>
</div>
""", unsafe_allow_html=True)

# Message d'encouragement
st.success("🚀 Prêt à explorer les données et faire des prédictions ? Choisissez une section ci-dessus pour commencer !")
