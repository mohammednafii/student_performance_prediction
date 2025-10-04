from utils import load_database , load_model , load_scaler
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Performance Étudiants",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)
df = load_database()

# Fonction pour générer des données d'exemple
@st.cache_data
def generate_sample_data():
    data = {
        'Marital_status': df["Marital_status"].values, 
        'Previous_qualification': df["Previous_qualification"].values,
        'Displaced': df["Displaced"].values,
        'Debtor': df["Debtor"].values,
        'tuition_paid': df["tuition_paid"].values,
        'Gender': df["Gender"].values,
        'has_scholarship': df["has_scholarship"].values,
        'enrollment_age': df["enrollment_age"].values,
        'course_sem1_enrolled': df["course_sem1_enrolled"].values,
        'course_sem1_passed': df["course_sem1_passed"].values,
        'course_sem1_grade': df["course_sem1_grade"].values,
        'course_sem2_enrolled': df["course_sem2_enrolled"].values,
        'course_sem2_passed': df["course_sem2_passed"].values,
        'course_sem2_grade': df["course_sem2_grade"].values,
        'Cluster': df["Cluster"].values
    }
    
    # Ajuster course_sem1_passed pour qu'il soit <= course_sem1_enrolled
    data['course_sem1_passed'] = np.minimum(data['course_sem1_passed'], data['course_sem1_enrolled'])
    data['course_sem2_passed'] = np.minimum(data['course_sem2_passed'], data['course_sem2_enrolled'])
    
    data['Target'] = df["Target"]
    
    return pd.DataFrame(data)

# Fonction pour valider les colonnes du dataset
def validate_dataset(df):
    """Valide que le dataset contient les colonnes nécessaires"""
    required_columns = [
        'Gender', 'has_scholarship', 'Target', 'course_sem1_grade', 
        'course_sem2_grade', 'enrollment_age', 
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, missing_columns
    return True, []

# Fonction pour préprocesser les données uploadées
def preprocess_uploaded_data(df):
    """Nettoie et prépare les données uploadées pour l'analyse"""
    # Conversion des types si nécessaire
    numeric_columns = ['enrollment_age', 'course_sem1_grade', 'course_sem2_grade', 
                      'course_sem1_enrolled', 'course_sem2_enrolled', 
                      'course_sem1_passed', 'course_sem2_passed']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Gestion des valeurs manquantes
    df = df.dropna(subset=['Target', 'Gender', 'has_scholarship'])
    
    return df

# Fonction de chargement des données
@st.cache_data
def load_data(uploaded_file=None, use_sample=False):
    if uploaded_file is not None:
        try:
            # Lire le fichier CSV uploadé
            df = pd.read_csv(uploaded_file)
            
            # Valider le dataset
            is_valid, missing_cols = validate_dataset(df)
            
            if not is_valid:
                st.error(f"Le fichier CSV ne contient pas toutes les colonnes nécessaires. Colonnes manquantes: {missing_cols}")
                st.info("Utilisation des données d'exemple à la place.")
                return generate_sample_data(), False
            
            # Préprocesser les données
            df = preprocess_uploaded_data(df)
            
            return df, True
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier: {str(e)}")
            st.info("Utilisation des données d'exemple à la place.")
            return generate_sample_data(), False
    
    elif use_sample:
        return generate_sample_data(), True
    
    else:
        return None, False

def upload_interface():
    st.header("📁 Chargement des Données")
    
    # Options de chargement
    data_option = st.radio(
        "Choisissez votre source de données:",
        ["Utiliser les données d'exemple", "Faire des prédictions avec un modèle"],
        horizontal=True
    )
    
    uploaded_file = None
    use_sample = False
    predict_mode = False
    
    if data_option == "Télécharger un fichier CSV":
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir les colonnes: Gender, has_scholarship, Target, course_sem1_grade, course_sem2_grade, enrollment_age"
        )
        
        if uploaded_file is not None:
            # Afficher des informations sur le fichier
            st.success(f"Fichier '{uploaded_file.name}' chargé avec succès!")
            
            # Preview des données
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.write("**Aperçu des données (5 premières lignes):**")
                st.dataframe(preview_df.head(), use_container_width=True)
                
                st.write("**Informations sur le dataset:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre de lignes", len(preview_df))
                with col2:
                    st.metric("Nombre de colonnes", len(preview_df.columns))
                with col3:
                    st.metric("Taille du fichier", f"{uploaded_file.size / 1024:.1f} KB")
                
                # Reset file pointer for actual loading
                uploaded_file.seek(0)
                
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    
    elif data_option == "Utiliser les données d'exemple":
        use_sample = True
        st.info("Utilisation des données d'exemple générées automatiquement (3040 étudiants)")
        
        # Bouton pour télécharger un exemple de structure CSV
        if st.button("📥 Télécharger un exemple de structure CSV"):
            sample_df = generate_sample_data().head(10)
            csv_buffer = io.StringIO()
            sample_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="💾 Télécharger exemple.csv",
                data=csv_buffer.getvalue(),
                file_name="exemple_structure.csv",
                mime="text/csv"
            )
    
    elif data_option == "Faire des prédictions avec un modèle":
        predict_mode = True
        st.info("🤖 Mode prédiction : Téléchargez un fichier CSV pour obtenir des prédictions du modèle")
        
        # Afficher les caractéristiques requises
        st.write("**Le modèle attend les caractéristiques suivantes :**")
        feature_names = [
            "Marital_status", "Previous_qualification", "Displaced", "Debtor", 
            "tuition_paid", "Gender", "has_scholarship", "enrollment_age",
            "course_sem1_enrolled", "course_sem1_passed", "course_sem1_grade",
            "course_sem2_enrolled", "course_sem2_passed", "course_sem2_grade", "Cluster"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            for i in range(0, len(feature_names), 2):
                st.write(f"• {feature_names[i]}")
        with col2:
            for i in range(1, len(feature_names), 2):
                if i < len(feature_names):
                    st.write(f"• {feature_names[i]}")
        
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier CSV pour prédiction",
            type=['csv'],
            help="Le fichier doit contenir toutes les caractéristiques listées ci-dessus"
        )
        
        if uploaded_file is not None:
            try:
                preview_df = pd.read_csv(uploaded_file)
                st.success(f"Fichier '{uploaded_file.name}' chargé avec succès!")
                
                st.write("**Aperçu des données à prédire (5 premières lignes):**")
                st.dataframe(preview_df.head(), use_container_width=True)
                
                st.write("**Informations sur le dataset:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre d'étudiants", len(preview_df))
                with col2:
                    st.metric("Nombre de colonnes", len(preview_df.columns))
                with col3:
                    # Vérifier les colonnes manquantes
                    missing_features = [f for f in feature_names if f not in preview_df.columns]
                    st.metric("Colonnes manquantes", len(missing_features))
                
                if missing_features:
                    st.warning(f"⚠️ Colonnes manquantes: {', '.join(missing_features)}")
                else:
                    st.success("✅ Toutes les colonnes requises sont présentes")
                
                uploaded_file.seek(0)
                
            except Exception as e:
                st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
    
    return uploaded_file, use_sample, predict_mode

def make_predictions(df, model):
    """Preprocess data and make predictions"""
    try:
        # Liste des colonnes à utiliser
        feature_names = [
         "Marital_status", "Previous_qualification", "Displaced", "Debtor", 
         "tuition_paid", "Gender", "has_scholarship", "enrollment_age",
         "course_sem1_enrolled", "course_sem1_passed", "course_sem1_grade",
          "course_sem2_enrolled", "course_sem2_passed", "course_sem2_grade", "Cluster"
         ]

        # Vérifier que toutes les colonnes sont présentes
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            st.error(f"Colonnes manquantes pour la prédiction: {', '.join(missing_features)}")
            return None

        # Extraire les données sous forme de tableau NumPy
        features = df[feature_names].values

        # Transformer chaque ligne du tableau en dictionnaire avec float natifs
        input_data_list = []
        for row in features:
           input_data = {feature: float(value) for feature, value in zip(feature_names, row)}
           input_data_list.append(input_data)

        # Créer un DataFrame pour les prédictions
        input_df = pd.DataFrame(input_data_list)
        
        # Faire les prédictions
        predictions = model.predict(input_df)
        
        # Ajouter les prédictions au dataframe original
        result_df = df.copy()
        result_df['Prediction'] = predictions
        
        # Mapper les prédictions à des labels
        result_df['Prediction_Label'] = result_df['Prediction'].map({
            0: 'Échec', 
            1: 'Réussite'
        })
        
        return result_df
        
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        return None

def analyze_predictions(df):
    """Analyser les résultats des prédictions"""
    
    st.header("📊 Analyse des Prédictions")
    
    # Métriques globales des prédictions
    st.subheader("🎯 Métriques Globales")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(df)
        st.metric("Total Étudiants", total_students)
    
    with col2:
        predicted_success = (df['Prediction'] == 1).sum()
        success_rate = (predicted_success / total_students) * 100
        st.metric("Prédictions Réussite", f"{predicted_success} ({success_rate:.1f}%)")
    
    with col3:
        predicted_failure = (df['Prediction'] == 0).sum()
        failure_rate = (predicted_failure / total_students) * 100
        st.metric("Prédictions Échec", f"{predicted_failure} ({failure_rate:.1f}%)")
    
    with col4:
        if 'enrollment_age' in df.columns:
            avg_age = df['enrollment_age'].mean()
            st.metric("Âge Moyen", f"{avg_age:.1f} ans")
    
    # Distribution des prédictions
    st.subheader("📈 Distribution des Prédictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique en barres des prédictions
        prediction_counts = df['Prediction_Label'].value_counts()
        fig = px.bar(
            x=prediction_counts.index, 
            y=prediction_counts.values,
            title="Distribution des Prédictions",
            labels={'x': 'Prédiction', 'y': 'Nombre d\'étudiants'},
            color=prediction_counts.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Graphique en secteurs
        fig = px.pie(
            values=prediction_counts.values, 
            names=prediction_counts.index,
            title="Répartition des Prédictions"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyse par caractéristiques
    st.subheader("🔍 Analyse par Caractéristiques")
    
    # Prédictions par genre
    if 'Gender' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prédictions par Genre**")
            gender_pred = df.groupby('Gender')['Prediction'].agg(['mean', 'count']).reset_index()
            gender_pred['Gender_Label'] = gender_pred['Gender'].map({0: 'Homme', 1: 'Femme'})
            gender_pred['Success_Rate'] = gender_pred['mean'] * 100
            
            fig = px.bar(
                gender_pred, 
                x='Gender_Label', 
                y='Success_Rate',
                title="Taux de Réussite Prédit par Genre",
                color='Success_Rate',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'has_scholarship' in df.columns:
                st.write("**Prédictions par Statut de Bourse**")
                scholarship_pred = df.groupby('has_scholarship')['Prediction'].agg(['mean', 'count']).reset_index()
                scholarship_pred['Scholarship_Label'] = scholarship_pred['has_scholarship'].map({0: 'Sans bourse', 1: 'Avec bourse'})
                scholarship_pred['Success_Rate'] = scholarship_pred['mean'] * 100
                
                fig = px.bar(
                    scholarship_pred, 
                    x='Scholarship_Label', 
                    y='Success_Rate',
                    title="Taux de Réussite Prédit par Bourse",
                    color='Success_Rate',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des notes prédites
    if 'course_sem1_grade' in df.columns and 'course_sem2_grade' in df.columns:
        st.subheader("📚 Analyse des Notes vs Prédictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des notes du semestre 1 par prédiction
            fig = px.box(
                df, 
                x='Prediction_Label', 
                y='course_sem1_grade',
                title="Distribution des Notes S1 par Prédiction",
                color='Prediction_Label'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution des notes du semestre 2 par prédiction
            fig = px.box(
                df, 
                x='Prediction_Label', 
                y='course_sem2_grade',
                title="Distribution des Notes S2 par Prédiction",
                color='Prediction_Label'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyse des facteurs de risque
    st.subheader("⚠️ Analyse des Facteurs de Risque")
    
    risk_factors = ['Debtor', 'Displaced']
    available_risk_factors = [col for col in risk_factors if col in df.columns]
    
    if available_risk_factors:
        cols = st.columns(len(available_risk_factors))
        
        for i, factor in enumerate(available_risk_factors):
            with cols[i]:
                factor_analysis = df.groupby(factor)['Prediction'].agg(['mean', 'count']).reset_index()
                factor_analysis['Success_Rate'] = factor_analysis['mean'] * 100
                
                st.write(f"**Impact {factor}**")
                for _, row in factor_analysis.iterrows():
                    status = "Oui" if row[factor] == 1 else "Non"
                    st.write(f"• {status}: {row['Success_Rate']:.1f}% réussite ({row['count']} étudiants)")
    
    # Résumé des insights
    st.subheader("💡 Insights Clés")
    
    insights = []
    
    # Insight sur le taux global
    success_rate = (df['Prediction'] == 1).mean() * 100
    if success_rate > 70:
        insights.append(f"✅ Taux de réussite prédit élevé ({success_rate:.1f}%)")
    elif success_rate < 50:
        insights.append(f"⚠️ Taux de réussite prédit faible ({success_rate:.1f}%)")
    else:
        insights.append(f"📊 Taux de réussite prédit modéré ({success_rate:.1f}%)")
    
    # Insight sur le genre
    if 'Gender' in df.columns:
        gender_analysis = df.groupby('Gender')['Prediction'].mean()
        if len(gender_analysis) == 2:
            diff = abs(gender_analysis.iloc[0] - gender_analysis.iloc[1]) * 100
            if diff > 10:
                better_gender = "Femmes" if gender_analysis.iloc[1] > gender_analysis.iloc[0] else "Hommes"
                insights.append(f"👥 Écart significatif entre genres: {better_gender} ont un meilleur taux prédit ({diff:.1f}% d'écart)")
    
    # Insight sur les bourses
    if 'has_scholarship' in df.columns:
        scholarship_analysis = df.groupby('has_scholarship')['Prediction'].mean()
        if len(scholarship_analysis) == 2:
            diff = (scholarship_analysis.iloc[1] - scholarship_analysis.iloc[0]) * 100
            if diff > 5:
                insights.append(f"🎓 Impact positif des bourses: +{diff:.1f}% de taux de réussite prédit")
            elif diff < -5:
                insights.append(f"🎓 Impact surprenant: les non-boursiers ont un meilleur taux prédit ({abs(diff):.1f}%)")
    
    for insight in insights:
        st.write(insight)
    
    return df

def main():
    st.title("🎓 Dashboard d'Analyse des Performances Étudiants")
    st.markdown("---")
    
    # Load ML model
    model = load_model()
    
    # Interface de téléchargement
    uploaded_file, use_sample, predict_mode = upload_interface()
    
    # Handle prediction mode
    if predict_mode and uploaded_file is not None:
        if model is None:
            st.error("Modèle non chargé. Impossible de faire des prédictions.")
            st.stop()
            
        try:
            # Load data for prediction
            df = pd.read_csv(uploaded_file)
            
            # Make predictions
            with st.spinner("Génération des prédictions en cours..."):
                result_df = make_predictions(df, model)
                
            if result_df is not None:
                st.success("✅ Prédictions terminées avec succès!")
                
                # Aperçu des données prédites
                st.header("👀 Aperçu des Données Prédites")
                
                # Afficher les premières lignes avec les prédictions
                st.write("**Échantillon des prédictions (10 premières lignes):**")
                display_cols = ['Prediction_Label'] + [col for col in result_df.columns if col not in ['Prediction', 'Prediction_Label']][:8]
                st.dataframe(result_df[display_cols].head(10), use_container_width=True)
                
                # Analyse complète des prédictions
                analyze_predictions(result_df)
                
                # Section de téléchargement
                st.header("💾 Télécharger les Résultats")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Télécharger toutes les données avec prédictions
                    csv_buffer = io.StringIO()
                    result_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📥 Télécharger les prédictions complètes (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name=f"predictions_completes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Télécharger seulement un résumé
                    summary_df = result_df[['Prediction_Label', 'Gender', 'has_scholarship', 'enrollment_age', 'course_sem1_grade', 'course_sem2_grade']].copy()
                    summary_csv = io.StringIO()
                    summary_df.to_csv(summary_csv, index=False)
                    
                    st.download_button(
                        label="📄 Télécharger le résumé (CSV)",
                        data=summary_csv.getvalue(),
                        file_name=f"resume_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            st.stop()  # Don't proceed to regular dashboard in prediction mode
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            st.stop()
    
    # Rest of your existing main function continues here...
    # [Keep all your existing dashboard code below]
    
    # Chargement des données
    df, data_loaded = load_data(uploaded_file, use_sample)
    
    if not data_loaded or df is None:
        st.warning("Veuillez charger des données pour continuer l'analyse.")
        st.stop()
    
    st.success(f"✅ Données chargées avec succès! ({len(df)} étudiants)")
    st.markdown("---")
    
    # Sidebar pour les filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtres dynamiques basés sur les données chargées
    genre_options = sorted(df['Gender'].unique()) if 'Gender' in df.columns else [0, 1]
    genre_filter = st.sidebar.multiselect(
        "Genre", 
        options=genre_options, 
        default=genre_options,
        format_func=lambda x: "Homme" if x == 0 else "Femme"
    )
    
    bourse_options = sorted(df['has_scholarship'].unique()) if 'has_scholarship' in df.columns else [0, 1]
    bourse_filter = st.sidebar.multiselect(
        "Bourse", 
        options=bourse_options, 
        default=bourse_options,
        format_func=lambda x: "Sans bourse" if x == 0 else "Avec bourse"
    )
    
    # Application des filtres
    df_filtered = df[
        (df['Gender'].isin(genre_filter)) &
        (df['has_scholarship'].isin(bourse_filter)) 
    ]
    
    if len(df_filtered) == 0:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        st.stop()
    
    # Métriques globales
    st.header("🎯 KPIs Globaux")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        taux_reussite = (df_filtered['Target'] == 1).mean() * 100
        st.metric("Taux de Réussite Global", f"{taux_reussite:.1f}%")
    
    with col2:
        taux_echec = (df_filtered['Target'] == 0).mean() * 100
        st.metric("Taux d'Échec Global", f"{taux_echec:.1f}%")
    
    with col3:
        if 'course_sem1_grade' in df_filtered.columns and 'course_sem2_grade' in df_filtered.columns:
            note_moyenne = (df_filtered['course_sem1_grade'].mean() + df_filtered['course_sem2_grade'].mean()) / 2
            st.metric("Note Moyenne", f"{note_moyenne:.1f}/20")
        else:
            st.metric("Note Moyenne", "N/A")
    
    with col4:
        if 'enrollment_age' in df_filtered.columns:
            age_moyen = df_filtered['enrollment_age'].mean()
            st.metric("Âge Moyen à l'Inscription", f"{age_moyen:.1f} ans")
        else:
            st.metric("Âge Moyen", "N/A")
    
    st.markdown("---")
    
    # KPIs par catégorie
    st.header("📊 Analyse par Catégorie")
    
    # Réussite par genre
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Réussite par Genre")
        success_by_gender = df_filtered.groupby('Gender')['Target'].agg(['mean', 'count']).reset_index()
        success_by_gender['Gender_label'] = success_by_gender['Gender'].map({0: 'Homme', 1: 'Femme'})
        success_by_gender['success_rate'] = success_by_gender['mean'] * 100
        
        fig = px.bar(success_by_gender, x='Gender_label', y='success_rate',
                     title="Taux de Réussite par Genre",
                     color='success_rate',
                     color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Réussite par Bourse")
        success_by_scholarship = df_filtered.groupby('has_scholarship')['Target'].agg(['mean', 'count']).reset_index()
        success_by_scholarship['scholarship_label'] = success_by_scholarship['has_scholarship'].map({0: 'Sans bourse', 1: 'Avec bourse'})
        success_by_scholarship['success_rate'] = success_by_scholarship['mean'] * 100
        
        fig = px.bar(success_by_scholarship, x='scholarship_label', y='success_rate',
                     title="Taux de Réussite par Statut de Bourse",
                     color='success_rate',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # KPIs de progression (seulement si les colonnes existent)
    if all(col in df_filtered.columns for col in ['course_sem1_enrolled', 'course_sem1_passed', 'course_sem2_enrolled', 'course_sem2_passed']):
        st.header("📈 Analyse de Progression")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Taux de Réussite par Semestre")
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy['taux_reussite_sem1'] = (df_filtered_copy['course_sem1_passed'] / df_filtered_copy['course_sem1_enrolled']) * 100
            df_filtered_copy['taux_reussite_sem2'] = (df_filtered_copy['course_sem2_passed'] / df_filtered_copy['course_sem2_enrolled']) * 100
            
            progression_data = pd.DataFrame({
                'Semestre': ['Semestre 1', 'Semestre 2'],
                'Taux_Réussite': [df_filtered_copy['taux_reussite_sem1'].mean(), df_filtered_copy['taux_reussite_sem2'].mean()]
            })
            
            fig = px.line(progression_data, x='Semestre', y='Taux_Réussite',
                          title="Évolution du Taux de Réussite",
                          markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'course_sem1_grade' in df_filtered.columns and 'course_sem2_grade' in df_filtered.columns:
                st.subheader("Évolution des Notes")
                df_filtered_copy = df_filtered.copy()
                df_filtered_copy['evolution_notes'] = df_filtered_copy['course_sem2_grade'] - df_filtered_copy['course_sem1_grade']
                
                fig = px.histogram(df_filtered_copy, x='evolution_notes', nbins=30,
                                  title="Distribution de l'Évolution des Notes (S2-S1)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                evolution_moyenne = df_filtered_copy['evolution_notes'].mean()
                st.metric("Évolution Moyenne des Notes", f"{evolution_moyenne:.2f} points")
        
        st.markdown("---")
    # Nouvelle section pour la visualisation des clusters
    if 'Cluster' in df_filtered.columns:
        st.header("🔮 Analyse des Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Répartition des Clusters")
            cluster_counts = df_filtered['Cluster'].value_counts().sort_index()
            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Distribution des Clusters"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Caractéristiques des Clusters")
            
            # Sélectionner les caractéristiques à analyser
            cluster_features = [
                'Target', 'course_sem1_grade', 'course_sem2_grade',
                'enrollment_age', 'has_scholarship'
            ]
            available_features = [f for f in cluster_features if f in df_filtered.columns]
            
            if available_features:
                selected_feature = st.selectbox(
                    "Sélectionnez une caractéristique à analyser",
                    available_features
                )
                
                # Boxplot de la caractéristique par cluster
                fig = px.box(
                    df_filtered,
                    x='Cluster',
                    y=selected_feature,
                    color='Cluster',
                    title=f"Distribution de {selected_feature} par Cluster",
                    labels={'Cluster': 'Cluster', selected_feature: selected_feature}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse approfondie des clusters
        st.subheader("Profil des Clusters")
        
        # Calcul des moyennes par cluster
        if len(available_features) > 0:
            cluster_profiles = df_filtered.groupby('Cluster')[available_features].mean().reset_index()
            
            # Afficher les caractéristiques moyennes par cluster
            st.write("**Caractéristiques moyennes par cluster:**")
            st.dataframe(cluster_profiles.style.background_gradient(cmap='Blues'), use_container_width=True)
            

    # KPIs comportementaux (seulement si les colonnes existent)
    behavioral_cols = ['Debtor', 'Displaced', 'Marital_status', 'tuition_paid']
    available_behavioral_cols = [col for col in behavioral_cols if col in df_filtered.columns]
    st.markdown("---")
    if available_behavioral_cols:
        st.header("🧮 Indicateurs Comportementaux")
        
        cols = st.columns(len(available_behavioral_cols))
        
        for i, col_name in enumerate(available_behavioral_cols):
            with cols[i]:
                pct = (df_filtered[col_name] == 1).mean() * 100
                
                if col_name == 'Debtor':
                    st.metric("% Étudiants Débiteurs", f"{pct:.1f}%")
                elif col_name == 'Displaced':
                    st.metric("% Étudiants Déplacés", f"{pct:.1f}%")
                elif col_name == 'Marital_status':
                    st.metric("% Étudiants non Mariés", f"{pct:.1f}%")
                elif col_name == 'tuition_paid':
                    st.metric("% Frais Payés", f"{pct:.1f}%")
        
        st.markdown("---")
    
    # KPIs personnalisés
    st.header("💡 Analyses Spécialisées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Analyses Personnalisées")
        
        if 'has_scholarship' in df_filtered.columns and 'Debtor' in df_filtered.columns:
            # Réussite sans bourse et avec dettes
            cas_difficile = df_filtered[
                (df_filtered['has_scholarship'] == 0) & 
                (df_filtered['Debtor'] == 1)
            ]
            if len(cas_difficile) > 0:
                taux_cas_difficile = (cas_difficile['Target'] == 1).mean() * 100
                st.write(f"📊 **Sans bourse + Avec dettes**: {taux_cas_difficile:.1f}% de réussite")
        
        if all(col in df_filtered.columns for col in ['has_scholarship', 'course_sem1_grade', 'course_sem2_grade']):
            # Moyenne notes boursiers vs non-boursiers
            notes_boursiers = df_filtered[df_filtered['has_scholarship'] == 1][['course_sem1_grade', 'course_sem2_grade']].mean().mean()
            notes_non_boursiers = df_filtered[df_filtered['has_scholarship'] == 0][['course_sem1_grade', 'course_sem2_grade']].mean().mean()
            
            st.write(f"📈 **Notes moyennes**:")
            st.write(f"   • Avec bourse: {notes_boursiers:.1f}/20")
            st.write(f"   • Sans bourse: {notes_non_boursiers:.1f}/20")
    
    with col2:
        st.subheader("Recommandations")
        st.write("🎯 **Actions suggérées :**")
        
        # Génération de recommandations basées sur les données
        if 'has_scholarship' in df_filtered.columns:
            scholarship_success = df_filtered.groupby('has_scholarship')['Target'].mean()
            if len(scholarship_success) == 2 and scholarship_success.iloc[1] > scholarship_success.iloc[0]:
                diff = (scholarship_success.iloc[1] - scholarship_success.iloc[0]) * 100
                st.write(f"• Augmenter les bourses (+{diff:.1f}% de réussite)")
        
        if 'Debtor' in df_filtered.columns:
            debtor_impact = df_filtered.groupby('Debtor')['Target'].mean()
            if len(debtor_impact) == 2 and debtor_impact.iloc[0] > debtor_impact.iloc[1]:
                st.write("• Mettre en place un soutien financier")
        
        st.write("• Suivi personnalisé des étudiants à risque")
        st.write("• Programmes de tutorat ciblés")
    
    # Matrice de corrélation
    st.header("🔗 Matrice de Corrélation")
    
    # Colonnes numériques disponibles
    potential_numeric_cols = ['Marital_status', 'Previous_qualification', 'Displaced', 'Debtor', 
                             'tuition_paid', 'Gender', 'has_scholarship', 'enrollment_age',
                             'course_sem1_grade', 'course_sem2_grade', 'Target']
    
    numeric_cols = [col for col in potential_numeric_cols if col in df_filtered.columns]
    
    if len(numeric_cols) > 2:
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Matrice de Corrélation des Variables")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Données brutes
    st.header("📋 Données Détaillées")
    st.dataframe(df_filtered, use_container_width=True)
    
    # Statistiques descriptives
    st.header("📊 Statistiques Descriptives")
    st.dataframe(df_filtered.describe(), use_container_width=True)
    
    # Bouton de téléchargement des données filtrées
    st.header("💾 Téléchargement")
    csv_buffer = io.StringIO()
    df_filtered.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Télécharger les données filtrées (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"donnees_filtrees_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()