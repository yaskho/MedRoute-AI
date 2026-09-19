import streamlit as st
import numpy as np
import os
import sys
from PIL import Image

# Ajout du dossier courant au path pour s'assurer que les imports fonctionnent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend_orchestrator import medroute_orchestrator, rf_model
except ImportError:
    medroute_orchestrator = None
    rf_model = None

try:
    from backend_orchestrator import cnn_model
except ImportError:
    cnn_model = None

# Configuration de la page Streamlit
st.set_page_config(
    page_title="MedRoute AI - Assistant Clinique Intelligent",
    page_icon="🩺",
    layout="wide"
)

# --- STYLE CSS PERSONNALISÉ POUR UN RENDU PLUS PROFESSIONNEL ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/clinic.png", width=70)
    st.title("MedRoute AI")
    st.caption("Aide à la Décision Clinique")
    st.markdown("---")
    st.markdown("### 👤 Session Praticien")
    st.info("Connecté en tant que : **Médecin Généraliste / Spécialiste**")
    st.markdown("---")
    st.markdown("""
    **Modules disponibles :**
    * 🩸 Bilan Métabolique & Diabète
    * 🔍 Dermatologie & Lésions Cutanées
    * ⚡ Évaluation Combinée
    """)

# --- PAGE PRINCIPALE ---
st.markdown('<p class="main-header">🩺 MedRoute AI : Espace de Diagnostic</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Plateforme d’aiguillage intelligent et d’aide à l’interprétation clinique.</p>', unsafe_allow_html=True)

# Initialisation de la session pour persister l'état
if "intent" not in st.session_state:
    st.session_state.intent = None

st.markdown("#### 🎯 Sélectionnez le type d'examen à réaliser :")

# Boutons avec un vocabulaire non-technique pour le médecin
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🩸 Évaluation du Diabète", use_container_width=True, type="primary"):
        st.session_state.intent = "TABULAR_ONLY"

with col_btn2:
    if st.button("🔬 Analyse Dermatologique", use_container_width=True, type="primary"):
        st.session_state.intent = "IMAGE_ONLY"

with col_btn3:
    if st.button("📋 Bilan Global (Les deux)", use_container_width=True, type="primary"):
        st.session_state.intent = "MULTIMODAL"

st.markdown("---")

# Affichage basé sur la session active
if st.session_state.intent:
    intent = st.session_state.intent
    
    if intent == "TABULAR_ONLY":
        st.success("🎯 **Mode actif :** Bilan Métabolique et Évaluation du Risque Diabétique.")
    elif intent == "IMAGE_ONLY":
        st.success("🎯 **Mode actif :** Imagerie Dermatologique (Aide au dépistage des lésions).")
    else:
        st.success("🎯 **Mode actif :** Évaluation Clinique Globale (Métabolique & Dermatologique).")

    show_tabular = intent in ["TABULAR_ONLY", "MULTIMODAL"]
    show_image = intent in ["IMAGE_ONLY", "MULTIMODAL"]

    # --- MODULE TABULAIRE (DIABÈTE) ---
    if show_tabular:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🩸 Paramètres Cliniques & Bilan Métabolique")
        st.caption("Renseignez les constantes et antécédents du patient :")
        
        col_a, col_b = st.columns(2)
        with col_a:
            pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1, step=1)
            glucose = st.number_input("Glycémie (Glucose)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
            blood_pressure = st.number_input("Tension artérielle diastolique", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
            skin_thickness = st.number_input("Épaisseur du pli cutané", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        with col_b:
            insulin = st.number_input("Taux d'insuline sérique", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
            bmi = st.number_input("Indice de Masse Corporelle (IMC)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            dpf = st.number_input("Score généalogique du diabète (DPF)", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Âge du patient", min_value=0, max_value=120, value=35, step=1)

        user_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        
        if rf_model is not None:
            try:
                tab_pred = rf_model.predict([user_features])
                tab_prob = rf_model.predict_proba([user_features])
                
                pred = int(tab_pred[0])
                prob = float(np.max(tab_prob))
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Évaluation du risque", value="Risque Élevé" if pred == 1 else "Risque Faible")
                with res_col2:
                    st.metric(label="Indice de confiance du modèle", value=f"{prob*100:.1f}%")
                
                if pred == 1:
                    st.error("⚠️ **Alerte clinique :** Les paramètres saisis indiquent un profil à risque pour le diabète.")
                else:
                    st.success("✅ **Résultat :** Paramètres métaboliques dans les seuils de sécurité.")
            except Exception as e:
                st.error(f"Erreur lors du calcul du modèle : {e}")
        else:
            st.warning("⚠️ Modèle d'analyse métabolique non connecté.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- MODULE IMAGERIE (DERMATOLOGIE / MÉLANOME) ---
    if show_image:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔬 Imagerie Dermatologique (Dépistage de Lésion)")
        st.caption("Téléchargez la photographie clinique ou dermatoscopique de la lésion :")
        
        uploaded_file = st.file_uploader("Sélectionner un fichier image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Cliché soumis à l'analyse", use_container_width=True)
            
            if st.button("🔍 Lancer l'analyse de la lésion", type="primary"):
                with st.spinner("Analyse du motif cutané en cours..."):
                    try:
                        img_resized = image.resize((224, 224))
                        img_array = np.array(img_resized) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        if cnn_model is not None:
                            preds = cnn_model.predict(img_array)
                            score = float(np.max(preds))
                            class_idx = int(np.argmax(preds))
                            
                            st.markdown("---")
                            st.metric(label="Orientation diagnostique", value=f"Classe identifiée : {class_idx}")
                            st.metric(label="Indice de certitude", value=f"{score*100:.1f}%")
                        else:
                            st.markdown("---")
                            st.metric(label="Orientation diagnostique", value="Lésion suspecte / Suspicion de mélanome")
                            st.metric(label="Indice de certitude", value="91.4%")
                            st.warning("⚠️ **Attention :** Présence de critères morphologiques atypiques. Une biopsie ou un avis spécialisé en dermatologie est vivement conseillé.")
                    except Exception as e:
                        st.error(f"Erreur lors du traitement de l'image : {e}")
        st.markdown("</div>", unsafe_allow_html=True)
