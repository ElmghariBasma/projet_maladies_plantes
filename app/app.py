import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# ────── CHARGEMENT DU MODÈLE ───────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.h5")

try:
    model = load_model()
except Exception as e:
    st.error("❌ Impossible de charger le modèle. Vérifie que 'model.h5' est dans le même dossier.")
    st.stop()

# ───────── FONCTION DE PRÉDICTION ────────
def predict_image(img: Image.Image):
    # Détermine automatiquement la taille d'entrée requise
    input_shape = model.input_shape
    if input_shape[1] is None or input_shape[2] is None:
        raise ValueError("Le modèle a une forme d'entrée dynamique non supportée.")
    target_size = (input_shape[1], input_shape[2])

    # Redimensionne l'image
    img = img.resize(target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalisation (à adapter si ton modèle utilise autre chose)
    if img_array.max() > 1.0:
        img_array /= 255.0

    # Prédiction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Liste des classes (doit correspondre à l'entraînement)
    class_names = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry___Powdery_mildew',
        'Cherry___healthy',
        'Corn___Common_rust',
        'Corn___Gray_leaf_spot',
        'Corn___Northern_Leaf_Blight',
        'Corn___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Citrus_greening',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper___Bacterial_spot',
        'Pepper___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites_Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    if predicted_class_idx >= len(class_names):
        return {"plant": "Inconnu", "condition": "Classe non reconnue", "confidence": 0.0}

    predicted_class = class_names[predicted_class_idx]
    if "___" in predicted_class:
        plant, condition = predicted_class.split("___", 1)
    else:
        plant, condition = "Inconnu", predicted_class

    return {
        "plant": plant,
        "condition": condition,
        "confidence": confidence
    }
# ───── CONFIG ──────
st.set_page_config(
    page_title="Hosplant",
    layout="wide",
    page_icon="🌿"
)

# ────── CSS ───────
st.markdown("""
<style>
.stApp > header {display: none !important;}

div.stButton > button {
    background-color: #2f855a !important;
    color: white !important;
    border-radius: 999px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
}
div.stButton > button:hover {
    background-color: #276749 !important;
}
div.stButton > button[kind="secondary"] {
    background-color: transparent !important;
    color: #2f855a !important;
    border: 1px solid #2f855a !important;
}
div.stButton > button[kind="secondary"]:hover {
    background-color: #e6f7e6 !important;
}

.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 2rem 0 0.5rem;
}
.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}

.upload-container {
    border: 2px dashed #d0d0d0;
    border-radius: 16px;
    padding: 60px 20px;
    text-align: center;
    max-width: 700px;
    margin: 2rem auto;
    background: #fafafa;
}

.plant-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    overflow: hidden;
    background: white;
    margin-bottom: 1.5rem;
}
.plant-image {
    width: 100%;
    height: 180px;
    object-fit: cover;
    border-radius: 12px 12px 0 0;
}
.tag {
    display: inline-block;
    background: #e6f7e6;
    color: #2f855a;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    margin: 2px 6px 2px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── NAVIGATION ───────────────────────
if "current_page" not in st.session_state:
    st.session_state.current_page = "Detect"

nav1, nav2, nav3 = st.columns([6, 2, 2])
with nav1:
    st.markdown("<h3 style='color:#2f855a;'>🌿 Hosplant</h3>", unsafe_allow_html=True)
with nav2:
    if st.button("Detect", type="primary" if st.session_state.current_page == "Detect" else "secondary", use_container_width=True):
        st.session_state.current_page = "Detect"
        st.rerun()
with nav3:
    if st.button("Supported Plants", type="primary" if st.session_state.current_page == "Plants" else "secondary", use_container_width=True):
        st.session_state.current_page = "Plants"
        st.rerun()

# ─────────────────────── PAGE DETECT ───────────────────────
if st.session_state.current_page == "Detect":
    st.markdown('<div class="title">Plant Disease Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a photo of your plant to detect diseases</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="upload-container">
        <h3>🖼️ Drop your plant image here</h3>
        <p style="color:#777;">or click below to upload</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")  # Assure RGB
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(image, caption="Uploaded image", width=320)

        with col2:
            if st.button(" Analyser", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    result = predict_image(image)

                condition_clean = result['condition'].replace('_', ' ').title()
                is_healthy = "healthy" in result['condition'].lower()
                status_color = "#e6f7e6" if is_healthy else "#ffebee"
                status_text = f"{result['plant']} SAIN(E)" if is_healthy else f"{result['plant']} MALADE"
                status_icon = "" if is_healthy else ""

                st.markdown(f"""
                <div style="
                    background: {status_color};
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    font-weight: bold;
                    color: {'#22543d' if is_healthy else '#c62828'};
                    display: flex;
                    align-items: center;
                    gap: 8px;
                ">
                    {status_icon} {status_text}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Confiance")
                st.markdown(f"""
                <div style="
                    font-size: 2.4rem;
                    font-weight: bold;
                    color: #2f855a;
                    margin: 0.5rem 0;
                ">
                    {result['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="
                    background: #e3f2fd;
                    padding: 0.8rem;
                    border-radius: 8px;
                    border-left: 4px solid #1976d2;
                    margin-top: 1rem;
                ">
                    <strong>Condition:</strong> {condition_clean}
                </div>
                """, unsafe_allow_html=True)

                

# ── PAGE SUPPORTED PLANTS ──
else:
    st.markdown('<div class="title">Supported Plants</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Diseases supported by our AI model</div>', unsafe_allow_html=True)

    plants = [
        {"name": "Tomato", "sci": "Solanum lycopersicum", "diseases": ["Early Blight", "Late Blight", "Leaf Mold", "Septoria Leaf Spot"], "img": "https://marvel-b1-cdn.bc0a.com/f00000000214316/transform.octanecdn.com/crop/900x600/https://octanecdn.com/pikearmstrong/armstronggardencom_109426483.jpeg"},
        {"name": "Potato", "sci": "Solanum tuberosum", "diseases": ["Early Blight", "Late Blight"], "img": "https://images.pexels.com/photos/144248/potatoes-vegetables-erdfrucht-bio-144248.jpeg?cs=srgb&dl=pexels-pixabay-144248.jpg&fm=jpg"},
        {"name": "Corn", "sci": "Zea mays", "diseases": ["Common Rust", "Gray Leaf Spot", "Northern Leaf Blight"], "img": "https://www.ugaoo.com/cdn/shop/articles/9f9b3771a2.jpg?v=1727692315"},
        {"name": "Grape", "sci": "Vitis vinifera", "diseases": ["Black Rot", "Esca", "Leaf Blight"], "img": "https://www.shutterstock.com/image-photo/autumn-vineyard-grapes-ready-winery-260nw-2613617809.jpg"},
        {"name": "Apple", "sci": "Malus domestica", "diseases": ["Apple scab", "Black rot", "Cedar apple rust"], "img": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRBBZi1RACC_WmRLh2DPkNeMNKcZmj8Bed8dA&s"},
        {"name": "Bell Pepper", "sci": "Capsicum annuum", "diseases": ["Bacterial spot"], "img": "https://growfolk.co.za/wp-content/uploads/2025/10/Bell-Green-Pepper-Seed.webp"},
        {"name": "Cherry", "sci": "Prunus avium", "diseases": ["Powdery mildew"], "img": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSqee88s_5TD2LZzG3JH-RojaYBaMZCPCoVHQ&s"},
        {"name": "Peach", "sci": "Prunus persica", "diseases": ["Bacterial spot"], "img": "https://cdn.mos.cms.futurecdn.net/i4Eo3wqMZBECRSaoYdeG4i.jpg"},
        {"name": "Squash", "sci": "Cucurbita pepo", "diseases": ["Powdery mildew", "Bacterial spot"], "img": "https://www.lovethegarden.com/sites/default/files/styles/scale_fallback/public/content/articles/uk/growing-butternut-squahes-2.jpg.jpeg?itok=PGnlcNrA"},
        {"name": "Strawberry", "sci": "Fragaria × ananassa", "diseases": ["Leaf scorch"], "img": "https://static.vecteezy.com/system/resources/previews/059/892/573/large_2x/ripe-red-strawberries-growing-on-a-strawberry-plant-in-a-field-photo.jpg"},
        {"name": "Soybean", "sci": "Glycine max", "diseases": ["Bacterial spot"], "img": "https://www.shutterstock.com/image-photo/young-green-pods-varietal-soybeans-260nw-1621769488.jpg"},
        {"name": "Orange", "sci": "Citrus sinensis", "diseases": ["Citrus greening"], "img": "https://www.buygrow.co.za/cdn/shop/products/valencia_orange.jpg?v=1670442649&width=1445"},
        {"name": "Raspberry", "sci": "Rubus idaeus", "diseases": ["Botrytis gray mold"], "img": "https://mobileimages.lowes.com/productimages/ea69dbf5-c93d-4d03-bdb4-9fe62ac6c2df/62476490.jpg"},
        {"name": "Blueberry", "sci": "Vaccinium corymbosum", "diseases": ["Mummy berry"], "img": "https://fermegiroflee.com/cdn/shop/files/MSPlE1wO.jpg?v=1742057875&width=1946"},
    ]

    cols_per_row = 4
    cols = st.columns(cols_per_row)

    for i, plant in enumerate(plants):
        with cols[i % cols_per_row]:
            st.markdown('<div class="plant-card">', unsafe_allow_html=True)
            st.markdown(f'<img src="{plant["img"]}" class="plant-image">', unsafe_allow_html=True)
            st.subheader(plant["name"])
            st.caption(plant["sci"])
            st.write("**DETECTABLE DISEASES**")
            for d in plant["diseases"]:
                st.markdown(f'<span class="tag">{d}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if (i + 1) % cols_per_row == 0 and i + 1 < len(plants):
            cols = st.columns(cols_per_row)