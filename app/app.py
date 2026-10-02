import os
import sys
import uuid
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kp_image
from fpdf import FPDF

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BACKEND_ERROR = ""
rf_model, cnn_model = None, None
try:
    import backend_orchestrator as _bo
    rf_model = getattr(_bo, "rf_model", None)
    cnn_model = getattr(_bo, "melanoma_model", None)  # nom reel dans le backend
    if cnn_model is None:
        cnn_model = getattr(_bo, "cnn_model", None)
except Exception as _err:
    BACKEND_ERROR = str(_err)

CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]  # ordre alphabetique (HAM10000)
CLASS_INFO = {
    "akiec": ("Kératose actinique / maladie de Bowen", "Lésion pré-cancéreuse ou carcinome in situ, liée à l'exposition solaire."),
    "bcc": ("Carcinome basocellulaire", "Cancer cutané le plus fréquent, à croissance lente et rarement métastatique."),
    "bkl": ("Lésion kératosique bénigne", "Kératose séborrhéique ou lentigo solaire : lésions bénignes fréquentes."),
    "df": ("Dermatofibrome", "Petite tumeur cutanée bénigne, ferme au toucher."),
    "mel": ("Mélanome", "Cancer cutané le plus grave : le pronostic dépend de la précocité de la prise en charge."),
    "nv": ("Naevus mélanocytaire", "Grain de beauté commun, le plus souvent bénin."),
    "vasc": ("Lésion vasculaire", "Angiome, angiokératome ou granulome pyogénique."),
}
WATCH = ["akiec", "bcc", "mel"]  # classes a surveiller pour l'indice de suspicion
FEATURES_FR = ["Grossesses", "Glycémie", "Tension diastolique", "Pli cutané",
               "Insuline", "IMC", "Score généalogique (DPF)", "Âge"]
ROUTES = {
    "TABULAR_ONLY": "Évaluation du risque de diabète",
    "IMAGE_ONLY": "Analyse dermatologique",
    "MULTIMODAL": "Bilan global (données cliniques + image)",
}
COLORS = {"ok": "#10B981", "warn": "#F59E0B", "high": "#EF4444", "info": "#64748B"}
DISCLAIMER = ("MedRoute AI est un prototype académique d'aide à la décision. Les modèles sont entraînés sur des jeux de "
              "données publics limités et n'ont pas été validés cliniquement. Les résultats ne constituent pas un diagnostic "
              "et ne remplacent pas l'examen clinique ni le jugement du praticien, qui reste seul responsable de l'interprétation.")


def level(p):
    """p en % -> (libelle, ton)"""
    if p < 40:
        return "Faible", "ok"
    if p < 75:
        return "Modéré", "warn"
    return "Élevé", "high"


def band(x, cuts, labels):
    for c, lab in zip(cuts, labels):
        if x < c:
            return lab
    return labels[-1]


def read_parameters(v):
    preg, glu, bp, skin, ins, bmi, dpf, age = v
    info = ("Information", "info")
    return [
        ("Glycémie (test de tolérance, 2 h)", f"{glu:.0f}", "< 140",
         *band(glu, [140, 200], [("Normale", "ok"), ("Élevée", "warn"), ("Très élevée", "high")])),
        ("Tension diastolique (mm Hg)", f"{bp:.0f}", "60 - 79",
         *band(bp, [60, 80, 90], [("Basse", "warn"), ("Normale", "ok"), ("Limite haute", "warn"), ("Élevée", "high")])),
        ("IMC (kg/m²)", f"{bmi:.1f}", "18,5 - 24,9",
         *band(bmi, [18.5, 25, 30], [("Insuffisance pondérale", "warn"), ("Normal", "ok"), ("Surpoids", "warn"), ("Obésité", "high")])),
        ("Insuline sérique (µU/mL)", f"{ins:.0f}", "16 - 166",
         *band(ins, [16, 166.01], [("Basse", "warn"), ("Normale", "ok"), ("Élevée", "warn")])),
        ("Épaisseur du pli cutané (mm)", f"{skin:.0f}", "-", *info),
        ("Score généalogique (DPF)", f"{dpf:.2f}", "-", *info),
        ("Âge (ans)", f"{age}", "-", *info),
        ("Nombre de grossesses", f"{preg}", "-", *info),
    ]


def badge(text, tone):
    return (f"<span style='background:{COLORS[tone]};color:#fff;padding:2px 10px;border-radius:999px;"
            f"font-size:0.8rem;font-weight:600'>{text}</span>")


def params_table_html(rows):
    body = "".join(f"<tr><td>{n}</td><td><b>{v}</b></td><td>{r}</td><td>{badge(l, t)}</td></tr>" for n, v, r, l, t in rows)
    return ("<table class='mr-table'><tr><th>Paramètre</th><th>Valeur</th><th>Référence</th><th>Lecture</th></tr>"
            f"{body}</table>")


def plot_gauge(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": " %"},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1D4ED8"},
               "steps": [{"range": [0, 40], "color": "#10B981"},
                         {"range": [40, 75], "color": "#F59E0B"},
                         {"range": [75, 100], "color": "#EF4444"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def plot_top3(probs):
    idx = np.argsort(probs)[::-1][:3][::-1]
    fig = go.Figure(go.Bar(
        x=[float(probs[i]) * 100 for i in idx],
        y=[CLASS_INFO[CLASS_CODES[i]][0] for i in idx],
        orientation="h", marker_color="#0F766E",
        text=[f"{probs[i] * 100:.1f} %" for i in idx], textposition="auto"))
    fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=30), xaxis=dict(range=[0, 100], title="Probabilité (%)"))
    return fig


def route_prompt(text):
    t = text.lower()
    diab = any(w in t for w in ["diabète", "diabete", "glycémie", "glycemie", "glucose", "métabolique", "metabolique",
                                "sucre", "tabulaire", "insuline", "imc", "biologique"])
    derm = any(w in t for w in ["peau", "lésion", "lesion", "mélanome", "melanome", "dermatolog", "image", "photo",
                                "tache", "cutané", "cutane", "grain de beauté", "cliché"])
    if diab and not derm:
        return "TABULAR_ONLY"
    if derm and not diab:
        return "IMAGE_ONLY"
    return "MULTIMODAL"


def _find_last_conv(model):
    if "out_relu" in [l.name for l in model.layers]:
        return "out_relu"
    for l in reversed(model.layers):
        try:
            if len(l.output.shape) == 4:
                return l.name
        except Exception:
            continue
    raise ValueError("Aucune couche convolutive trouvée pour Grad-CAM.")


def generate_gradcam_display(img_path, model, alpha=0.4):
    """Retourne l'image superposee (RGB) heatmap + image d'origine."""
    x = kp_image.img_to_array(kp_image.load_img(img_path, target_size=(224, 224)))
    x = np.expand_dims(x, axis=0) / 255.0

    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(_find_last_conv(model)).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        class_channel = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)).numpy()

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # BGR -> RGB (sinon rouge/bleu inverses)
    return np.clip(heatmap * alpha + img, 0, 255).astype(np.uint8)


def _s(t):
    for a, b in {"’": "'", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "…": "..."}.items():
        t = str(t).replace(a, b)
    return str(t).encode("latin-1", "replace").decode("latin-1")


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 118, 110)
        self.rect(0, 0, 210, 26, "F")
        self.set_xy(10, 7)
        self.set_font("helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, "MedRoute AI - Compte-rendu de consultation", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("helvetica", "I", 9)
        self.cell(0, 6, _s("Aide à la décision clinique - document réservé aux praticiens"), new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_y(32)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-14)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, _s(f"Page {self.page_no()} - Prototype non validé cliniquement - ne remplace pas le jugement du praticien"), align="C")

    def section(self, title):
        self.ln(3)
        self.set_font("helvetica", "B", 12)
        self.set_fill_color(226, 240, 238)
        self.set_text_color(15, 80, 74)
        self.cell(0, 8, _s(title), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def kv(self, k, v):
        self.set_font("helvetica", "B", 10)
        self.cell(52, 6, _s(k))
        self.set_font("helvetica", "", 10)
        self.multi_cell(0, 6, _s(v), new_x="LMARGIN", new_y="NEXT")

    def para(self, t, size=10):
        self.set_font("helvetica", "", size)
        self.multi_cell(0, 5.5, _s(t), new_x="LMARGIN", new_y="NEXT")

    def table(self, header, rows, widths):
        self.set_font("helvetica", "B", 9)
        self.set_fill_color(15, 118, 110)
        self.set_text_color(255, 255, 255)
        for h, w in zip(header, widths):
            self.cell(w, 7, _s(h), border=1, fill=True)
        self.ln()
        self.set_text_color(0, 0, 0)
        self.set_font("helvetica", "", 9)
        for r in rows:
            for c, w in zip(r, widths):
                self.cell(w, 6.5, _s(c), border=1)
            self.ln()


def create_pdf_report(meta, diab, derm):
    pdf = ReportPDF()
    pdf.set_margins(10, 10, 10)
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()

    pdf.section("1. Informations de consultation")
    pdf.kv("Date d'édition :", meta["date"])
    pdf.kv("Praticien :", meta["practitioner"] or "Non renseigné")
    pdf.kv("Examen réalisé :", meta["route"])
    if meta.get("request"):
        pdf.kv("Requête du praticien :", meta["request"])

    n = 2
    if diab:
        pdf.section(f"{n}. Bilan métabolique - risque de diabète")
        pdf.kv("Niveau de risque :", f"{diab['level']} (probabilité estimée : {diab['prob']:.1f} %)")
        pdf.ln(1)
        pdf.table(["Paramètre", "Valeur", "Référence", "Lecture"],
                  [(a, b, c, d) for a, b, c, d, _ in diab["rows"]], [70, 25, 35, 60])
        pdf.ln(2)
        pdf.para("Interprétation : " + diab["advice"])
        n += 1

    if derm:
        pdf.section(f"{n}. Analyse dermatologique par imagerie")
        pdf.kv("Lésion la plus probable :", derm["name"])
        pdf.kv("Certitude du modèle :", f"{derm['conf']:.1f} %")
        pdf.kv("Indice de suspicion :", f"{derm['suspicion']:.1f} % (mélanome, carcinome, kératose actinique)")
        pdf.ln(1)
        pdf.table(["Classe", "Probabilité"], [(a, f"{b:.1f} %") for a, b in derm["top3"]], [120, 40])
        pdf.ln(3)
        orig, grad = derm.get("orig_path"), derm.get("grad_path")
        if orig and os.path.exists(orig):
            w = 85
            ow, oh = Image.open(orig).size
            h = w * oh / ow
            if pdf.get_y() + h + 12 > 275:
                pdf.add_page()
            y = pdf.get_y()
            pdf.set_font("helvetica", "B", 9)
            pdf.set_xy(10, y)
            pdf.cell(w, 5, _s("Image analysée"))
            pdf.image(orig, x=10, y=y + 6, w=w)
            if grad and os.path.exists(grad):
                pdf.set_xy(105, y)
                pdf.cell(w, 5, _s("Carte d'explicabilité (Grad-CAM)"))
                pdf.image(grad, x=105, y=y + 6, w=w)
            pdf.set_y(y + 6 + h + 3)
            if grad:
                pdf.para("Les zones chaudes (rouge/jaune) indiquent les régions de la lésion ayant le plus influencé la décision du modèle.", 9)
                pdf.ln(1)
        pdf.para("Interprétation : " + derm["advice"])
        n += 1

    pdf.section(f"{n}. Limites et avertissements")
    pdf.para(DISCLAIMER, 9)
    return bytes(pdf.output())


st.set_page_config(page_title="MedRoute AI - Assistant Clinique", page_icon="🩺", layout="wide")
st.markdown("""
<style>
.mr-hero{background:linear-gradient(120deg,#0F766E,#1D4ED8);color:#fff;padding:26px 32px;border-radius:14px;margin-bottom:22px}
.mr-title{font-size:1.9rem;font-weight:700;color:#fff;line-height:1.2}
.mr-sub{margin-top:6px;opacity:.92;font-size:1.02rem;color:#fff}
.mr-verdict{padding:14px 18px;border-radius:12px;color:#fff;font-weight:700;font-size:1.15rem;margin:6px 0 12px}
.mr-table{width:100%;border-collapse:collapse;margin:6px 0 12px}
.mr-table th{text-align:left;padding:8px;border-bottom:2px solid rgba(128,128,128,.4);font-size:.85rem}
.mr-table td{padding:8px;border-bottom:1px solid rgba(128,128,128,.25);font-size:.92rem}
</style>
""", unsafe_allow_html=True)

for _k, _v in {"intent": None, "pdf": None, "request": "", "tmp": None}.items():
    st.session_state.setdefault(_k, _v)
if st.session_state.tmp is None:
    st.session_state.tmp = tempfile.mkdtemp()

with st.sidebar:
    st.markdown("## 🩺 MedRoute AI")
    st.caption("Aide à la décision clinique")
    st.markdown("---")
    st.markdown("### Consultation")
    st.text_input("Praticien", key="practitioner", placeholder="Ex. : Dr Martin")
    st.caption(f"Date : {datetime.now():%d/%m/%Y}")
    st.markdown("---")
    st.markdown("**Modules**\n\n🩸 Risque de diabète\n\n🔬 Lésions cutanées\n\n📋 Bilan global")
    st.markdown("---")
    st.caption("Prototype d'aide à la décision : ne remplace pas le jugement du médecin.")

st.markdown("""
<div class="mr-hero">
  <div class="mr-title">MedRoute AI</div>
  <div class="mr-sub">Choisissez un examen ou décrivez votre besoin : la plateforme oriente automatiquement votre demande vers le bon modèle.</div>
</div>
""", unsafe_allow_html=True)

if BACKEND_ERROR:
    st.error(f"Le backend n'a pas pu être chargé : {BACKEND_ERROR}")

st.markdown("#### 1. Quel examen souhaitez-vous réaliser ?")
CARDS = [("TABULAR_ONLY", "🩸", "Risque de diabète", "À partir des constantes et du bilan biologique du patient."),
         ("IMAGE_ONLY", "🔬", "Lésion cutanée", "Analyse d'une photographie ou d'une image dermatoscopique."),
         ("MULTIMODAL", "📋", "Bilan global", "Combine les données cliniques et l'image de la lésion.")]
for col, (key, icon, title, desc) in zip(st.columns(3), CARDS):
    with col:
        with st.container(border=True):
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            selected = st.session_state.intent == key
            if st.button("✔ Sélectionné" if selected else "Sélectionner", key=f"btn_{key}",
                         use_container_width=True, type="primary" if selected else "secondary"):
                st.session_state.intent, st.session_state.pdf, st.session_state.request = key, None, ""
                st.rerun()

st.markdown("#### Ou décrivez votre besoin en langage naturel")
with st.form("prompt_form"):
    txt = st.text_input("Requête", label_visibility="collapsed",
                        placeholder="Ex. : Évaluer le risque diabétique et analyser cette lésion suspecte")
    sent = st.form_submit_button("Analyser la requête", type="primary")
if sent and txt.strip():
    st.session_state.intent, st.session_state.request, st.session_state.pdf = route_prompt(txt), txt.strip(), None
    st.rerun()

intent = st.session_state.intent
if not intent:
    st.info("Sélectionnez un examen ci-dessus ou saisissez votre demande pour commencer.")
    st.stop()

st.markdown("---")
if st.session_state.request:
    st.caption(f"Requête : « {st.session_state.request} »")
st.markdown(f"#### 2. {ROUTES[intent]}")
summary_box = st.container()
show_tab = intent in ("TABULAR_ONLY", "MULTIMODAL")
show_img = intent in ("IMAGE_ONLY", "MULTIMODAL")


def diabetes_module():
    with st.container(border=True):
        st.subheader("🩸 Bilan métabolique")
        st.caption("Renseignez les constantes du patient.")
        c1, c2 = st.columns(2)
        with c1:
            preg = st.number_input("Nombre de grossesses", 0, 20, 1, 1)
            glu = st.number_input("Glycémie (test de tolérance, 2 h)", 0.0, 300.0, 120.0, 1.0)
            bp = st.number_input("Tension artérielle diastolique (mm Hg)", 0.0, 200.0, 70.0, 1.0)
            skin = st.number_input("Épaisseur du pli cutané (mm)", 0.0, 100.0, 20.0, 1.0)
        with c2:
            ins = st.number_input("Insuline sérique (µU/mL)", 0.0, 900.0, 80.0, 1.0)
            bmi = st.number_input("IMC (kg/m²)", 0.0, 70.0, 25.0, 0.1)
            dpf = st.number_input("Score généalogique du diabète (DPF)", 0.0, 3.0, 0.5, 0.01)
            age = st.number_input("Âge du patient", 0, 120, 35, 1)
        v = [preg, glu, bp, skin, ins, bmi, dpf, age]

        zeros = [n for n, x in zip(["glycémie", "tension", "pli cutané", "insuline", "IMC"], [glu, bp, skin, ins, bmi]) if x == 0]
        if zeros:
            st.warning("Valeur nulle pour : " + ", ".join(zeros) + ". Une valeur de 0 correspond à une donnée manquante : "
                       "le résultat peut être peu fiable.")
        if rf_model is None:
            st.error("Le modèle de prédiction du diabète n'est pas disponible.")
            return None
        try:
            proba = rf_model.predict_proba([v])[0]
            p = float(proba[list(rf_model.classes_).index(1)]) * 100
        except Exception as e:
            st.error(f"Erreur lors du calcul du modèle : {e}")
            return None

        lvl, tone = level(p)
        st.markdown("---")
        m1, m2 = st.columns(2)
        m1.metric("Niveau de risque", lvl)
        m2.metric("Probabilité estimée (modèle)", f"{p:.1f} %")
        st.plotly_chart(plot_gauge(p, "Risque de diabète"), use_container_width=True)
        rows = read_parameters(v)
        st.markdown("**Lecture des paramètres**")
        st.markdown(params_table_html(rows), unsafe_allow_html=True)
        st.caption("Références indicatives (seuils usuels) : à interpréter selon le contexte clinique.")
        advice = {"high": "Profil à risque élevé : confirmer par glycémie à jeun et HbA1c, puis prise en charge selon les recommandations en vigueur.",
                  "warn": "Risque intermédiaire : dépistage complémentaire (glycémie à jeun, HbA1c) et suivi des facteurs de risque conseillés.",
                  "ok": "Risque estimé faible d'après les paramètres saisis : suivi habituel et prévention."}[tone]
        {"high": st.error, "warn": st.warning, "ok": st.success}[tone](advice)
        return {"level": lvl, "prob": p, "rows": rows, "advice": advice}


def analyze_image(image, fname):
    uid = uuid.uuid4().hex[:8]
    orig_path = os.path.join(st.session_state.tmp, f"orig_{uid}.jpg")
    image.save(orig_path, quality=95)
    arr = np.expand_dims(np.array(image.resize((224, 224))).astype("float32") / 255.0, axis=0)
    probs = cnn_model.predict(arr, verbose=0)[0]
    top = int(np.argmax(probs))
    susp = float(sum(probs[CLASS_CODES.index(c)] for c in WATCH)) * 100
    _, tone = level(susp)
    advice = {"high": "Indice de suspicion élevé : avis dermatologique rapide avec examen dermoscopique et discussion d'une biopsie.",
              "warn": "Suspicion intermédiaire : examen dermoscopique par un spécialiste et surveillance rapprochée conseillés.",
              "ok": "Suspicion faible : surveillance clinique habituelle (règle ABCDE), à réévaluer en cas d'évolution de la lésion."}[tone]
    res = {"file": fname, "code": CLASS_CODES[top], "name": CLASS_INFO[CLASS_CODES[top]][0], "conf": float(probs[top]) * 100,
           "suspicion": susp, "probs": probs, "advice": advice, "orig_path": orig_path, "grad_path": None,
           "top3": [(CLASS_INFO[CLASS_CODES[i]][0], float(probs[i]) * 100) for i in np.argsort(probs)[::-1][:3]]}
    try:
        overlay = generate_gradcam_display(orig_path, cnn_model)
        res["grad_path"] = os.path.join(st.session_state.tmp, f"grad_{uid}.jpg")
        Image.fromarray(overlay).save(res["grad_path"], quality=95)
    except Exception as e:
        res["grad_err"] = str(e)
    return res


def render_derm(res):
    lvl, tone = level(res["suspicion"])
    st.markdown("---")
    st.markdown(f"<div class='mr-verdict' style='background:{COLORS[tone]}'>Lésion la plus probable : {res['name']}</div>",
                unsafe_allow_html=True)
    st.caption(CLASS_INFO[res["code"]][1])
    m1, m2 = st.columns(2)
    m1.metric("Certitude du modèle", f"{res['conf']:.1f} %")
    m2.metric("Indice de suspicion", f"{res['suspicion']:.1f} %", help="Somme des probabilités : mélanome, carcinome basocellulaire, kératose actinique.")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(plot_gauge(res["suspicion"], "Indice de suspicion lésionnelle"), use_container_width=True)
    with g2:
        st.markdown("**Trois diagnostics les plus probables**")
        st.plotly_chart(plot_top3(res["probs"]), use_container_width=True)
    st.markdown("**Explicabilité (Grad-CAM)**")
    i1, i2 = st.columns(2)
    i1.image(res["orig_path"], caption="Image analysée", use_container_width=True)
    if res.get("grad_path"):
        i2.image(res["grad_path"], caption="Zones ayant guidé la décision du modèle", use_container_width=True)
        st.caption("Les zones chaudes (rouge/jaune) sont les régions les plus influentes pour le réseau. "
                   "Elles orientent l'examen mais ne constituent pas une preuve clinique.")
    else:
        i2.warning(f"La carte Grad-CAM n'a pas pu être générée : {res.get('grad_err', 'erreur inconnue')}")
    {"high": st.error, "warn": st.warning, "ok": st.success}[tone](res["advice"])


def derm_module():
    with st.container(border=True):
        st.subheader("🔬 Imagerie dermatologique")
        st.caption("Téléchargez la photographie clinique ou dermatoscopique de la lésion.")
        up = st.file_uploader("Fichier image (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if up is None:
            return None
        image = Image.open(up).convert("RGB")
        st.image(image, caption="Cliché soumis à l'analyse", width=320)
        if cnn_model is None:
            st.error("Le modèle d'analyse d'image n'est pas disponible : aucun résultat ne peut être produit.")
            return None
        if st.button("🔍 Lancer l'analyse de la lésion", type="primary"):
            with st.spinner("Analyse de la lésion et génération de la carte d'explicabilité..."):
                try:
                    st.session_state.derm = analyze_image(image, up.name)
                except Exception as e:
                    st.session_state.derm = None
                    st.error(f"Erreur lors du traitement de l'image : {e}")
            st.session_state.pdf = None
        res = st.session_state.get("derm")
        if res and res["file"] == up.name:
            render_derm(res)
            return res
        return None


diab_res = diabetes_module() if show_tab else None
derm_res = derm_module() if show_img else None

with summary_box:
    if diab_res or derm_res:
        with st.container(border=True):
            st.markdown("**Synthèse**")
            s1, s2 = st.columns(2)
            if diab_res:
                s1.metric("Risque de diabète", diab_res["level"], f"{diab_res['prob']:.1f} %", delta_color="off")
            if derm_res:
                s2.metric("Lésion la plus probable", derm_res["name"], f"suspicion {derm_res['suspicion']:.1f} %", delta_color="off")
    else:
        st.caption("La synthèse apparaîtra ici une fois les résultats disponibles.")

st.markdown("---")
with st.expander("ℹ️ Comprendre ces résultats"):
    st.markdown("**Niveaux de risque et d'indice**\n\n- 🟢 Faible : moins de 40 %\n- 🟠 Modéré : de 40 à 75 %\n- 🔴 Élevé : au-delà de 75 %")
    if show_tab and rf_model is not None and hasattr(rf_model, "feature_importances_"):
        imp = np.array(rf_model.feature_importances_) * 100
        if len(imp) == len(FEATURES_FR):
            order = np.argsort(imp)
            fig = go.Figure(go.Bar(x=imp[order], y=[FEATURES_FR[i] for i in order], orientation="h", marker_color="#0F766E"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=30), title="Poids des paramètres dans le modèle diabète (%)")
            st.plotly_chart(fig, use_container_width=True)
    if show_img:
        st.markdown("**Types de lésions reconnus par le modèle**")
        for c in CLASS_CODES:
            st.markdown(f"- **{CLASS_INFO[c][0]}** : {CLASS_INFO[c][1]}")
    st.markdown("**Limites**\n\n" + DISCLAIMER)

st.markdown("---")
st.subheader("📄 Compte-rendu de consultation")
if st.button("Préparer le compte-rendu PDF", type="primary"):
    if not (diab_res or derm_res):
        st.warning("Aucun résultat à inclure : complétez d'abord l'examen.")
    else:
        try:
            meta = {"date": f"{datetime.now():%d/%m/%Y %H:%M}",
                    "practitioner": st.session_state.get("practitioner", ""), "route": ROUTES[intent],
                    "request": st.session_state.request}
            st.session_state.pdf = create_pdf_report(meta, diab_res, derm_res)
        except Exception as e:
            st.session_state.pdf = None
            st.error(f"Erreur lors de la génération du PDF : {e}")
if st.session_state.pdf:
    st.download_button("📥 Télécharger le compte-rendu (PDF)", data=st.session_state.pdf,
                       file_name=f"Compte_rendu_MedRoute_{datetime.now():%Y%m%d_%H%M}.pdf", mime="application/pdf")
    st.success("Compte-rendu prêt.")
