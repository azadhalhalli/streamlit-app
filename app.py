"""
Twitter Duygu Analizi — Streamlit Arayüzü
SVM | SMOTE  &  LGBM | Class Weight modelleri
"""

import streamlit as st
import pickle
import re
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Sayfa Ayarları ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Metin Duygu Duyarlılık Analizi",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Tema & CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --dark:   #0D1117;
    --panel:  #161B22;
    --border: #30363D;
    --text:   #C9D1D9;
    --bad:    #E74C3C;
    --good:   #2ECC71;
    --neutral:#3498DB;
    --accent: #F39C12;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--dark);
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--panel) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header gradient bar */
.hero-bar {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 40%, #1a2230 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-bar::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--bad), var(--neutral), var(--good));
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem; font-weight: 700;
    color: #fff; margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 0.88rem; color: #8b949e;
    margin: 0; letter-spacing: 0.3px;
}

/* Sentiment result card */
.result-card {
    border-radius: 10px;
    padding: 24px 28px;
    margin: 16px 0;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card.bad    { background: rgba(231,76,60,0.1);  border-color: rgba(231,76,60,0.4); }
.result-card.good   { background: rgba(46,204,113,0.1); border-color: rgba(46,204,113,0.4);}
.result-card.neutral{ background: rgba(52,152,219,0.1); border-color: rgba(52,152,219,0.4);}

.result-emoji { font-size: 2.8rem; margin-bottom: 8px; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 2px;
}
.result-label.bad     { color: var(--bad); }
.result-label.good    { color: var(--good); }
.result-label.neutral { color: var(--neutral); }
.result-conf { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

/* Probability bars */
.prob-bar-container { margin: 8px 0; }
.prob-label { font-size: 0.82rem; color: #8b949e; margin-bottom: 3px; }
.prob-track {
    background: var(--border);
    border-radius: 99px;
    height: 10px; overflow: hidden;
}
.prob-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s ease;
}
.prob-fill.bad     { background: linear-gradient(90deg, #c0392b, #e74c3c); }
.prob-fill.good    { background: linear-gradient(90deg, #27ae60, #2ecc71); }
.prob-fill.neutral { background: linear-gradient(90deg, #2980b9, #3498db); }

/* Model badge */
.model-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 6px;
}
.model-badge.svm  { background: rgba(231,76,60,0.2);  color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }
.model-badge.lgbm { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4);}

/* Metric cards */
.metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700; color: #fff;
}
.metric-label { font-size: 0.75rem; color: #8b949e; margin-top: 3px; }

/* Text area */
textarea {
    background-color: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
}
textarea:focus { border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(46,160,67,0.35) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--panel);
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8b949e;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: var(--dark) !important;
    color: #fff !important;
}

/* Info/warning boxes */
.info-box {
    background: rgba(52,152,219,0.1);
    border-left: 3px solid var(--neutral);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.85rem;
    color: #8b949e;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* Select box */
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Renk paleti (matplotlib için) ─────────────────────────────────────────────
DARK   = '#0D1117'
PANEL  = '#161B22'
BORDER = '#30363D'
TEXT   = '#C9D1D9'
COLORS = {'bad': '#E74C3C', 'good': '#2ECC71', 'neutral': '#3498DB'}
EMOJI_MAP = {'bad': '😠', 'good': '😊', 'neutral': '😐'}
CLASS_NAMES = ['bad', 'good', 'neutral']

plt.rcParams.update({
    'figure.facecolor': DARK, 'axes.facecolor': PANEL,
    'axes.edgecolor': BORDER, 'axes.labelcolor': TEXT,
    'xtick.color': TEXT, 'ytick.color': TEXT, 'text.color': TEXT,
    'grid.color': BORDER, 'grid.linestyle': '--',
    'grid.linewidth': 0.5, 'grid.alpha': 0.5,
})

FEAT_COLS = [
    'tweet_length', 'word_count', 'char_per_word',
    'has_hashtag', 'has_mention', 'has_url',
    'has_exclaim', 'has_question',
    'hashtag_count', 'mention_count',
    'uppercase_ratio', 'emoji_count', 'has_emoji',
]

# ── Model Yükleme ─────────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_models():
    try:
        import emoji as emoji_lib
        with open('tfidf.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        
        # results.pkl'i Hugging Face'ten indir
        results_path = hf_hub_download(
            repo_id="azadhalhalli/my-results",
            filename="results.pkl"
        )
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        return tfidf, results, emoji_lib, None
    except FileNotFoundError as e:
        return None, None, None, str(e)
    except Exception as e:
        return None, None, None, str(e)

tfidf, results, emoji_lib, load_error = load_models()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_light(text: str, emoji_lib) -> str:
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', 'user', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = emoji_lib.demojize(text, delimiters=(' ', ' '))
    text = text.replace('\\n', ' ').replace('\\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text: str, emoji_lib):
    feats = {
        'tweet_length':   len(text),
        'word_count':     len(text.split()),
        'char_per_word':  round(len(text) / max(len(text.split()), 1), 2),
        'has_hashtag':    int('#' in text),
        'has_mention':    int('@' in text),
        'has_url':        int(bool(re.search(r'http|www', text, re.I))),
        'has_exclaim':    int('!' in text),
        'has_question':   int('?' in text),
        'hashtag_count':  text.count('#'),
        'mention_count':  text.count('@'),
        'uppercase_ratio': sum(c.isupper() for c in text if c.isalpha()) /
                           max(len([c for c in text if c.isalpha()]), 1),
        'emoji_count':    len(emoji_lib.emoji_list(text)),
        'has_emoji':      int(len(emoji_lib.emoji_list(text)) > 0),
    }
    return np.array([feats[c] for c in FEAT_COLS], dtype=float)

def predict_tweet(text: str, model, tfidf, emoji_lib):
    clean = preprocess_light(text, emoji_lib)
    tfidf_vec = tfidf.transform([clean])
    feat_vec  = extract_features(text, emoji_lib).reshape(1, -1)
    X = sp.hstack([tfidf_vec, sp.csr_matrix(feat_vec)])
    pred  = model.predict(X)[0]
    label = CLASS_NAMES[pred]
    try:
        proba = model.predict_proba(X)[0]
    except:
        proba = None
    return label, proba, clean, feat_vec[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px 0;'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;font-weight:700;color:#fff;'>
            📝 SentimentAI
        </div>
        <div style='font-size:0.75rem;color:#8b949e;margin-top:2px;'>Metin · Duygu Duyarlılık Analizi</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Model Seçimi**")

    model_choice = st.selectbox(
        "Hangi model ile analiz yapılsın?",
        ["Her İkisi", "SVM | SMOTE", "LGBM | Class Weight"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**📊 Model Performansı**")

    if results:
        svm_f1  = results.get('SVM  | SMOTE', {}).get('macro_f1', 0)
        lgbm_f1 = results.get('LGBM | Class Weight', {}).get('macro_f1', 0)

        st.markdown(f"""
        <div class='metric-card' style='margin-bottom:8px;'>
            <div style='font-size:0.7rem;color:#e74c3c;font-family:Space Mono,monospace;margin-bottom:4px;'>SVM · SMOTE</div>
            <div class='metric-value' style='color:#e74c3c;font-size:1.3rem;'>{svm_f1:.4f}</div>
            <div class='metric-label'>Macro-F1</div>
        </div>
        <div class='metric-card'>
            <div style='font-size:0.7rem;color:#2ecc71;font-family:Space Mono,monospace;margin-bottom:4px;'>LGBM · Class Weight</div>
            <div class='metric-value' style='color:#2ecc71;font-size:1.3rem;'>{lgbm_f1:.4f}</div>
            <div class='metric-label'>Macro-F1</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Modeller yüklenmedi")

    st.markdown("---")
    st.markdown("**📋 Sınıflar**")
    for cls, color in COLORS.items():
        st.markdown(
            f"<span style='display:inline-flex;align-items:center;gap:8px;margin:3px 0;'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;'></span>"
            f"<span style='font-size:0.85rem;'>{cls.upper()}</span></span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#484f58;line-height:1.6;'>
        <b style='color:#8b949e;'>Veri Seti:</b> 217,622 metin<br>
        <b style='color:#8b949e;'>Features:</b> TF-IDF (5K) + FE (13)<br>
        <b style='color:#8b949e;'>Stratejiler:</b> SMOTE, Class Weight<br>
    </div>
    """, unsafe_allow_html=True)

# ── Ana İçerik ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-bar'>
    <div class='hero-title'>📝 Metin Duygu Duyarlılık Analizi</div>
    <div class='hero-sub'>7 Model × 4 Dengesizlik Stratejisi &nbsp;·&nbsp; TF-IDF + Feature Engineering &nbsp;·&nbsp; XAI/SHAP</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Canlı Analiz", "📈 Model Karşılaştırması", "ℹ️ Hakkında"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Canlı Analiz
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if load_error:
        st.error(f"❌ Modeller yüklenemedi: `{load_error}`")
        st.markdown("""
        <div class='info-box'>
            <b>Çözüm:</b> <code>tfidf.pkl</code> ve <code>results.pkl</code> dosyalarının
            bu script ile aynı klasörde olduğundan emin olun.<br><br>
            Notebook'ta şu kodla kaydedin:
            <pre style='margin:8px 0 0 0;background:#0d1117;padding:10px;border-radius:6px;'>
import pickle
with open('tfidf.pkl', 'wb') as f: pickle.dump(tfidf, f)
with open('results.pkl', 'wb') as f: pickle.dump(results, f)</pre>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_input, col_result = st.columns([1.1, 1], gap="large")

        with col_input:
            st.markdown("##### ✍️ Metin Girin")
            example_tweets = [
                "Custom text...",
                "I absolutely love this new restaurant, the food was amazing and the service was excellent! 🚀",
                "The package arrived damaged and the customer service was unhelpful. Very disappointing experience.",
                "The conference will be held next week at the main hall. Registration is open.",
                "Worst purchase I have ever made. Completely useless, waste of money! 😡",
                "The weather report indicates mild temperatures for the upcoming weekend.",
            ]
            selected_example = st.selectbox(
                "Örnek metin seç (veya aşağıya yaz):",
                example_tweets,
                label_visibility="visible",
            )

            tweet_text = st.text_area(
                "Metin",
                value="" if selected_example == "Custom text..." else selected_example,
                height=120,
                max_chars=5000,
                placeholder="Analiz etmek istediğiniz metni buraya girin... (en fazla 5000 karakter)",
                label_visibility="collapsed",
            )

            char_count = len(tweet_text)
            word_count = len(tweet_text.split()) if tweet_text.strip() else 0
            st.markdown(
                f"<div style='font-size:0.75rem;color:#484f58;text-align:right;margin-top:-8px;'>"
                f"{char_count} / 5000 karakter · {word_count} kelime</div>",
                unsafe_allow_html=True,
            )

            analyze_btn = st.button("🔍  ANALİZ ET", use_container_width=True)

            # Feature Engineering göster
            if tweet_text.strip():
                with st.expander("🔧 Feature Engineering Değerleri", expanded=False):
                    feats = extract_features(tweet_text, emoji_lib)
                    feat_dict = dict(zip(FEAT_COLS, feats))
                    cols = st.columns(2)
                    for idx, (k, v) in enumerate(feat_dict.items()):
                        with cols[idx % 2]:
                            st.markdown(
                                f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
                                f"border-bottom:1px solid {BORDER};font-size:0.8rem;'>"
                                f"<span style='color:#8b949e;'>{k}</span>"
                                f"<span style='font-family:Space Mono,monospace;color:#c9d1d9;'>{v:.3f}</span></div>",
                                unsafe_allow_html=True,
                            )

        with col_result:
            st.markdown("##### 📊 Sonuçlar")

            if analyze_btn and tweet_text.strip():
                models_to_run = []
                if model_choice in ["Her İkisi", "SVM | SMOTE"]:
                    models_to_run.append(("SVM | SMOTE", "svm", results.get('SVM  | SMOTE', {}).get('model')))
                if model_choice in ["Her İkisi", "LGBM | Class Weight"]:
                    models_to_run.append(("LGBM | Class Weight", "lgbm", results.get('LGBM | Class Weight', {}).get('model')))

                for model_name, model_key, model in models_to_run:
                    if model is None:
                        st.warning(f"{model_name} bulunamadı")
                        continue

                    with st.spinner(f"{model_name} analiz ediyor..."):
                        label, proba, clean_text, feats = predict_tweet(
                            tweet_text, model, tfidf, emoji_lib
                        )

                    badge_cls = "svm" if model_key == "svm" else "lgbm"
                    color = COLORS[label]

                    st.markdown(
                        f"<span class='model-badge {badge_cls}'>{model_name}</span>",
                        unsafe_allow_html=True,
                    )

                    st.markdown(f"""
                    <div class='result-card {label}'>
                        <div class='result-emoji'>{EMOJI_MAP[label]}</div>
                        <div class='result-label {label}'>{label.upper()}</div>
                        {'<div class="result-conf">Macro-F1: ' + f'{max(proba):.4f}</div>' if proba is not None else ''}
                    </div>
                    """, unsafe_allow_html=True)

                    if proba is not None:
                        st.markdown("**Olasılık Dağılımı**")
                        for i, cls in enumerate(CLASS_NAMES):
                            pct = proba[i] * 100
                            cls_color = COLORS[cls]
                            st.markdown(f"""
                            <div class='prob-bar-container'>
                                <div class='prob-label'>{cls.upper()} — {pct:.1f}%</div>
                                <div class='prob-track'>
                                    <div class='prob-fill {cls}' style='width:{pct}%;'></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    if len(models_to_run) > 1:
                        st.markdown("---")

                # Temizlenmiş metni göster
                with st.expander("🧹 Temizlenmiş Metin", expanded=False):
                    st.markdown(
                        f"<div style='background:{PANEL};border:1px solid {BORDER};border-radius:6px;"
                        f"padding:12px 16px;font-size:0.85rem;color:{TEXT};font-style:italic;'>"
                        f"{clean_text}</div>",
                        unsafe_allow_html=True,
                    )

            elif analyze_btn:
                st.warning("⚠️ Lütfen bir metin girin.")
            else:
                st.markdown("""
                <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:200px;border:1px dashed #30363D;border-radius:10px;'>
                    <div style='font-size:2.5rem;margin-bottom:8px;opacity:0.4;'>🔍</div>
                    <div style='font-size:0.85rem;color:#484f58;'>Metin girin ve "ANALİZ ET" butonuna basın</div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Karşılaştırması
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not results:
        st.error("Modeller yüklenmedi — `results.pkl` bulunamadı.")
    else:
        st.markdown("#### 🏆 Tüm Modeller — Performans Tablosu")

        model_data = []
        for key, val in results.items():
            model_data.append({
                'Model': key,
                'Macro-F1': round(val.get('macro_f1', 0), 4),
                'Balanced Acc': round(val.get('bal_acc', 0), 4),
                'Süre (s)': round(val.get('time', 0), 1),
            })

        import pandas as pd
        df_models = pd.DataFrame(model_data).sort_values('Macro-F1', ascending=False).reset_index(drop=True)

        def highlight_best(s):
            is_max = s == s.max()
            return ['background-color: rgba(46,204,113,0.2); color: #2ecc71; font-weight: bold;'
                    if v else '' for v in is_max]

        st.dataframe(
            df_models.style.apply(highlight_best, subset=['Macro-F1']),
            use_container_width=True, hide_index=True,
        )

        # ── F1 Bar Chart ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📊 Macro-F1 Karşılaştırması")

        df_sorted = df_models.sort_values('Macro-F1', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.55)), facecolor=DARK)
        ax.set_facecolor(PANEL)

        bar_colors = []
        for name in df_sorted['Model']:
            if 'SVM' in name:   bar_colors.append('#E74C3C')
            elif 'LGBM' in name: bar_colors.append('#2ECC71')
            elif 'RF' in name:   bar_colors.append('#9B59B6')
            elif 'LR' in name:   bar_colors.append('#3498DB')
            elif 'XGB' in name:  bar_colors.append('#F39C12')
            elif 'KNN' in name:  bar_colors.append('#1ABC9C')
            else:                bar_colors.append('#95A5A6')

        bars = ax.barh(df_sorted['Model'], df_sorted['Macro-F1'],
                       color=bar_colors, edgecolor=DARK, linewidth=0.8, alpha=0.9)

        for b, v in zip(bars, df_sorted['Macro-F1']):
            ax.text(v + 0.003, b.get_y() + b.get_height()/2,
                    f'{v:.4f}', va='center', color='white', fontsize=9)

        for sp in ax.spines.values(): sp.set_color(BORDER)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.set_xlabel('Macro-F1', color=TEXT)
        ax.set_title('Model Performans Karşılaştırması', color='white', fontsize=12,
                     fontweight='bold', pad=10)
        ax.set_xlim(df_sorted['Macro-F1'].min() - 0.02, df_sorted['Macro-F1'].max() + 0.04)
        ax.grid(color=BORDER, linestyle='--', linewidth=0.5, alpha=0.5, axis='x')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── Confusion Matrix ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🔲 Confusion Matrix")

        cm_model_key = st.selectbox(
            "Confusion matrix için model seç:",
            list(results.keys()),
        )

        cm_data = results[cm_model_key].get('cm')
        if cm_data is not None:
            import numpy as np
            fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK)
            ax.set_facecolor(PANEL)

            im = ax.imshow(cm_data, cmap='Blues', aspect='auto')

            n = len(CLASS_NAMES)
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels([c.upper() for c in CLASS_NAMES], color=TEXT, fontsize=10)
            ax.set_yticklabels([c.upper() for c in CLASS_NAMES], color=TEXT, fontsize=10)

            thresh = cm_data.max() / 2.0
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{cm_data[i, j]:,}', ha='center', va='center',
                            color='white' if cm_data[i, j] > thresh else TEXT, fontsize=11, fontweight='bold')

            ax.set_xlabel('Tahmin', color=TEXT, fontsize=11)
            ax.set_ylabel('Gerçek', color=TEXT, fontsize=11)
            ax.set_title(f'{cm_model_key} — Confusion Matrix', color='white',
                         fontsize=11, fontweight='bold', pad=10)
            for sp in ax.spines.values(): sp.set_color(BORDER)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ── Per-class F1 ───────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 🎯 Sınıf Bazlı F1 Skoru")

        report = results[cm_model_key].get('report', {})
        if report:
            per_class = {cls: report.get(cls, {}).get('f1-score', 0) for cls in CLASS_NAMES}
            fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=DARK)
            ax.set_facecolor(PANEL)
            cls_colors = [COLORS[c] for c in CLASS_NAMES]
            bars = ax.bar([c.upper() for c in CLASS_NAMES],
                          [per_class[c] for c in CLASS_NAMES],
                          color=cls_colors, edgecolor=DARK, linewidth=1.2, width=0.5)
            for b, v in zip(bars, per_class.values()):
                ax.text(b.get_x() + b.get_width()/2, v + 0.008,
                        f'{v:.4f}', ha='center', va='bottom', color='white',
                        fontsize=9, fontweight='bold')
            for sp in ax.spines.values(): sp.set_color(BORDER)
            ax.tick_params(colors=TEXT)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('F1-Score', color=TEXT)
            ax.set_title(f'{cm_model_key} — Sınıf F1 Skorları',
                         color='white', fontsize=11, fontweight='bold', pad=8)
            ax.grid(color=BORDER, linestyle='--', linewidth=0.5, alpha=0.5, axis='y')
            ax.set_axisbelow(True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Hakkında
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎓 Proje Hakkında
        Bu uygulama bir **Bitirme Tezi** kapsamında geliştirilen metin duygu analizi
        sisteminin interaktif arayüzüdür.

        **Pipeline:**
        > Ham Metin → Ön İşleme → TF-IDF + Feature Engineering → Model → Tahmin

        ---
        ### 📊 Veri Seti
        | Özellik | Değer |
        |---------|-------|
        | Toplam metin | 217,622 |
        | Sınıflar | bad / good / neutral |
        | TF-IDF features | 5,000 |
        | FE features | 13 |
        | Train/Test | %80 / %20 |

        ---
        ### ⚙️ Dengesizlik Stratejileri
        - **Baseline** — Düzeltme yok
        - **Class Weighting** — Azınlık sınıfa daha yüksek ağırlık
        - **SMOTE** — Sentetik azınlık örneği üretimi
        - **SMOTE + Tomek Links** — SMOTE + gürültü temizleme
        """)

    with col2:
        st.markdown("""
        ### 🤖 Modeller
        | Model | Strateji |
        |-------|----------|
        | Naive Bayes | Baseline |
        | KNN | Baseline |
        | **SVM** | **SMOTE** ⭐ |
        | Logistic Regression | Class Weight |
        | Random Forest | Class Weight |
        | XGBoost | SMOTE / SMOTE+Tomek |
        | **LightGBM** | **Class Weight** ⭐ |

        ---
        ### 🔧 Feature Engineering (13 adet)
        """)
        for i, feat in enumerate(FEAT_COLS):
            st.markdown(
                f"<span style='font-family:Space Mono,monospace;font-size:0.8rem;"
                f"background:{PANEL};border:1px solid {BORDER};padding:2px 8px;"
                f"border-radius:4px;margin:2px;display:inline-block;'>{feat}</span>",
                unsafe_allow_html=True,
            )

        st.markdown("""
        ---
        ### 🏆 En İyi Modeller
        - **En yüksek Macro-F1:** SVM | SMOTE (0.7866)
        - **En iyi XAI:** LGBM | Class Weight (TreeExplainer)
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#484f58;font-size:0.8rem;padding:16px 0;'>
        Metin Duygu Duyarlılık Analizi · Bitirme Tezi &nbsp;·&nbsp;
        <span style='font-family:Space Mono,monospace;'>TF-IDF + Feature Engineering + SHAP</span>
    </div>
    """, unsafe_allow_html=True)
