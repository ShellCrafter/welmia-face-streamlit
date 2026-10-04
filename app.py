import io
import time
import zipfile

import streamlit as st
import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Welmia Face",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOGO
# ============================================================

LOGO_PATH = "static/welmia_logo.png"

try:
    st.logo(
        LOGO_PATH,
        size="medium",
        link="https://welmia.kesug.com",
    )
except Exception:
    pass


# ============================================================
# CUSTOM CSS
# ============================================================

st.html(
    """
    <style>

    /* ========================================================
       ROOT
    ======================================================== */

    :root {
        --black: #0b0f14;
        --dark: #111820;
        --gray: #667085;
        --light-gray: #98a2b3;
        --border: rgba(15, 23, 42, 0.09);

        --blue: #2563eb;
        --blue-light: #60a5fa;
        --green: #10b981;
        --green-light: #34d399;

        --glass: rgba(255, 255, 255, 0.72);
    }


    /* ========================================================
       PAGE
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 8% 8%,
                rgba(37, 99, 235, 0.075),
                transparent 25%
            ),
            radial-gradient(
                circle at 92% 12%,
                rgba(16, 185, 129, 0.065),
                transparent 24%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(37, 99, 235, 0.045),
                transparent 30%
            ),
            #ffffff;
    }


    .main .block-container {
        max-width: 1180px;
        padding-top: 4.5rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       STREAMLIT UI
    ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       HERO
    ======================================================== */

    .hero {
        position: relative;
        text-align: center;
        padding: 35px 20px 40px;
    }


    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;

        padding: 8px 14px;

        border-radius: 999px;

        border: 1px solid rgba(37, 99, 235, 0.20);

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.055),
                rgba(16, 185, 129, 0.055)
            );

        color: #344054;

        font-size: 12px;
        font-weight: 700;

        letter-spacing: 0.12em;
        text-transform: uppercase;

        box-shadow:
            0 0 25px rgba(37, 99, 235, 0.06);
    }


    .hero-dot {
        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: #10b981;

        box-shadow:
            0 0 12px rgba(16, 185, 129, 0.65);
    }


    .hero-title {
        margin: 22px 0 0;

        font-size: clamp(48px, 8vw, 82px);

        line-height: 0.92;

        letter-spacing: -0.065em;

        font-weight: 850;

        color: #080b10;
    }


    .hero-title-gradient {
        background:
            linear-gradient(
                100deg,
                #0b0f14 10%,
                #2563eb 48%,
                #10b981 88%
            );

        -webkit-background-clip: text;
        background-clip: text;

        color: transparent;
    }


    .hero-subtitle {
        max-width: 680px;

        margin: 23px auto 0;

        color: #667085;

        font-size: 17px;

        line-height: 1.7;
    }


    .hero-meta {
        display: flex;

        justify-content: center;

        gap: 10px;

        flex-wrap: wrap;

        margin-top: 23px;
    }


    .meta-chip {
        padding: 7px 11px;

        border-radius: 9px;

        border: 1px solid rgba(15, 23, 42, 0.08);

        background: rgba(255, 255, 255, 0.72);

        color: #667085;

        font-size: 12px;

        font-weight: 600;

        box-shadow:
            0 5px 20px rgba(15, 23, 42, 0.035);
    }


    /* ========================================================
       SECTION CARD
    ======================================================== */

    .glass-card {
        position: relative;

        padding: 27px;

        border-radius: 24px;

        border: 1px solid rgba(15, 23, 42, 0.085);

        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.88),
                rgba(248, 250, 252, 0.72)
            );

        box-shadow:
            0 25px 70px rgba(15, 23, 42, 0.055),
            inset 0 1px 0 rgba(255, 255, 255, 0.95);

        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);

        overflow: hidden;
    }


    .glass-card::before {
        content: "";

        position: absolute;

        top: 0;
        left: 0;
        right: 0;

        height: 2px;

        background:
            linear-gradient(
                90deg,
                transparent,
                #2563eb,
                #10b981,
                transparent
            );

        opacity: 0.85;
    }


    .section-label {
        display: flex;

        align-items: center;

        gap: 10px;

        margin-bottom: 20px;

        font-size: 19px;

        font-weight: 750;

        color: #111827;
    }


    .section-icon {
        display: inline-flex;

        align-items: center;

        justify-content: center;

        width: 34px;
        height: 34px;

        border-radius: 10px;

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.10),
                rgba(16, 185, 129, 0.10)
            );

        border: 1px solid rgba(37, 99, 235, 0.12);

        font-size: 16px;
    }


    /* ========================================================
       BUTTONS
    ======================================================== */

    .stButton > button {

        min-height: 50px !important;

        border-radius: 14px !important;

        border: 1px solid #111827 !important;

        background:
            linear-gradient(
                100deg,
                #0b0f14,
                #111827
            ) !important;

        color: white !important;

        font-size: 15px !important;

        font-weight: 750 !important;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.12);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease !important;
    }


    .stButton > button:hover {

        transform: translateY(-2px);

        border-color: #2563eb !important;

        box-shadow:
            0 12px 30px rgba(37, 99, 235, 0.17),
            0 0 0 1px rgba(16, 185, 129, 0.12) !important;
    }


    .stButton > button:active {
        transform: translateY(0);
    }


    .stDownloadButton > button {

        min-height: 42px !important;

        border-radius: 12px !important;

        border: 1px solid rgba(37, 99, 235, 0.17) !important;

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.035),
                rgba(16, 185, 129, 0.035)
            ) !important;

        color: #1d4ed8 !important;

        font-weight: 700 !important;

        transition:
            all 0.18s ease !important;
    }


    .stDownloadButton > button:hover {

        border-color: #10b981 !important;

        background:
            linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.075),
                rgba(16, 185, 129, 0.075)
            ) !important;

        color: #047857 !important;
    }


    /* ========================================================
       SLIDER
    ======================================================== */

    div[data-testid="stSlider"] {

        padding-top: 4px;
    }


    div[data-testid="stSlider"] div[role="slider"] {

        background: #2563eb !important;

        border-color: #ffffff !important;

        box-shadow:
            0 0 0 2px rgba(37, 99, 235, 0.14),
            0 0 15px rgba(37, 99, 235, 0.25);
    }


    /* ========================================================
       SELECT
    ======================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 13px !important;

        border-color: rgba(15, 23, 42, 0.10) !important;

        background: rgba(255, 255, 255, 0.75) !important;

        transition:
            border-color 0.18s ease,
            box-shadow 0.18s ease !important;
    }


    div[data-baseweb="select"] > div:focus-within {

        border-color: rgba(37, 99, 235, 0.55) !important;

        box-shadow:
            0 0 0 3px rgba(37, 99, 235, 0.08) !important;
    }


    /* ========================================================
       NUMBER INPUT
    ======================================================== */

    div[data-testid="stNumberInput"] input {

        border-radius: 12px !important;
    }


    /* ========================================================
       IMAGE CARDS
    ======================================================== */

    div[data-testid="stImage"] {

        border-radius: 20px;

        overflow: hidden;

        background: #f8fafc;

        box-shadow:
            0 15px 40px rgba(15, 23, 42, 0.06);
    }


    div[data-testid="stImage"] img {

        border-radius: 20px;

        border: 1px solid rgba(15, 23, 42, 0.08);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease;
    }


    div[data-testid="stImage"] img:hover {

        transform: translateY(-3px);

        box-shadow:
            0 18px 45px rgba(37, 99, 235, 0.10);
    }


    /* ========================================================
       RESULT INFO
    ======================================================== */

    .result-bar {

        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 15px;

        flex-wrap: wrap;

        margin: 25px 0 18px;

        padding: 14px 17px;

        border-radius: 14px;

        border: 1px solid rgba(16, 185, 129, 0.14);

        background:
            linear-gradient(
                90deg,
                rgba(37, 99, 235, 0.035),
                rgba(16, 185, 129, 0.045)
            );

        color: #667085;

        font-size: 13px;
    }


    .result-status {

        display: flex;

        align-items: center;

        gap: 8px;

        font-weight: 700;

        color: #111827;
    }


    .status-dot {

        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #10b981;

        box-shadow:
            0 0 12px rgba(16, 185, 129, 0.65);
    }


    /* ========================================================
       EMPTY STATE
    ======================================================== */

    .empty-card {

        margin-top: 28px;

        padding: 65px 25px;

        text-align: center;

        border-radius: 24px;

        border: 1px dashed rgba(37, 99, 235, 0.20);

        background:
            linear-gradient(
                145deg,
                rgba(37, 99, 235, 0.025),
                rgba(16, 185, 129, 0.025)
            );
    }


    .empty-icon {

        font-size: 48px;

        margin-bottom: 12px;

        filter:
            drop-shadow(
                0 8px 18px
                rgba(37, 99, 235, 0.15)
            );
    }


    .empty-title {

        color: #111827;

        font-size: 19px;

        font-weight: 750;
    }


    .empty-text {

        color: #667085;

        font-size: 14px;

        margin-top: 7px;
    }


    /* ========================================================
       FOOTER
    ======================================================== */

    .footer {

        text-align: center;

        margin-top: 55px;

        padding-top: 28px;

        border-top:
            1px solid rgba(15, 23, 42, 0.06);

        color: #98a2b3;

        font-size: 12px;

        line-height: 1.8;
    }


    .footer-brand {

        color: #344054;

        font-weight: 750;
    }


    .footer-link {

        color: #2563eb;

        text-decoration: none;
    }


    /* ========================================================
       MOBILE
    ======================================================== */

    @media (max-width: 700px) {

        .main .block-container {

            padding-left: 1rem;
            padding-right: 1rem;

            padding-top: 3.2rem;
        }


        .hero {

            padding-left: 8px;
            padding-right: 8px;
        }


        .hero-title {

            font-size: 50px;
        }


        .hero-subtitle {

            font-size: 15px;

            line-height: 1.6;
        }


        .glass-card {

            padding: 19px;

            border-radius: 20px;
        }


        .result-bar {

            align-items: flex-start;

            flex-direction: column;
        }
    }

    </style>
    """
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_REPO = "Welmia/welmia-face-1.0-3.6m-base"

MODEL_FILENAME = "model.safetensors"

LATENT_DIM = 100

DEVICE = torch.device("cpu")


# ============================================================
# GENERATOR
# ============================================================

class Generator(torch.nn.Module):

    def __init__(self, nz=100, ngf=64):

        super().__init__()

        self.net = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(
                nz,
                ngf * 8,
                4,
                1,
                0,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 8
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 4
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf * 2
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf,
                ngf // 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(
                ngf // 2
            ),

            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf // 2,
                3,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.Tanh(),
        )


    def forward(self, z):

        return self.net(z)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
    )

    model = Generator()

    state_dict = load_file(
        model_path,
        device="cpu",
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# GENERATE FACES
# ============================================================

def generate_faces(
    model,
    num_images,
    seed=None,
):

    if seed is not None:

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(seed)

        z = torch.randn(
            num_images,
            LATENT_DIM,
            1,
            1,
            generator=generator,
            device=DEVICE,
        )

    else:

        z = torch.randn(
            num_images,
            LATENT_DIM,
            1,
            1,
            device=DEVICE,
        )


    with torch.inference_mode():

        output = model(z)

        output = (
            output.clamp(-1, 1) + 1
        ) / 2


    images = []


    for image_tensor in output:

        image_tensor = (
            image_tensor
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        image_array = (
            image_tensor * 255
        ).astype("uint8")

        image = Image.fromarray(
            image_array,
            mode="RGB",
        )

        images.append(image)


    return images


# ============================================================
# IMAGE TO PNG
# ============================================================

def image_to_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# CREATE ZIP
# ============================================================

def create_zip(images):

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(
        zip_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for index, image in enumerate(
            images,
            start=1,
        ):

            image_bytes = image_to_bytes(
                image
            )

            zip_file.writestr(
                f"welmia_face_{index}.png",
                image_bytes,
            )


    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">

        <div class="hero-pill">

            <span class="hero-dot"></span>

            WELMIA AI · SYNTHETIC FACES

        </div>


        <h1 class="hero-title">

            Welmia
            <span class="hero-title-gradient">
                Face
            </span>

        </h1>


        <p class="hero-subtitle">

            Generate unique synthetic human faces
            using a tiny 3.6M parameter DCGAN.
            Fast, lightweight and completely synthetic.

        </p>


        <div class="hero-meta">

            <div class="meta-chip">
                ⚡ CPU Optimized
            </div>

            <div class="meta-chip">
                🧠 3.6M Parameters
            </div>

            <div class="meta-chip">
                🖼️ 128 × 128
            </div>

            <div class="meta-chip">
                🔒 Synthetic Only
            </div>

        </div>

    </div>
    """
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

st.html(
    """
    <div class="glass-card">

        <div class="section-label">

            <span class="section-icon">
                ✨
            </span>

            Generation

        </div>

    </div>
    """
)


col1, col2, col3 = st.columns(
    [1, 1, 1],
    gap="large",
)


with col1:

    num_images = st.slider(
        "Number of faces",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
    )


with col2:

    generation_mode = st.selectbox(
        "Generation mode",
        [
            "Random",
            "Fixed seed",
        ],
        index=0,
    )


with col3:

    if generation_mode == "Fixed seed":

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999999,
            value=42,
            step=1,
        )

    else:

        seed = None

        st.markdown(
            "🎲 **Random mode**  \n"
            "Every generation uses fresh noise."
        )


st.write("")


generate_button = st.button(
    "✨  Generate Faces",
    use_container_width=True,
)


st.caption(
    "Random mode creates new faces every time. "
    "Fixed seed mode makes results reproducible."
)


# ============================================================
# LOAD MODEL
# ============================================================

model = load_model()


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    start_time = time.perf_counter()


    with st.spinner(
        "Generating synthetic faces..."
    ):

        images = generate_faces(
            model=model,
            num_images=num_images,
            seed=seed,
        )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    st.session_state["images"] = images

    st.session_state[
        "generation_time"
    ] = elapsed

    st.session_state[
        "generation_seed"
    ] = seed


# ============================================================
# RESULTS
# ============================================================

if "images" in st.session_state:

    images = st.session_state["images"]

    generation_time = (
        st.session_state.get(
            "generation_time",
            0,
        )
    )

    generation_seed = (
        st.session_state.get(
            "generation_seed",
            None,
        )
    )


    if generation_seed is None:

        seed_text = "Random seed"

    else:

        seed_text = (
            f"Fixed seed: "
            f"{generation_seed}"
        )


    st.html(
        f"""
        <div class="result-bar">

            <div class="result-status">

                <span class="status-dot"></span>

                Generation complete

            </div>

            <div>
                {len(images)} faces
                &nbsp;·&nbsp;
                {generation_time:.2f}s
                &nbsp;·&nbsp;
                {seed_text}
            </div>

        </div>
        """
    )


    # ========================================================
    # GALLERY
    # ========================================================

    columns_per_row = 4


    for row_start in range(
        0,
        len(images),
        columns_per_row,
    ):

        row_images = images[
            row_start:
            row_start + columns_per_row
        ]


        columns = st.columns(
            columns_per_row,
            gap="medium",
        )


        for local_index, image in enumerate(
            row_images
        ):

            absolute_index = (
                row_start
                + local_index
                + 1
            )


            with columns[local_index]:

                st.image(
                    image,
                    use_container_width=True,
                )


                image_bytes = (
                    image_to_bytes(
                        image
                    )
                )


                st.download_button(
                    label="↓  PNG",
                    data=image_bytes,
                    file_name=(
                        f"welmia_face_"
                        f"{absolute_index}.png"
                    ),
                    mime="image/png",
                    use_container_width=True,
                    key=(
                        f"download_"
                        f"{absolute_index}"
                    ),
                )


    # ========================================================
    # DOWNLOAD ALL
    # ========================================================

    st.write("")


    zip_bytes = create_zip(
        images
    )


    st.download_button(
        label="📦  Download All Faces",
        data=zip_bytes,
        file_name="welmia_faces.zip",
        mime="application/zip",
        use_container_width=True,
        key="download_all_faces",
    )


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.html(
        """
        <div class="empty-card">

            <div class="empty-icon">
                🧬
            </div>

            <div class="empty-title">
                Ready to generate
            </div>

            <div class="empty-text">
                Choose your settings and create
                a new set of synthetic faces.
            </div>

        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div class="footer">

        <span class="footer-brand">
            Welmia Face 1.0
        </span>

        · 3.6M parameter synthetic face generator

        <br>

        Synthetic faces only · Not real individuals

        <br>

        <a
            class="footer-link"
            href="https://welmia.kesug.com"
            target="_blank"
        >
            welmia.kesug.com
        </a>

    </div>
    """
)
