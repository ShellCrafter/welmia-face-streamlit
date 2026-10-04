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

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(0, 0, 0, 0.035),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 90%,
                rgba(0, 0, 0, 0.025),
                transparent 30%
            ),
            #ffffff;
        color: #111111;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }


    /* --------------------------------------------------------
       HIDE DEFAULT STREAMLIT ELEMENTS
    -------------------------------------------------------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* --------------------------------------------------------
       HERO
    -------------------------------------------------------- */

    .welmia-hero {
        text-align: center;
        padding: 30px 20px 25px 20px;
        margin-bottom: 25px;
    }

    .welmia-badge {
        display: inline-block;
        padding: 7px 13px;
        border: 1px solid rgba(0, 0, 0, 0.10);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #555555;
        margin-bottom: 15px;
    }

    .welmia-title {
        font-size: clamp(42px, 7vw, 72px);
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: -0.055em;
        margin: 0;
        color: #080808;
    }

    .welmia-subtitle {
        max-width: 650px;
        margin: 18px auto 0 auto;
        font-size: 17px;
        line-height: 1.6;
        color: #666666;
    }


    /* --------------------------------------------------------
       GLASS CARD
    -------------------------------------------------------- */

    .glass-card {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(0, 0, 0, 0.09);
        border-radius: 24px;
        padding: 24px;
        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.055),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 14px !important;
        border: 1px solid #111111 !important;
        background: #111111 !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        min-height: 48px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: #2a2a2a !important;
        border-color: #2a2a2a !important;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        border-radius: 12px !important;
        border: 1px solid rgba(0, 0, 0, 0.12) !important;
        background: rgba(255, 255, 255, 0.85) !important;
        color: #111111 !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover {
        border-color: #111111 !important;
        background: #ffffff !important;
    }


    /* --------------------------------------------------------
       INPUTS
    -------------------------------------------------------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 12px !important;
    }


    /* --------------------------------------------------------
       IMAGE
    -------------------------------------------------------- */

    div[data-testid="stImage"] img {
        border-radius: 18px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.07);
    }


    /* --------------------------------------------------------
       INFO TEXT
    -------------------------------------------------------- */

    .small-info {
        text-align: center;
        color: #777777;
        font-size: 13px;
        margin-top: 15px;
    }

    .generation-info {
        text-align: center;
        color: #555555;
        font-size: 14px;
        margin: 12px 0 25px 0;
    }


    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .welmia-footer {
        text-align: center;
        padding-top: 50px;
        color: #888888;
        font-size: 13px;
    }

    .welmia-footer strong {
        color: #333333;
    }


    /* --------------------------------------------------------
       MOBILE
    -------------------------------------------------------- */

    @media (max-width: 700px) {

        .main .block-container {
            padding-top: 2.5rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .welmia-title {
            font-size: 46px;
        }

        .welmia-subtitle {
            font-size: 15px;
        }

        .glass-card {
            padding: 18px;
            border-radius: 20px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
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

            torch.nn.BatchNorm2d(ngf * 8),

            torch.nn.ReLU(True),


            torch.nn.ConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(ngf * 4),

            torch.nn.ReLU(True),


            torch.nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(ngf * 2),

            torch.nn.ReLU(True),


            torch.nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(ngf),

            torch.nn.ReLU(True),


            torch.nn.ConvTranspose2d(
                ngf,
                ngf // 2,
                4,
                2,
                1,
                bias=False,
            ),

            torch.nn.BatchNorm2d(ngf // 2),

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

    model.load_state_dict(state_dict)

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
            output.clamp(-1, 1)
            + 1
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
# IMAGE TO PNG BYTES
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

        for index, image in enumerate(images, start=1):

            image_bytes = image_to_bytes(image)

            zip_file.writestr(
                f"welmia_face_{index}.png",
                image_bytes,
            )

    zip_buffer.seek(0)

    return zip_buffer.getvalue()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="welmia-hero">

        <div class="welmia-badge">
            WELMIA AI · SYNTHETIC FACES
        </div>

        <h1 class="welmia-title">
            Welmia Face
        </h1>

        <p class="welmia-subtitle">
            Generate unique synthetic human faces using
            a tiny 3.6M parameter DCGAN.
            Fast, lightweight and completely synthetic.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SETTINGS CARD
# ============================================================

st.markdown(
    '<div class="glass-card">',
    unsafe_allow_html=True,
)

st.markdown("### Generation")

col1, col2, col3 = st.columns(3)


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
            """
            <div style="
                padding-top: 8px;
                color: #666;
                font-size: 14px;
            ">
                🎲 A new random face set every time
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown("<br>", unsafe_allow_html=True)


generate_button = st.button(
    "✨ Generate Faces",
    use_container_width=True,
)


st.markdown(
    """
    <div class="small-info">
        Random mode creates new faces every generation.
        Fixed seed mode reproduces the same results.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

model = load_model()


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    start_time = time.perf_counter()

    with st.spinner("Generating synthetic faces..."):

        images = generate_faces(
            model=model,
            num_images=num_images,
            seed=seed,
        )

    elapsed = time.perf_counter() - start_time

    st.session_state["images"] = images
    st.session_state["generation_time"] = elapsed
    st.session_state["generation_seed"] = seed


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "images" in st.session_state:

    images = st.session_state["images"]

    generation_time = st.session_state.get(
        "generation_time",
        0,
    )

    generation_seed = st.session_state.get(
        "generation_seed",
        None,
    )


    if generation_seed is None:

        seed_text = "Random seed"

    else:

        seed_text = f"Seed: {generation_seed}"


    st.markdown(
        f"""
        <div class="generation-info">
            Generated <strong>{len(images)}</strong> synthetic faces
            · {generation_time:.2f}s
            · {seed_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # GALLERY
    # --------------------------------------------------------

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


                image_bytes = image_to_bytes(
                    image
                )


                st.download_button(
                    label="↓ PNG",
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


    # --------------------------------------------------------
    # DOWNLOAD ALL
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    zip_bytes = create_zip(images)

    st.download_button(
        label="📦 Download All Faces",
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

    st.markdown(
        """
        <div class="glass-card" style="
            text-align: center;
            margin-top: 30px;
            padding: 55px 25px;
        ">

            <div style="
                font-size: 52px;
                margin-bottom: 12px;
            ">
                🧬
            </div>

            <h3 style="
                margin: 0;
                color: #111;
            ">
                Ready to generate
            </h3>

            <p style="
                color: #777;
                margin-top: 10px;
            ">
                Choose your settings and create
                a new set of synthetic faces.
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="welmia-footer">

        <strong>Welmia Face 1.0</strong>
        · 3.6M parameter synthetic face generator

        <br><br>

        Synthetic faces only · Not real individuals

    </div>
    """,
    unsafe_allow_html=True,
)
