import io
import math
import random

import streamlit as st
import torch
from PIL import Image
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Welmia Face 1.0",
    page_icon="🧬",
    layout="wide",
)

MODEL_REPO = "Welmia/welmia-face-1.0-3.6m-base"
MODEL_FILENAME = "model.safetensors"

DEVICE = torch.device("cpu")
LATENT_DIM = 100


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 1.1rem;
            opacity: 0.75;
            margin-bottom: 2rem;
        }

        .info-box {
            padding: 1rem;
            border-radius: 12px;
            background: rgba(128, 128, 128, 0.10);
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERATOR MODEL
# Exact architecture from Welmia Face 1.0 README
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
                bias=False
            ),

            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 8,
                ngf * 4,
                4,
                2,
                1,
                bias=False
            ),

            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                4,
                2,
                1,
                bias=False
            ),

            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                4,
                2,
                1,
                bias=False
            ),

            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf,
                ngf // 2,
                4,
                2,
                1,
                bias=False
            ),

            torch.nn.BatchNorm2d(ngf // 2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(
                ngf // 2,
                3,
                4,
                2,
                1,
                bias=False
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

    model = Generator(
        nz=LATENT_DIM,
        ngf=64,
    )

    state_dict = load_file(
        model_path,
        device="cpu",
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# GENERATE FACES
# ============================================================

@torch.inference_mode()
def generate_faces(model, num_images, seed=None):

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

    images = model(z)

    # Convert from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0

    images = images.clamp(
        0.0,
        1.0,
    )

    # Convert CHW -> HWC
    images = images.permute(
        0,
        2,
        3,
        1,
    )

    # Convert to uint8
    images = (
        images.cpu().numpy() * 255
    ).astype("uint8")

    return images


# ============================================================
# CONVERT NUMPY IMAGE TO PIL
# ============================================================

def numpy_to_pil(image_array):

    return Image.fromarray(
        image_array,
        mode="RGB",
    )


# ============================================================
# CREATE ZIP
# ============================================================

def create_zip(images):

    import zipfile

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for index, image_array in enumerate(images):

            image = numpy_to_pil(
                image_array
            )

            image_buffer = io.BytesIO()

            image.save(
                image_buffer,
                format="PNG",
            )

            zip_file.writestr(
                f"welmia_face_{index + 1}.png",
                image_buffer.getvalue(),
            )

    buffer.seek(0)

    return buffer


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 Welmia Face 1.0</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        3.6M parameter synthetic face generator powered by a tiny DCGAN.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Generation")

    num_images = st.slider(
        "Number of faces",
        min_value=1,
        max_value=50,
        value=8,
        step=1,
    )

    use_seed = st.checkbox(
        "Use fixed seed",
        value=True,
    )

    if use_seed:

        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
        )

    else:

        seed = None

    st.divider()

    st.markdown(
        """
        **Model**

        • 3,608,006 parameters  
        • 128 × 128 resolution  
        • Latent dimension: 100  
        • CPU-friendly  
        • SafeTensors checkpoint  
        """
    )

    st.divider()

    st.caption(
        "Welmia Face 1.0 — 3.6M Base"
    )

    st.caption(
        "Synthetic faces only."
    )


# ============================================================
# MODEL LOADING
# ============================================================

with st.spinner(
    "Loading Welmia Face model..."
):

    try:

        model = load_model()

    except Exception as error:

        st.error(
            "Failed to load the Welmia Face model."
        )

        st.exception(error)

        st.stop()


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "🧬 Generate Faces",
    type="primary",
    use_container_width=True,
)


# ============================================================
# GENERATION
# ============================================================

if generate_button:

    with st.spinner(
        f"Generating {num_images} face"
        + ("s..." if num_images != 1 else "...")
    ):

        images = generate_faces(
            model=model,
            num_images=num_images,
            seed=seed,
        )

    st.session_state["generated_images"] = images

    st.session_state["generated_seed"] = seed


# ============================================================
# DISPLAY GENERATED IMAGES
# ============================================================

if "generated_images" in st.session_state:

    images = st.session_state[
        "generated_images"
    ]

    st.divider()

    st.subheader(
        f"✨ Generated {len(images)} face"
        + ("s" if len(images) != 1 else "")
    )

    # Determine grid columns
    columns = min(5, len(images))

    rows = math.ceil(
        len(images) / columns
    )

    image_index = 0

    for _ in range(rows):

        cols = st.columns(
            columns
        )

        for col in cols:

            if image_index >= len(images):
                break

            image = numpy_to_pil(
                images[image_index]
            )

            with col:

                st.image(
                    image,
                    use_container_width=True,
                )

                image_buffer = io.BytesIO()

                image.save(
                    image_buffer,
                    format="PNG",
                )

                st.download_button(
                    label="⬇️ PNG",
                    data=image_buffer.getvalue(),
                    file_name=(
                        f"welmia_face_"
                        f"{image_index + 1}.png"
                    ),
                    mime="image/png",
                    key=(
                        f"download_"
                        f"{image_index}"
                    ),
                    use_container_width=True,
                )

            image_index += 1


    # ========================================================
    # DOWNLOAD ALL
    # ========================================================

    st.divider()

    zip_buffer = create_zip(
        images
    )

    st.download_button(
        label="📦 Download All Faces (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="welmia_faces.zip",
        mime="application/zip",
        use_container_width=True,
    )


# ============================================================
# INFORMATION
# ============================================================

else:

    st.info(
        "Choose the number of faces and click "
        "**Generate Faces** to begin."
    )


st.divider()

st.caption(
    "Welmia Face 1.0 — 3.6M Base • "
    "Synthetic faces only • "
    "128×128 • CPU-friendly"
)
