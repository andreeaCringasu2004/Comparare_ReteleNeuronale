import os
import torch

torch.backends.cudnn.enabled = False

from torchvision.utils import save_image

from generator import Generator



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = "generator.pth"


OUTPUT_DIR = "generated_samples"



NUM_IMAGES = 20


NOISE_DIM = 100



os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================
# MODEL
# =====================


generator = Generator(

    NOISE_DIM

)



generator.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=DEVICE

    )

)



generator.to(DEVICE)


generator.eval()



# =====================
# GENERARE
# =====================


with torch.no_grad():


    noise = torch.randn(

        NUM_IMAGES,

        NOISE_DIM,

        device=DEVICE

    )


    generated = generator(
        noise
    )



    # revenim din [-1,1] in [0,1]

    generated = (
        generated + 1
    ) / 2



    for i,img in enumerate(generated):


        save_image(

            img,

            f"{OUTPUT_DIR}/fake_{i}.png"

        )



print(
    f"Generate {NUM_IMAGES} imagini in {OUTPUT_DIR}"
)
