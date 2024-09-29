# Code usage

- Clone the repository 
- Use the python command `pip install ip-adapter` in the python environment you choose to use. It will only import the base version of the actual AI model. I have provided the zip `IP-Adapter-Instruct.zip` file of the updated version in order to make it easy to replace the base version with newest version.
- The `.bin` files for `ip-adapter-faceid`s are absolutely necessary. Without these, the code will not work. Below is the google drive link to those files due to their enormous size. These bin files must be in the parent folder.
- https://drive.google.com/drive/folders/1geaZjtR5PqZwI0F2AmS0pALL4jvClHJI?usp=sharing
- Navigate to python directory where this folder is, and replace it with the folder provided in the zip file `IP-Adapter-Instruct.zip` 
- Make sure you have `ollama` installed in your device. 
- Once done, navigate to `gradio_v2.py` and launch it for testing. (Here `gradio_v1.py` is the base version which is useful in generating images and testing purposes. `gradio_v2.py` is the final version, for now.)
- **REMEMBER:** Mistral nemo, though very strong with accuracy, may sometimes be inaccurate. The reason why this LLM was used is because no other LLM was accurate enough and mistral nemo was the only sustainable and accurate LLM for text generation and image analysis. The ip-adapter-faceid runs on both opencv and stable diffusion, so trying to improve any one can cause an imbalance in the other which leads to distorted images. The implemented code struck the perfect balance between exact facial feature extraction and image generation accuracy.
- For further documentation and deep dive into the trial code, check `documentation.md`.
