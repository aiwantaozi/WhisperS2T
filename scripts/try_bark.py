# conda create -n bark python=3.11
# conda activate bark
# pip install git+https://github.com/suno-ai/bark.git
# pip install git+https://github.com/huggingface/transformers.git
# pip install


import scipy
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("""We are getting reports that 11 people have been rescued. Each of them, remember, has to be pulled through the cave by expert divers. It is such a risky and complicated operation. So let's take you straight there to Tham Luang. Our correspondent Dan Johnson is there. Uh and Dan, signs of ambulances with sirens going. Tell us what you've witnessed in the past few minutes or so.
Yes. """, voice_preset=voice_preset)

print(1)
audio_array = model.generate(**inputs)
print(2)
audio_array = audio_array.cpu().numpy().squeeze()
print(3)

sample_rate = model.generation_config.sample_rate
print(4)

scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
