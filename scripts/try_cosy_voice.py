import torchaudio
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice-300M-SFT',
                  local_dir='pretrained_models/CosyVoice-300M-SFT')

# run: 一来cosyvoice，要放到CosyVoice项目里面，安装CosyVoice项目的初始化步骤，不然运行不了

cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-SFT',
                      load_jit=False, fp16=True)
# sft usage
print(cosyvoice.list_avaliable_spks())
# change stream=True for chunk stream inference
for i, j in enumerate(cosyvoice.inference_sft('你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？', '中文女', stream=False)):
    torchaudio.save('sft_{}.wav'.format(i), j['tts_speech'], 22050)
