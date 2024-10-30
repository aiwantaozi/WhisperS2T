import argparse
from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers import BasicTextNormalizer
import time
import os
import whisperx
from tqdm import tqdm
import pandas as pd
from pydub import AudioSegment
import jiwer

# run
# pip install -r ../benchmark_openai-whipser_requirements.txt
# python my_benchmark_openai_mac.py --repo_path /Users/michelia/Documents/project4ai/audio/data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_path', default="", type=str)
    parser.add_argument('--end_line', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--compute_type', default="float16", type=str)
    parser.add_argument('--device', default="cuda", type=str)

    args = parser.parse_args()
    return args


def run(repo_path, end_line=1, compute_type="float16", batch_size=16, device="cuda", lang="en"):
    results_dir = f"{repo_path}/results/WhisperX-bs_{batch_size}"
    os.makedirs(results_dir, exist_ok=True)

    model = whisperx.load_model(
        "large-v2", device, compute_type=compute_type, language=lang, asr_options={'beam_size': 1})

    data = pd.read_csv(
        f'{repo_path}/kincaid46_subset/kincaid46.txt', sep=" ")
    files = [f"{repo_path}/{fn}" for fn in data['audio_path'].tolist()]
    text_files = [f"{repo_path}/{fn}" for fn in data['audio_text'].tolist()]

    if lang == "en":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    if end_line == -1:
        end_line = len(files)

    audio_duration_list = []
    pred_text_size_list = []
    pred_text_list = []
    pred_begin_time = time.time()
    for fn in tqdm(files[:end_line], desc="audio trascribing"):
        audio = whisperx.load_audio(fn)
        result = model.transcribe(
            audio, batch_size=batch_size, without_timestamps=True)
        normalize_result = normalizer(" ".join([_['text'].strip()
                                                for _ in result['segments']]))

        audio_segment = AudioSegment.from_file(fn)
        duration_sec = len(audio_segment) / 1000.0

        pred_text_size_list.append(len(normalize_result.split()))
        pred_text_list.append(normalize_result)
        audio_duration_list.append(duration_sec)

    pred_end_time = time.time()

    ref_text_list = []
    for fn in tqdm(text_files[:end_line], desc="truth text loading"):
        with open(fn, 'r') as f:
            ref_text_list.append(normalizer(f.read().strip()))

    wer_list = []
    cer_list = []
    for idx in range(end_line):
        metrics = calculate_metrics(
            ref_text_list[idx], pred_text_list[idx])
        wer_list.append(metrics.get('wer'))
        cer_list.append(metrics.get('cer'))

    # save results
    data.insert(2, 'size', '')
    data.insert(3, 'wer', '')
    data.insert(4, 'cer', '')
    data.insert(5, 'audio_duration', '')
    data.insert(6, 'pred_text', '')

    data.loc[:end_line-1, 'size'] = pred_text_size_list
    data.loc[:end_line-1, 'wer'] = wer_list
    data.loc[:end_line-1, 'cer'] = cer_list
    data.loc[:end_line-1, 'audio_duration'] = audio_duration_list
    data.loc[:end_line-1, 'pred_text'] = pred_text_list
    data.to_csv(f"{results_dir}/KINCAID46_M4A.tsv", sep="\t", index=False)

    pred_latency = pred_end_time - pred_begin_time
    pred_duration = sum(audio_duration_list)
    total_metrics = calculate_metrics(ref_text_list, pred_text_list)
    pred_wer = total_metrics.get('wer')
    pred_cer = total_metrics.get('cer')
    pred_size = sum(pred_text_size_list)

    infer_time = [
        ["Dataset", "Time"],
        ["KINCAID46 m4a", pred_latency],
        ["Total Audio Files", end_line],
        ["Total Audio Duration", pred_duration],
        ["Total Text Size", pred_size],
        ["Total WER", pred_wer],
        ["Total CER", pred_cer],
    ]

    infer_time = pd.DataFrame(infer_time[1:], columns=infer_time[0])
    infer_time.to_csv(f"{results_dir}/infer_time.tsv", sep="\t", index=False)


def calculate_metrics(references, transcriptions):
    return {
        'wer': jiwer.wer(references, transcriptions),
        'cer': jiwer.cer(references, transcriptions)
    }


if __name__ == '__main__':
    args = parse_arguments()
    run(args.repo_path, batch_size=args.batch_size,
        end_line=args.end_line, compute_type=args.compute_type, device=args.device, lang="en")
    # for mac set device to cpu
