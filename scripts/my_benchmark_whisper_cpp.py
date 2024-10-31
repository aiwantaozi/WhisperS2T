import argparse
import platform
import subprocess
from whisper.normalizers import EnglishTextNormalizer
from whisper.normalizers import BasicTextNormalizer
import time
import os
from tqdm import tqdm
import pandas as pd
from pydub import AudioSegment
import jiwer


# run
# conda create -n whisper-benchmark-whispercpp python=3.11
# conda activate whisper-benchmark-whispercpp
# cd scripts
# pip install -r ../benchmark_requirements_whisper_cpp.txt --ignore-installed
# mac:
#   python my_benchmark_whisper_cpp.py --repo_path ../data --end_line 1 --device mps --binary_path /Users/michelia/Documents/project4ai/whisper.cpp/main --model_path /Users/michelia/Documents/project4ai/whisper.cpp/models/ggml-large-v2.bin
# nvidia
#   需要另外build whisper.cpp
#   cd whisper.cpp
#   sh ./models/download-ggml-model.sh large-v2
#   make clean
#   GGML_CUDA=1 make -j
#   ./main -m ./models/ggml-large-v2.bin -f samples/jfk.wav -otxt jfk.txt
#   python my_benchmark_whisper_cpp.py --repo_path ../data --end_line 1 --device cuda --binary_path /home/michelia/code/whisper.cpp/main --model_path /home/michelia/code/whisper.cpp/models/ggml-large-v2.bin

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_path', default="", type=str)
    parser.add_argument('--model_path', default="", type=str)
    parser.add_argument('--repo_path', default="", type=str)
    parser.add_argument('--end_line', default=1, type=int)
    parser.add_argument('--device', default="cuda", type=str)
    parser.add_argument('--threads', default=8, type=int)

    args = parser.parse_args()
    return args


def run(repo_path, end_line=1, device="cuda", threads=8, binary_path="", model_path="", lang="en"):
    system = platform.system()

    results_dir = f"{repo_path}/results/WhisperCPP-bs-{system}-{device}-threads{threads}"
    os.makedirs(results_dir, exist_ok=True)

    data = pd.read_csv(
        f'{repo_path}/kincaid46_subset/kincaid46-wav.txt', sep=" ")
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
        output = run_command(binary_path, model_path, fn, device, threads)

        normalize_result = normalizer(output.strip())

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

# ref: https://colab.research.google.com/drive/1B8BtVepMyvlFuQQyv87AWKn_UNkav6xu?usp=sharing#scrollTo=PcTuPy-fvxJp
# ref: https://github.com/g8a9/multilingual-asr-gender-gap/blob/800a1824adf66c95cbd367795976c6926dc419fa/src/2_sample_and_compute_metrics_binary.py#L28
# https://whisperapi.com/word-error-rate-wer
# https://www.educative.io/answers/how-to-compute-word-error-rate-wer-with-openai-whisper


def calculate_metrics(references, transcriptions):
    return {
        'wer': jiwer.wer(references, transcriptions),
        'cer': jiwer.cer(references, transcriptions)
    }


def run_command(binary_path: str, model_path: str, audio_file_path: str, device: str, threads: int) -> str:
    command = [
        binary_path,
        "-m",
        model_path,
        "-f",
        audio_file_path,
        "-debug",
        "--threads",
        str(threads),
        "--no-timestamps",
    ]

    if device == "cpu":
        command.append("-ng")

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )

        if result.returncode != 0:
            raise Exception(f"Unexpected return code: {result.returncode}")

        output = result.stdout
        if output == "" or output is None:
            raise Exception(
                f"Output is empty, return code: {result.returncode}")

        return output
    except Exception as e:
        raise Exception(
            f"Failed to execute {command}: {e},"
            f" stdout: {result.stdout}, stderr: {result.stderr}"
        )


if __name__ == '__main__':
    args = parse_arguments()
    run(args.repo_path,
        binary_path=args.binary_path,
        model_path=args.model_path,
        end_line=args.end_line,
        device=args.device,
        threads=args.threads,
        lang="en")   # for mac use fp32
