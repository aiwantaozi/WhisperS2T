#!/bin/bash

# 检查输入参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

input_directory=$1
output_directory=$2

# 创建输出目录（如果不存在）
mkdir -p "$output_directory"

# 遍历输入目录中的所有 .m4a 文件
for input_file in "$input_directory"/*.m4a; do
    # 获取文件名（不带扩展名）
    filename=$(basename "$input_file" .m4a)
    
    # 构建输出文件路径
    output_file="$output_directory/$filename.wav"
    
    # 转换文件
    ffmpeg -i "$input_file" -ar 16000 -ac 1 -c:a pcm_s16le "$output_file"
    
    echo "Converted $input_file to $output_file"
done

echo "All files have been converted."