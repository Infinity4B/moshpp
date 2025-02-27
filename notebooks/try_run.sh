#!/bin/bash

# 定义函数用于重复运行程序直到成功，并记录日志
run_until_success() {
    local program="$1"
    local log_file="$2"
    while true; do
        # 尝试运行 Python 程序，并将输出追加到日志文件
        if python3 "$program" >> "$log_file" 2>&1; then
            echo "程序 $program 运行成功。相关日志已记录到 $log_file"
            break
        else
            echo "程序 $program 运行失败，将在 5 秒后重试...相关日志已记录到 $log_file"
            sleep 5
        fi
    done
}

# 第一个 Python 程序的文件名
program="notebooks/solve_labeled_mocap.py"
# 日志文件的文件名
log_file="logs/moshpp.log"

# 运行第一个程序直到成功
run_until_success "$program" "$log_file"

echo "程序都已成功运行。"