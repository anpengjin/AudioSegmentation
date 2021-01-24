# encoding:utf-8

"""
@Author: anpj
@Date: 2020-1-7 11:15:22
@Description: 句子分割-模型

要求：
1. 不丢字
2. 速度快
3. 切分的音频时长尽量在一个固定的范围内

功能：
输入原始音频路径、切割的起止时间，将切割好的音频输出到指定文件夹下
""" 
import os
import time

from model import AudioSplitByTimeAndSpec
from utils import read_audio

#--------------------------------------------------------------------------------------------
# 对外api
def main(origin_audio_path:str=None, output_path:str=None, start_time:int=0, end_time:int=None):
    if end_time is None:
        y, sr = read_audio(origin_audio_path, start_time)
    else:
        duration = end_time - start_time + 1
        y, sr = read_audio(origin_audio_path, start_time, duration)
    left_pcm, right_pcm = y.tolist(), y.tolist()
    audio = os.path.basename(origin_audio_path)
    task_name = audio.split(".")[0]
    item = {
            "task_name":task_name,
            "sample_rate":sr,
            "nb_channels":"1",
            "sample_bits":"16",
            "data_type":"1",
            "start_time":start_time,
            "end_time":end_time,
            "task_end":1,
            "flag": 1,
            "left_pcm": left_pcm,
            "right_pcm": right_pcm,
            "end_split_time":None,
            "remain":None,
        }
    audio_file = AudioSplitByTimeAndSpec(item=item, output_path=output_path)
    audio_file.split()

# 测试
def test():
    """
    input:
        origin_audio_path：原始音频路径
        start_time：要切割的音频的起始时间/秒，默认0
        end_time：要切割的音频的结束时间/秒，默认到音频结束
        output_path：切分后的音频保存的文件夹
    """
    origin_audio_path = "/source/d2/tts/DataSet/MutliHosts/host2/1.wav" # 原始音频路径
    start_time, end_time = 0, 60 # 音频的起止时间，结束时间
    output_path = "output_split_audio" #切分后音频的保存文件夹
    
    # main(origin_audio_path, output_path, start_time, end_time)
    main(origin_audio_path, output_path)

if __name__ == "__main__":
    import fire 
    fire.Fire()
