# SoftVC VITS Singing Voice Conversion

## 模型简介

歌声音色转换模型，通过Soft-VC内容编码器提取源音频语音特征，并提取音频f0，将两者结合输入VITS替换原本的文本输入达到歌声转换的效果。

## 使用方式

1、main.py为一键合成长时间音频（数分钟以上），使用方法参考注释

2、gui开启127.0.0.1:7860网页，可在页面加载模型、转换语音（太长可能爆显存）

3、http_server可以转换http方式传输的音频，使用方法参考http_cli
