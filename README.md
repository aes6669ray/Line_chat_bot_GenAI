# Line_chat_bot_GenAI

### 此項目演示如何把抱臉蟲(Hugging Face)上面開源的AI生圖/影模型串接Line去實現透過line打造自己的免費AI生圖/影的小工具

### 大綱:透過ngrok本地架設這個LINE聊天機器人。模型都是使用開源的模型。但是flux1 dev版並非可商用地模型，所以使用前請一定要確認開源模型一併提供的協議是否有支持商用。
### 心得:主要是玩AIGC的使用者/開發者較少把這項功能移植到Line上或是教學如何自己架設一個可以玩AIGC的伺服器所以就打算自己當興趣弄一個(可能也歸咎於歐美印等地區也較少用戶使用LINE,所以幾乎查不到相關文章)。還是建議用comfyui或webui去玩SD模型。python在調整參數還有資源調用上真的不比comfyui的優化好。

## 

### 效果演示:
#### 文生圖
![image](https://github.com/aes6669ray/Line_chat_bot_GenAI/blob/main/demo/%E6%96%87%E7%94%9F%E5%9C%96.jpg)

#### 圖生影
![image](https://github.com/aes6669ray/Line_chat_bot_GenAI/blob/main/demo/%E5%9C%96%E7%94%9F%E5%BD%B1.jpg)
![image](https://github.com/aes6669ray/Line_chat_bot_GenAI/blob/main/demo/%E5%BD%B1%E7%89%87demo.gif)

##
### 使用說明:app.py操控所有LINE上的PUSH、REPLY機制。把想要使用的模型都寫成FUNCTION並存在models_py(模型本體存在models)，再從app.py去調用與設定REPLY時機。只要看得懂LINE的參數、FUNCTION的調用就沒甚麼難點。建議先熟悉怎麼用python建立LINE聊天機器人
## 

### 設備:
##### GPU:5090 Vram 32GB 
##### SSD: T705 GEN 5 4TB 
##### CPU: AMD 9800X3D 8core
##### DRAM: DDR5 64 GB


## 

### 功能與運用模型:
### 文生圖:black-forest-labs/FLUX.1-schnell 來源:https://huggingface.co/black-forest-labs/FLUX.1-schnell
### 文生影:Wan-AI/Wan2.1-T2V-1.3B 來源:https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
### 圖生圖:black-forest-labs/FLUX.1-dev 來源:https://huggingface.co/black-forest-labs/FLUX.1-dev
有加上ControlNet:InstantX/FLUX.1-dev-Controlnet-Union, mode:2 [depth] 來源:https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro
### 圖生影:Lightricks/LTX-Video 來源:https://huggingface.co/Lightricks/LTX-Video 
-用wan2.1i2v太慢了大多會超過1min,LTX輕量很多
