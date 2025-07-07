from flask import Flask, request, abort
from linebot.v3 import (
    WebhookHandler
)
from linebot.v3.exceptions import (
    InvalidSignatureError
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    PostbackEvent,
    ImageMessageContent
)
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    TextMessage,
    QuickReply,
    QuickReplyItem,
    PostbackAction,
    ImageMessage,
    ShowLoadingAnimationRequest,
    VideoMessage
)
import os
import gc
### 載入想要調用的模型
from models_py.flux1schnell import flux_schnell_model 
from models_py.flux1dev_inpaint import flux_dev_inpaint_model
from models_py.want2vR import wan21_t2v_model
from models_py.ch_to_en_translator import translate
from models_py.ltxi2v import LTX_i2v_model



app = Flask(__name__)

##輸入TOKEN
configuration = Configuration(access_token='')
handler = WebhookHandler('')


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        app.logger.info("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'

user_states = {}

@handler.add(MessageEvent, message=TextMessageContent)
def handle_messsage(event):
    user_id = event.source.user_id
    text = event.message.text
    
    # 獲取用戶的當前狀態，如果沒有則為 None
    state = user_states.get(user_id)

    if state == 'awaiting_text_prompt':
        reply_loading_message(
            user_id = user_id,
            waiting_time = 60
        )

        image_output_path = flux_schnell_model(prompt=translate(text), output_path="./static/")
        images = str(request.url_root+image_output_path).replace("http","https") #要回傳的圖片
        app.logger.info("url=" + images)

        reply_message(event, 
                      [TextMessage(text="以下為生成的圖片"),
                       ImageMessage(original_content_url=images, preview_image_url=images)])

        del user_states[user_id], images
        gc.collect()

    elif state == 'awaiting_t2v':
        reply_loading_message(
            user_id = user_id,
            waiting_time = 60
        )

        video_output_path = wan21_t2v_model(prompt=text, output_path="./static/")
        video = str(request.url_root+video_output_path).replace("http","https") #要回傳的圖片
        app.logger.info("url=" + video)
  

        reply_message(event, 
                      [TextMessage(text="以下為生成的影片"),
                       VideoMessage(original_content_url=video, preview_image_url=video)])

        del user_states[user_id], video
        gc.collect()
    
    elif state == 'awaiting_text_prompt_img2img':
        reply_loading_message(
            user_id = user_id,
            waiting_time =  60
        )

        image_output_path = flux_dev_inpaint_model(prompt=translate(text),input_image=user_states.get(user_id+"img") , output_path="./static/")
        images = str(request.url_root+image_output_path).replace("http","https") #要回傳的圖片
        app.logger.info("url=" + images)

        reply_loading_message(
            user_id = user_id,
            waiting_time = 60
        )

        reply_message(event, 
                      [TextMessage(text="以下為生成的圖片"),
                       ImageMessage(original_content_url=images, preview_image_url=images)])

        del user_states[user_id], user_states[user_id+"img"], images
        gc.collect()

    elif state == 'awaiting_text_prompt_i2v':
        reply_loading_message(
            user_id = user_id,
            waiting_time = 60 
        )

        video_output_path = LTX_i2v_model(prompt=translate(text),input_image=user_states.get(user_id+"img") , output_path="./static/")
        video = str(request.url_root+video_output_path).replace("http","https") #要回傳的圖片
        app.logger.info("url=" + video)


        reply_message(event, 
                      [TextMessage(text="以下為生成的影片"),
                       VideoMessage(original_content_url=video, preview_image_url=video)])

        del user_states[user_id], user_states[user_id+"img"], video
        gc.collect()
    
    elif text.lower() in ['hi', 'hello','圖片生成','你好'] and 'awaiting' not in user_states.get(user_id, ''):
        if user_id in user_states:
            del user_states[user_id] # 如果用戶在其他流程中，也一併清除狀態
        
        quick_reply_items=[
            QuickReplyItem(
                action=PostbackAction(
                    label="文生圖",
                    data="文生圖",
                    display_text="文生圖"
                )
            ),
            QuickReplyItem(
                action=PostbackAction(
                    label="圖生圖",
                    data="圖生圖",
                    display_text="圖生圖"
                )
            ),
            QuickReplyItem(
                action=PostbackAction(
                    label="文生影",
                    data="文生影",
                    display_text="文生影"
                )
            ),
            QuickReplyItem(
                action=PostbackAction(
                    label="圖生影",
                    data="圖生影",
                    display_text="圖生影"
                )
            )
        ]
        reply_message(event, [TextMessage(
            text='請選擇要生成的方式',
            quick_reply=QuickReply(
                items=quick_reply_items
            )
        )])
    else:
        reply_message(event, [TextMessage(text='無法識別指令，請輸入「圖片生成」來開始。')])


@handler.add(PostbackEvent)
def handle_postback(event):
    user_id = event.source.user_id
    
    if event.postback.data == "文生圖":
        user_states[user_id] = 'awaiting_text_prompt'
        reply_message(event,[TextMessage(text='請輸入文字')])
    elif event.postback.data == "圖生圖":
        user_states[user_id] = 'awaiting_image'
        reply_message(event,[TextMessage(text='請上傳一張圖片，為保持品質產出時間會較長一些，請耐心等候')])
    elif event.postback.data == "文生影":
        user_states[user_id] = 'awaiting_t2v'
        reply_message(event,[TextMessage(text='請輸入文字，並盡可能詳細說描述影片中出現物品、角色、畫面')])
    elif event.postback.data == "圖生影":
        user_states[user_id] = 'awaiting_i2v'
        reply_message(event,[TextMessage(text='請上傳一張圖片，為保持品質產出時間會較長一些，請耐心等候')])

@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    user_id = event.source.user_id
    state = user_states.get(user_id)

    # 檢查使用者狀態是否為 'awaiting_image'
    if state == 'awaiting_image':
        message_id = event.message.id
        with ApiClient(configuration) as api_client:
            content = MessagingApiBlob(api_client).get_message_content(message_id) # 獲取圖片的二進位內容
            # os.makedirs("downloaded_images/", exist_ok=True) # 建立儲存資料夾
            
            filename = str(user_id) +"-"+ f"{event.timestamp}.jpg" # 指定儲存路徑與檔名
            filepath = os.path.join("downloaded_images/", filename)

            user_states[user_id+"img"] = filepath #把圖片記在user_states給img2img,如果要多張圖片輸入就要變成list.append

            with open(filepath, 'wb') as f:
                f.write(content)

        reply_loading_message(
            user_id = user_id,
            waiting_time = 60
        )

        reply_message(
            event, 
            [TextMessage(text='請輸入想以此圖片為基礎生成的文字敘述')]
        )
        
        user_states[user_id] = 'awaiting_text_prompt_img2img'

    elif state == 'awaiting_i2v':
        message_id = event.message.id
        with ApiClient(configuration) as api_client:
            content = MessagingApiBlob(api_client).get_message_content(message_id) # 獲取圖片的二進位內容
            # os.makedirs("downloaded_images/", exist_ok=True) # 建立儲存資料夾
            
            filename = str(user_id) +"-"+ f"{event.timestamp}.jpg" # 指定儲存路徑與檔名
            filepath = os.path.join("downloaded_images/", filename)

            user_states[user_id+"img"] = filepath

            with open(filepath, 'wb') as f:
                f.write(content)

        reply_loading_message(
            user_id = user_id,
            waiting_time = 60
        )

        reply_message(
            event, 
            [TextMessage(text='請輸入想以此圖片為基礎生成的文字敘述')]
        )
        

        user_states[user_id] = 'awaiting_text_prompt_i2v'


# 回覆訊息
def reply_message(event, message):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=message
            )
        )

# 顯示等待動畫
def reply_loading_message(user_id, waiting_time):
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.show_loading_animation(
            ShowLoadingAnimationRequest(
                chatId = user_id,
                loadingSeconds = waiting_time
            )            
        )


if __name__ == "__main__":
    app.run()