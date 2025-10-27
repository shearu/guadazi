import gradio as gr
import base64
import tempfile
from openai import OpenAI
import os
from PIL import Image
import io

api_key=""
# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://aistudio.baidu.com/llm/lmapi/v3" # 星河社区大模型API服务的BaseURL
)

def image_to_base64(image_path):
    """将图像转换为base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading image: {e}")
        return None
def call_text_api(title):
    """纯文本API调用"""
    response = client.chat.completions.create(
        model="ernie-3.5-8k",  # 使用文本模型
        messages=[{
            "role": "user", 
            "content": f"用00后的口吻吐槽：{title}。要求用到'绝绝子''离谱''塌房'等网络用语，回复要简短犀利。"
        }]
    )
    return response.choices[0].message.content

def call_multimodal_api(image_path, title):
    """调用多模态API"""
    if not os.path.exists(image_path):
        return f"错误：图片文件不存在"
    
    image_base64 = image_to_base64(image_path)
    if not image_base64:
        return "错误：图片转换失败"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"用00后的口吻，结合图片内容吐槽一下：{title}。要求用到'绝绝子''离谱''塌房'等网络用语，回复要简短犀利。"
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="ernie-4.5-vl-28b-a3b",
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API调用失败: {str(e)}"

def analyze_news(news_title, image):
    """处理用户输入"""
    
    try:
        if image is not None:
            # 保存图片到临时文件
            if isinstance(image, str):
                image_path = image
            else:
                # 处理各种图片格式
                if hasattr(image, 'name'):
                    image_path = image.name
                else:
                    # 创建临时文件
                    _, temp_path = tempfile.mkstemp(suffix='.png')
                    if hasattr(image, 'save'):
                        # PIL Image
                        image.save(temp_path)
                    else:
                        # 其他格式，尝试转换
                        pil_image = Image.fromarray(image)
                        pil_image.save(temp_path)
                    image_path = temp_path
            
            # 调用API
            result = call_multimodal_api(image_path, news_title)
        else:
            # 无图片：只调用文本API
            result = call_text_api(news_title)  # 纯文本处理函数
            # return result
        # 返回结果和预览图片
        return result, image
        
    except Exception as e:
        return f"处理失败: {str(e)}", None

# 创建改进的Gradio界面
with gr.Blocks(theme=gr.themes.Soft(), title="🍉 瓜搭子AI吃瓜分析器") as demo:
    gr.Markdown("""
    # 🍉 瓜搭子 - 00后专属AI吃瓜助手
    **上传吃瓜图片，让AI用00后口吻犀利吐槽！**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 输入信息")
            title_input = gr.Textbox(
                label="吃瓜标题",
                value="某网红直播带货翻车，产品全是假货",
                placeholder="输入你要吐槽的吃瓜事件...",
                lines=2
            )
            image_input = gr.Image(
            label="上传吃瓜图片（可选）",
            type="numpy",
            height=200,
            show_download_button=True
            )
            
            submit_btn = gr.Button("🍉 AI吐槽分析", variant="primary", size="lg")
            clear_btn = gr.Button("🔄 清空", size="lg")
            
        with gr.Column(scale=2):
            gr.Markdown("### 🤖 分析结果")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📷 图片预览")
                    image_preview = gr.Image(
                        label="上传的图片预览",
                        height=200,
                        interactive=False
                    )
                with gr.Column(scale=2):
                    gr.Markdown("#### 💬 AI吐槽")
                    output_text = gr.Textbox(
                        label="",
                        lines=5,
                        placeholder="AI的犀利吐槽将显示在这里...",
                        show_copy_button=True
                    )
    
    # 绑定事件
    def update_preview(image):
        """更新图片预览"""
        return image
    
    def clear_all():
        """清空所有输入"""
        return "", None, None, ""
    
    # 当图片上传时更新预览
    image_input.change(
        fn=update_preview,
        inputs=[image_input],
        outputs=[image_preview]
    )
    
    # 提交分析
    submit_btn.click(
        fn=analyze_news,
        inputs=[title_input, image_input],
        outputs=[output_text, image_preview]
    )
    
    # 清空
    clear_btn.click(
        fn=clear_all,
        outputs=[title_input, image_input, image_preview, output_text]
    )

print("正在启动瓜搭子演示界面...")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)