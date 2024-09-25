import streamlit as st
import json
import time
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import gradio as gr


def extract_json_from_text(text):
    # 使用正则表达式查找JSON部分的开始和结束位置
    try:
        start_index = text.index('{')
        end_index = text.rindex('}') + 1  # +1 包括最后的右大括号
        json_str = text[start_index:end_index]  # 提取JSON字符串
        data = json.loads(json_str)  # 解析JSON字符串
        return data
    except ValueError as e:
        print(f"Error finding JSON in text: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            # Preparation for inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = ''.join(output_text)

            print('-'*10)
            print(output_text)
            output = extract_json_from_text(output_text)
            if output==None:
                raise Exception('JSON decode error : {}'.format(output_text))

            return output
        except Exception as e:
            messages.append(
                {"role": "user", "content": """
                Please response in a valid JSON response.
                Response Format:
                ```json
                {
                    "title": "xxx", (Summarize this step response)
                    "content": "xxx", (Please provide an one-more different step analysis here that you believe is likely to answer the question correctly)
                    "next_action" "xxx",  ("continue" or "final_answer")
                }```

                """}
            )
            if attempt == 2:
                if is_final_answer:
                    return {"title": "Error",
                            "content": f"{output_text}"}
                else:
                    return {"title": "Error", "content": f"Failed to generate step after 3 attempts. Error: {str(e)}",
                            "next_action": "final_answer"}
            time.sleep(1)  # Wait for 1 second before retrying


def generate_response(prompt, img_query=None):
    if(img_query == None):
        messages = [
            {"role": "system", "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

            1. Provide a clear, concise title describing the current reasoning phase.
            2. Elaborate on your thought process in the content section.
            3. Decide whether to continue reasoning or provide a final answer.

            Response Format:
            Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

            Key Instructions:
            - Employ at least 5 distinct reasoning steps.
            - Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
            - Actively explore and evaluate alternative answers or approaches.
            - Critically assess your own reasoning; identify potential flaws or biases.
            - When re-examining, employ a fundamentally different approach or perspective.
            - Utilize at least 3 diverse methods to derive or verify your answer.
            - Incorporate relevant domain knowledge and best practices in your reasoning.
            - Quantify certainty levels for each step and the final conclusion when applicable.
            - Consider potential edge cases or exceptions to your reasoning.
            - Provide clear justifications for eliminating alternative hypotheses.


            Example of a valid JSON response:
            ```json
            {
                "title": "Initial Problem Analysis",
                "content": "xxx" (To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.),
                "next_action": "continue"
            }```
            """
             
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
        ]
    else:
        messages = [
                {"role": "system", "content": """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

                1. Provide a clear, concise title describing the current reasoning phase.
                2. Elaborate on your thought process in the content section.
                3. Decide whether to continue reasoning or provide a final answer.

        
                Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' if you think you need to continue explain or 'final_answer' if you think you can correctly answer the question based on the above step-by-step explaintion)
                Response in a valid JSON response, response Format:
                Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

                Key Instructions:
                - Employ at least 5 distinct reasoning steps.
                - Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
                - Actively explore and evaluate alternative answers or approaches.
                - Critically assess your own reasoning; identify potential flaws or biases.
                - When re-examining, employ a fundamentally different approach or perspective.
                - Utilize at least 3 diverse methods to derive or verify your answer.
                - Incorporate relevant domain knowledge and best practices in your reasoning.
                - Quantify certainty levels for each step and the final conclusion when applicable.
                - Consider potential edge cases or exceptions to your reasoning.
                - Provide clear justifications for eliminating alternative hypotheses.


                Example of a valid JSON response:
                ```json
                {
                    "title": "Initial Problem Analysis",
                    "content": "xxx" (To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.),
                    "next_action": "continue" or "final_answer"
                }```
                """
                },
                {"role": "user", "content": [{"type": "image", "image": img_query},{"type": "text", "text":prompt}]},
                {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
            ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Thinking Step {step_count}: ", step_data['content'], thinking_time))
        # print(f"Thinking Step {step_count}: ", step_data['content'], thinking_time)

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        if step_count>5 or step_data['next_action'] == 'final_answer':
            break
        else:
            messages.append(
                {"role": "user", "content": """
                Please continue provide detailed, step-by-step explanations of your thought process. And response in a valid JSON response.
                Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' if you think you need to continue explain to answer the question correctly, or 'final_answer' if you think you can correctly answer the question based on the above step-by-step explaintion)
                Noting: the value of 'next_action' is continue, the content of this step must be very different from previous content. Otherwise, please set 'next_action' to final_answer.
                Response Format:
                ```json
                {
                    "title": "xxx", (Summarize this step response)
                    "content": "xxx", (Please provide an one-more different step analysis here that you believe is likely to answer the question correctly)
                    "next_action" "xxx",  ("continue" or "final_answer")
                }```

                """}
            )

        step_count += 1

        # # Yield after each step for Streamlit to update
        # yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})

    start_time = time.time()
    final_data = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time))

    yield steps, total_thinking_time


def inference(user_query, uploaded_file):
    if uploaded_file is not None:
        # 打开图像并显示它
        img = uploaded_file
        img_query = "/data/nas/mayiwei/code/MLLM/o1/demo.png"
        # 保存文件到指定路径
        img.save(img_query)
    else:
        img_query = None
    
    steps = []
    total_thinking_time = 0
    for steps_part, thinking_time in generate_response(user_query, img_query):
        steps.extend(steps_part)
        total_thinking_time = thinking_time
        
    return "Generating response...", total_thinking_time, steps

model_name = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)




with gr.Blocks() as demo:
    gr.Markdown("## Multimodal OpenAI-O1  (MO1)")
    gr.Markdown("""
    This is an early prototype of using prompts to create a class OpenAI-o1 inference chain to improve output accuracy.
    """)
    
    with gr.Row():
        with gr.Column():
            query = gr.Textbox(label="Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?", value="How many 'R's are in the word strawberry?")
            image = gr.Image(type="pil", label="Upload an image")
            btn = gr.Button("Submit")
    
    with gr.Row():
        with gr.Column():
            markdown_output = gr.Markdown()
            thinking_time = gr.Textbox(label="Total thinking time")
            dataframe_output = gr.Dataframe(headers=["Step", "Content", "Thinking Time"])
                
    
    
    def process_click(query, image):
        output_md, output_time, output_df = inference(query, image)
        return output_md, output_time, output_df
    
    btn.click(process_click, inputs=[query, image], outputs=[markdown_output, thinking_time, dataframe_output])

demo.launch()
