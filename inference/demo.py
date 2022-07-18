import io
from PIL import Image
from detect import Detector
import os
import gradio as gr


detector = Detector()


def predict(image):
    results = detector.detect_render(image)
    return results


def set_model(model_path):
    try:
        detector.set_model(model_path.name)
        return "Model set"
    except Exception as e:
        return str(e)


demo = gr.Blocks()

with demo:
    gr.Markdown("Inference demo")
    with gr.Tabs():
        with gr.TabItem("Set model"):
            text_input = gr.File()
            text_output = gr.Text()
            text_button = gr.Button("Set")
        with gr.TabItem("Inference"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Predict")

    text_button.click(set_model, inputs=text_input, outputs=text_output)
    image_button.click(predict, inputs=image_input, outputs=image_output)

demo.launch()