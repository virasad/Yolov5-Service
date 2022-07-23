import os

import gradio as gr

from detect import Detector

detector = Detector()
models_dir = os.path.join(os.path.dirname(__file__), "models")


# print(os.path.join(models_dir, "best_small_phase1.pt"))
# detector.set_model(os.path.join(models_dir, "best_small_phase1.pt"))


def predict(image):
    results = detector.detect_render(image)
    return results


models = {
    "Small-phase1": os.path.join(models_dir, "best_small_phase1.pt"),
    "Small-phase2": os.path.join(models_dir, "best_small_phase2.pt"),
    "Medium-phase2": os.path.join(models_dir, "best_medium_phase2.pt")
}


def set_model(model_path):
    model_path = models[model_path]
    print(f"Setting model to {model_path}")
    try:
        detector.set_model(model_path)
        return "Model set"
    except Exception as e:
        return str(e)


demo = gr.Blocks(title="Yolov5-Service for detecting catalyst", )

with demo:
    gr.Markdown("""
    Inference **Catalyst** demo.
    This Package develop by [Rutilea](https://www.rutilea.com/)


    How to use:
    1. Select model
    2. Select image or you can choose from examples
    3. Click predict

    The default model is `Small-phase1`.
    """)

    phase1_images_dir = os.path.join(os.path.dirname(__file__), "images", "phase1")
    example1_images = os.listdir(phase1_images_dir)
    phase2_images_dir = os.path.join(os.path.dirname(__file__), "images", "phase2")
    example2_images = os.listdir(phase2_images_dir)
    example1_images = [os.path.join(os.path.dirname(__file__), "images", 'phase1', image) for image in example1_images]
    example2_images = [os.path.join(os.path.dirname(__file__), "images", 'phase2', image) for image in example2_images]

    with gr.Tabs():
        with gr.TabItem("Set model"):
            text_input = gr.Radio(choices=["Small-phase1", "Small-phase2", "Medium-phase2"],
                                  value="Small-phase1",
                                  label="Choose one model to predict Model", )
            text_output = gr.Text(label="Result", value="", placeholder="Result will be shown here")
            text_button = gr.Button("Set-model")

        with gr.TabItem("Inference"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Predict")

    text_button.click(set_model, inputs=text_input, outputs=text_output)
    image_button.click(predict, inputs=image_input, outputs=image_output)
    gr.Markdown("""

    Example images **Phase 1**:

    """)
    gr.Examples(
        examples=example1_images,
        inputs=[image_input],
        outputs=[image_output],
        fn=predict,
    )
    gr.Markdown("""

    Example images **Phase 2**:

    """)
    gr.Examples(
        examples=example2_images,
        inputs=[image_input],
        outputs=[image_output],
        fn=predict,
    )

demo.launch()
