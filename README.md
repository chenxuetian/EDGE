# EDGE
The code repository of paper *EDGE: Enhanced Grounded GUI Understanding with Enriched Multi-Granularity Synthetic Data*.

We have also published our [model](https://huggingface.co/EDGEwww25/EDGE-Model) and [dataset](https://huggingface.co/datasets/EDGEwww25/EDGE-Dataset) as a newly created anonymous user on Huggingface.

## Examples
*EDGE* can answer user questions about GUI interactions in natural language, or directly specify bounding boxes or point coordinates. It is recommended to specify the grounding form in the query.

<img src=example_images/answered/iphone.png width=200>
<img src=example_images/answered/appstore.png width=195>

(left) \
**Q:** Where is the icon of Safari? (with bbox (x1, y1, x2, y2)) \
**A:** (0.3042, 0.8797, 0.4611, 0.957)\
(right) \
**Q:** Where should I click to find apps for children? (with point (x, y)) \
**A:** (0.6458, 0.1305)

<img src=example_images/answered/heatmap.png> \
**Q:** How can I share this page with others? \
**A:** You can click on the "Share" button (0.9668, 0.8559, 0.9914, 0.9016) in the bottom right corner to share the page on social media or email.

## Files
- `annotate_webpages/`: an Node.js project for annotating webpages and elementary tasks
- `edge_dataset/`: the dataset creation scripts, as well as a config file for dataset generating and a provided jsonl file containing a tiny portion of the training QAs as an example dataset
- `example_images/`: examples of GUI images used for inference
- `gen_advc_tasks/`: scripts for synthesizing advanced tasks using Claude-3.5
- `monkey_model/`: model code; please download the model parameters from [the model repository](https://huggingface.co/EDGEwww25/EDGE-Model)to try the inference of *EDGE*.
- `utils/`: utilities for dataset, training and inference
- `.gitignore`
- `finetune.sh`: the fine-tuning script used during *EDGE* training
- `inference.ipynb`: the notebook for interactive inference
- `README.md`: this README file

## Inference
1. Download model parameters in [the model repository](https://huggingface.co/EDGEwww25/EDGE-Model) and move them (two `.bin` files with a `.bin.index.json` file) to the `monkey_model` directory
2. Run `inference.ipynb` for interactive inference with our provided examples.
3. You can specify your own GUI screenshots and ask *EDGE* questions. *EDGE* can perform OCR, Referring, Grounding, and Action Grounding, and it is currently difficult to answer questions that require planning and multi-step execution.

## Training
Due to the large number of images, the images in the training set have not been released yet. We will improve it as soon as possible.

1. **Annotate Webpages:** See the instructions in the directory [annotate_webpages](annotate_webpages) to prepare webpages and generate annotations for elementary tasks.

2. **Synthesize Advanced Tasks**: See the instructions in the directory [gen_advc_tasks] to synthesize data for advanced tasks.

3. **Prepare Other Images**: Collect images like the external icons, llava-instruct-150k, monkey traning data, and various augmentation data. We will add the code and detailed description of this part as soon as possible.

4. **Create Dataset**: Use code similar to that in `inference.ipynb` to generate your own image-question-answer training set. 

5. **Train**: Execute the fine-tuning script `finetune.sh` to start model training.
