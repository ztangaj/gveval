# G-VEval: Image and Video Captioning Evaluation

## Overview

This repository provides tools for evaluating image and video captioning using G-VEval. The evaluation includes calculating correlations with human scores for various datasets such as Flickr-8k-expert, Flickr-8k-CF, and MSVD-Eval.

## Files

- **demo.py**: Demonstrates a sample run of G-VEval for image and video captioning evaluation.
- **correlation.py**: Calculates the correlation with human scores for the Flickr-8k-expert, Flickr-8k-CF, and MSVD-Eval datasets.
- **dataset_check.py**: Checks if the datasets are correctly installed.

## Setup

1. **Create a Data Directory**:
   Create a folder named `/data` in the root directory of the project.

2. **Download and Extract Datasets**:
   - For MSVD original videos, download and extract the dataset from [YouTubeClips.tar](https://www.cs.utexas.edu/~ml/clamp/videoDescription/YouTubeClips.tar) into the `/data` directory.
   - For Flickr8k datasets, download the dataset from [this link](https://drive.google.com/drive/folders/1oQY8zVCmf0ZGUfsJQ_OnqP2_kw1jGIXp?usp=sharing) and place it in the `/data` directory.

3. **Add OpenAI API Key**:
   Add your OpenAI API key in the `.env` file located in the root directory of the project:
   ```
   OPENAI_API_KEY='your-api-key-here'
   ```

4. **Human ACCR Scores**:
   The human ACCR scores for MSVD-Eval are already provided in the `MSVD-Eval.json` file.

## Usage

### Running the Demo

The `demo.py` file demonstrates a sample run of G-VEval for image and video captioning evaluation.

### Checking Dataset Installation

Use the `dataset_check.py` file to verify if the datasets are correctly installed.

