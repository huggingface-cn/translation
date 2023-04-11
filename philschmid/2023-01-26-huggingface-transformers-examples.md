---
created: 2023-04-11T19:26:34 (UTC +08:00)
tags: []
source: https://www.philschmid.de/huggingface-transformers-examples
author: Philipp Schmid
---

# Hugging Face Transformers Examples

> ## Excerpt
> Learn how to leverage Hugging Face Transformers to easily fine-tune your models.

---
Each release of [Transformers](https://huggingface.co/docs/transformers/index) has its own set of examples script, which are tested and maintained. This is important to keep in mind when using `examples/` since if you try to run an example from, e.g. a newer version than the `transformers` version you have installed it might fail. All examples provide documentation in the repository with a README, which includes documentation about the feature of the example and which arguments are supported. All `examples` provide an identical set of arguments to make it easy for users to switch between tasks. Now, let's get started.

### 1\. Setup Development Environment

Our first step is to install the Hugging Face Libraries, including `transformers` and `datasets`. The version of `transformers` we install will be the version of the examples we are going to use. If you have `transformers` already installed, you need to check your version.

```
pip install torch
pip install "transformers==4.25.1" datasets  --upgrade
```

### 2\. Download the example script

The example scripts are stored in the [GitHub repository](https://github.com/huggingface/transformers) of transformers. This means we need first to clone the repository and then checkout the release of the `transformers` version we have installed in step 1 (for us, `4.25.1`)

```
git clone https://github.com/huggingface/transformers
cd transformers
git checkout tags/v4.25.1 # change 4.25.1 to your version if different
```

### 3\. Fine-tune BERT for text-classification

Before we can run our script we first need to define the arguments we want to use. For `text-classification` we need at least a `model_name_or_path` which can be any supported architecture from the [Hugging Face Hub](https://huggingface.co/) or a local path to a `transformers` model. Additional parameter we will use are:

-   `dataset_name` : an ID for a dataset hosted on the [Hugging Face Hub](https://huggingface.co/datasets)
-   `do_train` & `do_eval`: to train and evaluate our model
-   `num_train_epochs`: the number of epochs we use for training.
-   `per_device_train_batch_size`: the batch size used during training per GPU
-   `output_dir`: where our trained model and logs will be saved

You can find a full list of supported parameter in the [script](https://github.com/huggingface/transformers/blob/6f3faf3863defe394e566c57b7d1ad3928c4ef49/examples/pytorch/text-classification/run_glue.py#L71). Before we can run our script we have to make sure all dependencies needed for the example are installed. Every example script which requires additional dependencies then `transformers` and `datasets` provides a `requirements.txt` in the directory, which can try to install.

```
pip install -r examples/pytorch/text-classification/requirements.txt
```

Thats it, now we can run our script from a CLI, which will start training BERT for `text-classification` on the `emotion` dataset.

```
python3 examples/pytorch/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --dataset_name emotion \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --num_train_epochs 3 \
  --output_dir /bert-test
```

### 4\. Fine-tune BART for summarization

In 3. we learnt how easy it is to leverage the `examples` fine-tun a BERT model for `text-classification`. In this section we show you how easy it to switch between different tasks. We will now fine-tune BART for summarization on the [CNN dailymail dataset](https://huggingface.co/datasets/cnn_dailymail). We will provide the same arguments than for `text-classification`, but extend it with:

-   `dataset_config_name` to use a specific version of the dataset
-   `text_column` the field in our dataset, which holds the text we want to summarize
-   `summary_column` the field in our dataset, which holds the summary we want to learn.

Every example script which requires additional dependencies then `transformers` and `datasets` provides a `requirements.txt` in the directory, which can try to install.

```
pip install -r examples/pytorch/summarization/requirements.txt
```

Thats it, now we can run our script from a CLI, which will start training BERT for `text-classification` on the `emotion` dataset.

```
python3 examples/pytorch/summarization/run_summarization.py \
  --model_name_or_path facebook/bart-base \
  --dataset_name cnn_dailymail \
  --dataset_config_name "3.0.0" \
  --text_column "article" \
  --summary_column "highlights" \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --num_train_epochs 3 \
  --output_dir /bert-test
```
