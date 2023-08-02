# Setting up AWS Trainium for Hugging Face Transformers

> ## Excerpt
> Learn how to quickly set up an AWS Trainium using the Hugging Face Neuron Deep Learning AMI and fine-tune BERT

---
A couple of weeks ago, [Hugging Face and AWS announced they will partner](https://huggingface.co/blog/aws-partnership) to make AI open and more accessible. Part of this partnership is to develop tools that make it easier for practitioners to leverage AWS purpose-built instance [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) and [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) to train, fine-tune, and deploy Transformer and Diffusion models on AWS.

_“AWS Trainium is the second-generation machine learning (ML) chip that AWS purpose-built for deep learning training. \[…\] Trainium-based EC2 Trn1 instances solve this challenge by delivering faster time to train while offering up to 50% cost-to-train savings over comparable GPU-based instances.” -_ [AWS](https://aws.amazon.com/machine-learning/trainium/)

We are super excited to bring these price-performance advantages to Transformers and Diffusers. 🚀

In this hands-on post, We'll show you how to quickly set up an AWS Trainium using the [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2) to fine-tune a BERT model for text classification.

Let's get started! 🔥

## Setting up an AWS Trainium instance on AWS

The simplest way to work with AWS Trainium and Hugging Face Transformers is the [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2) (DLAMI). The DLAMI comes with all required libraries pre-packaged for you, including the Neuron Drivers, Transformers, Datasets, and Accelerate.

To create an EC2 Trainium instance, you can start from the console or the Marketplace. This guide will start from the [EC2 console](https://console.aws.amazon.com/ec2sp/v2/).

Starting from the [EC2 console](https://console.aws.amazon.com/ec2sp/v2/) in the us-east-1 region, you first click on **Launch an instance** and define a name for the instance (`trainium-huggingface-demo`).

![01-name-instance](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F01-name-instance.png&w=1920&q=75)

Next you search the Amazon Marketplace for Hugging Face AMIs. Entering “Hugging Face” in the search bar for “Application and OS Images” and hitting “enter”.

![02-search-ami](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F02-search-ami.png&w=1920&q=75)

This should now open the “Choose an Amazon Machine Image” view with the search. You can now navigate to “AWS Marketplace AMIs” and find the [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2) and click select.

![03-select-ami](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F03-select-ami.png&w=3840&q=75)

_You will be asked to subscribe if you aren’t. The AMI is completely free of charge, and you will only pay for the EC2 compute._

Then you need to define a key pair, which will be used to connect to the instance via `ssh`. You can create one in place if you don't have a key pair.

![04-select-key](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F04-select-key.png&w=1920&q=75)

After that, create or select a [security group](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html). Important you want to allow `ssh` traffic.

![05-select-sg](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F05-select-sg.png&w=1920&q=75)

You are ready to launch the instance. Therefore click on “Launch Instance” on the right side.

![06-launch-instance](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F06-launch-instance.png&w=828&q=75)

AWS will now provision the instance using the [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2). Additional configurations can be made by increasing the disk space or creating an instance profile to access other AWS services.

After the instance runs, you can view and copy the public IPv4 address to `ssh` into the machine.

![07-copy-dns](https://www.philschmid.de/_next/image?url=%2Fstatic%2Fblog%2Fsetup-aws-trainium%2F07-copy-dns.png&w=3840&q=75)

Replace the empty strings `""` in the snippet below with the IP address of your instances and the path to the key pair you created/selected when launching the instance.

```
PUBLIC_DNS="" # IP address, e.g. ec2-3-80-....
KEY_PATH="" # local path to key, e.g. ssh/trn.pem

ssh -i $KEY_PATH ubuntu@$PUBLIC_DNS
```

After you are connected, you can run `neuron-ls` to ensure you have access to the Trainium accelerators. You should see a similar output than below.

```
ubuntu@ip-172-31-79-164:~$ neuron-ls
instance-type: trn1.2xlarge
instance-id: i-0570615e41700a481
+--------+--------+--------+---------+
| NEURON | NEURON | NEURON |   PCI   |
| DEVICE | CORES  | MEMORY |   BDF   |
+--------+--------+--------+---------+
| 0      | 2      | 32 GB  | 00:1e.0 |
+--------+--------+--------+---------+
```

## Fine-tune a BERT model for text classification

The [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2) not only comes with all libraries pre-packaged, it also includes all supported `huggingface-neuron-samples/` scripts from the [optimum-neuron library](https://github.com/huggingface/optimum-neuron/tree/main/examples). This means you can directly launch our training job using the [text-classification script](https://github.com/huggingface/optimum-neuron/tree/main/examples/text-classification).

The training scripts use the new `TrainiumTrainer`, a purpose-built Transformers Trainer for AWS Trainium. The `TrainiumTrainer` comes with several benefits, including a compilation cache. This means we can skip the compilation step (~10-15 min) for your model + configuration if it is cached already. Learn more about the cache in the [documentation](https://huggingface.co/docs/optimum-neuron/guides/cache_system).

The training script will download the [bert-base-uncased](https://huggingface.co/bert-base-uncased) model from the Hugging Face hub and fine-tunes it on the [emotion](https://huggingface.co/datasets/philschmid/emotion) dataset, which consists of 10 000 Twitter messages with six labels: anger, fear, joy, love, sadness, and surprise.

The `trn1.2xlarge` instance comes with 2 Neuron Cores. Therefore `torchrun` is used to leverage both and launch our training.

```
torchrun --nproc_per_node=2 huggingface-neuron-samples/text-classification/run_glue.py \
--model_name_or_path bert-base-uncased \
--dataset_name philschmid/emotion \
--do_train \
--do_eval \
--bf16 True \
--per_device_train_batch_size 16 \
--learning_rate 5e-5 \
--num_train_epochs 3 \
--output_dir ./bert-emotion
```

_**Note**: If you see bad, bad accuracy, you might want to deactivate `bf16` for now._

After 2 minutes and 42 seconds, the training was completed and achieved an excellent accuracy of `0.925`.

```
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     0.3032
  train_runtime            = 0:02:42.34
  train_samples            =      16000
  train_samples_per_second =    295.663
  train_steps_per_second   =      9.239

***** eval metrics *****
  epoch                   =        3.0
  eval_accuracy           =      0.925
  eval_loss               =     0.2057
  eval_runtime            = 0:00:07.41
  eval_samples            =       2000
  eval_samples_per_second =    269.779
  eval_steps_per_second   =     16.861
```

Last but not least, terminate the EC2 instance to avoid unnecessary charges. Looking at the price-performance, the training only cost `7ct` (`1.34$/h * 0.05h = 0.07$`)

## Conclusion

In conclusion, the combination of AWS Trainium and Hugging Face Transformers provides a powerful, simple, and cost-effective solution for training state-of-the-art natural language processing models. By leveraging the purpose-built Trainium instances, practitioners can achieve faster training times and cost savings over comparable GPU-based instances. With the [Hugging Face Neuron Deep Learning AMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2), all required libraries are pre-packaged, making it easy to fine-tune Transformer models for extractive or generative use cases.

Give it a try, and let us know what you think. We welcome your questions and feedback on the [Hugging Face Forum.](https://discuss.huggingface.co/c/aws-inferentia-trainium/66)