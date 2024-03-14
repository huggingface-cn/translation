---
title: "Introducing ConTextual: How well can your Multimodal model jointly reason over text and image in text-rich scenes?"
thumbnail: /blog/assets/leaderboards-on-the-hub/thumbnail_contextual.png
authors:
- user: rohan598
  guest: true
- user: hbXNov
  guest: true
- user: kaiweichang
  guest: true
- user: violetpeng
  guest: true
- user: clefourrier
---
# ConTextual简介：在文本丰富的场景中，您的多模态模型对文本和图像的联合推理能力如何？

模型在理解纯文本方面已经相当出色，但文本在图像中的理解同样重要，因为这样可以提供重要的上下文信息。例如，在导航地图或理解表情包时，理解图像中的文本与视觉上下文之间的交互至关重要。这种能力可以推动许多现实世界的应用，如人工智能助手，或是帮助视觉障碍人士的辅助工具。

我们将这些任务称为“上下文敏感的文本丰富视觉推理任务”。

目前，大多数对指令调优的大型多模态模型（LMMs）的评估集中在测试模型如何响应以问题形式或命令句式提出的人类指令（“计算这个”，“列出那个”等），这些指令是针对图像的... 但并没有测试它们理解和处理文本丰富且上下文敏感的场景的能力！

这就是为什么我们（来自加州大学洛杉矶分校的研究人员）创造了ConTextual，一个用于评估LMMs的上下文敏感的文本丰富视觉推理数据集。我们还发布了一个排行榜，这样社区就可以自行了解在这个任务上哪些模型表现最佳。

<script type="module" src="https://gradio.s3-us-west-2.amazonaws.com/3.45.1/gradio.js"> </script>

<gradio-app theme_mode="light" space="ucla-contextual/contextual_leaderboard"></gradio-app>

如果您想深入了解，还可以查看这些附加资源：[论文](https://arxiv.org/abs/2401.13311)，[代码](https://github.com/rohan598/ConTextual)，[数据集](https://huggingface.co/datasets/ucla-contextual/contextual_all)，[验证数据集](https://huggingface.co/datasets/ucla-contextual/contextual_val)，以及[排行榜](https://huggingface.co/spaces/ucla-contextual/contextual_leaderboard)。

## ConTextual 是什么？

 ConTextual 是一个上下文敏感的文本丰富视觉推理数据集，包含506个用于LMM评估的具有挑战性的指令。我们创建了一系列指令，这些指令针对文本丰富的图像，并满足这样的约束：它们需要对图像中的文本和视觉线索进行上下文敏感的联合推理。

它涵盖了8个现实世界的视觉场景——时间阅读、购物、导航、抽象场景、移动应用程序、网页、信息图表以及各种自然场景。（见图分别为每个数据集的样本）。

![Real world visual scenarios examples](https://con-textual.github.io/static/images/teaser_figure.png)

每个样本包括：

- 一张文本丰富的图像
- 一条由人类编写的指令（问题或命令性任务）
- 一条由人类编写的参考回答
  该数据集以两种形式发布：
- （a）一个包含来自完整数据集的100个实例的验证集，其中包含指令、图像和针对指令的参考答案。
- （b）一个仅包含指令和图像的测试数据集。

排行榜上包含模型在验证集和测试数据集上的结果（这些信息也出现在论文中）。开发集允许实践者轻松测试和迭代他们的方法。评估沙盒在我们的 GitHub 上提供。

## 实验

在我们的初步实验中，我们的基准测试评估了13个模型的性能。我们将它们分为三类：

- **增强型LLM方法**：GPT4 + 视觉信息，形式为图像的OCR和/或密集图像字幕；
- **闭源LLM**：GPT4V(ision) 和 Gemini-Vision-Pro；
- **开源LLM**：LLaVA-v1.5-13B、ShareGPT4V-7B、Instruct-Blip-Vicuna-7B、mPlugOwl-v2-7B、Bliva-Vicuna-7B、Qwen-VL-7B 和 Idefics-9B。

我们的数据集为每条指令提供了一个参考响应，这使我们能够测试各种自动评估方法。在评估中，我们使用LLM作为法官的方法，提示GPT-4提供指令、参考响应和预测响应。该模型必须返回预测响应是否可接受。 （选择GPT4是因为在我们实验中它与人类判断的相关性最高。）

让我们来看一些例子！

[示例1](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-on-the-hub/contextual-qualitative-ex-1.png)

在此示例中，尽管GPT-4V的逻辑推理是正确的，但它对指令的响应是错误的。绿色表示与参考匹配的响应，而红色突出显示响应中的错误。此外，还提供了“总结推理”部分，以概述GPT-4V得出答案的推理过程。

[示例2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/leaderboards-on-the-hub/contextual-qualitative-ex-2.png)

在此示例中，GPT-4V正确地响应了指令。然而，性能最佳的ShareGPT-4V-7B（最佳表现的开源LLM）和具有布局感知OCR + 标题的GPT-4（增强型LLM）由于缺乏对文本和图像的联合推理，产生了错误的响应。
您可以在我们[paper](https://arxiv.org/abs/2401.13311)的附录部分找到更多此类示例！

## 关键结论！

在我们进行这项工作的过程中，我们发现：

- 现代LLM（专有和开源模型）在ConTextual数据集上的表现不佳，而人类在这方面的表现很好，这暗示了模型改进的可能性，以增强对富含文本的图像的推理能力，这是一个具有重大现实应用领域的领域。
- 专有LLM在涉及时间阅读的信息图表推理上表现不佳，这表明与人类相比，它们的能力存在差距。值得注意的是，表现最佳的模型GPT-4V在抽象推理方面超过了人类，这可能是因为它接触到了模因和引言数据，但在时间相关任务上挣扎，而人类在这方面表现出色。
- 对于像LLaVA-1.5-13B和ShareGPT-4V-7B这样的开源模型，它们在达到可接受人类评分的领域（抽象和自然场景上下文）与其他领域（时间阅读、信息图表、导航、购物、网络和移动使用）之间存在强烈差距。因此，我们样本中涵盖的许多领域可能对这些模型来说是分布之外的。因此，开源模型应该致力于增加其训练数据的多样性。
- 通过OCR或字幕将视觉信息转换为文本，然后增强LLM的Large Language Model表现显著不佳，人类批准率为17.2%。我们的样本需要精确的视觉感知以及细粒度的视觉-语言对齐来解决。

我们的分析表明，有希望的下一步包括：

- 开发增强的图像编码器，
- 创建高度准确的图像描述，
- 促进细粒度的视觉-语言对齐，以改善模型的感知并减少幻觉的发生。

这将反过来导致更有效的上下文敏感的富含文本的视觉推理。

## 下一步是什么？

我们也希望评估您的模型，以帮助共同推进视觉语言模型的发展！要提交，请遵循我们以下的指南。

我们希望这个基准测试将有助于开发细致的视觉-语言对齐技术，并欢迎各种形式的合作！您可以通过以下方式联系我们：[Rohan](rwadhawan7@g.ucla.edu) 和 [Hritik](hbansal@g.ucla.edu)，并在这里了解更多关于团队的信息：[Rohan](https://web.cs.ucla.edu/~rwadhawan7/)，[Hritik](https://sites.google.com/view/hbansal)，[Kai-Wei Chang](https://web.cs.ucla.edu/~kwchang/)，[Nanyun (Violet) Peng](https://vnpeng.net/)。

## 如何提交？

我们接受测试集和验证集的提交。请按照相应的程序进行。

### 验证集提交

要提交您的验证结果到排行榜，您可以运行我们的自动评估代码（使用 GPT4 的评估管道），按照 [这些说明](https://github.com/rohan598/ConTextual?tab=readme-ov-file#-evaluation-pipeline-gpt-4) 进行。

我们期望提交的格式是 json 格式，如下所示：

```json
{"model_name": {"img_url": "The boolean score of your model on the image, 1 for success and 0 for failure"}}
```

- 用您的模型名称（字符串）替换 model name
- 用实例的 img_url 替换 img_url（字符串）
- img url 的值应为 0 或 1（整数）
  应该有 100 个预测，对应于 val 集的 100 个 url。

要提交，请访问 HuggingFace 上托管的 [排行榜](https://huggingface.co/spaces/ucla-contextual/contextual_leaderboard) 并填写提交表格。

### 测试集提交

一旦您对验证结果满意，您可以将您的模型预测发送给 [Rohan](rwadhawan7@g.ucla.edu) 和 [Hritik](hbansal@g.ucla.edu)。

请在您的电子邮件中包括：

- 您的模型名称。
- 组织（隶属关系）。
- （可选）GitHub 仓库或论文链接。

我们期望提交的格式与验证集类似的 json 格式，如下所示：

```json
{"model_name": {"img_url": "predicted response"}}
```

- 使用您的模型名称（字符串）替换 model name
- 使用实例的 img_url 替换 img_url（字符串）
- img url 的值是对应实例的预测响应（字符串）

应该有 506 个预测，对应于测试集的 506 个 url。
