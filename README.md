# Encoder-Decoder-with-Attention-Mechanism

**Sequence-to-Sequence Model with Attention Mechanism (Built from Scratch)**

## 📌 Overview

This project implements a Neural Machine Translation (NMT) system to translate sentences from **English to Hindi** using a custom-built **Encoder-Decoder architecture** enhanced with the **Bahdanau Attention mechanism**, implemented from scratch using **PyTorch**.

The model is trained on a large-scale parallel corpus (IIT Bombay English-Hindi dataset) and demonstrates the core components of machine translation: sequence modeling, context handling, and alignment through attention.

## 🚀 Highlights

* Trained on **500,000 sentence pairs**
* Implemented **Encoder-Decoder with Bahdanau Attention** from scratch
* Trained for **15 epochs** on **GPU** (local machine)
* Training time: **78+ hours**
* Evaluated using **BLEU Score**: Achieved **1.2**, limited by computational constraints
* Dataset: IIT Bombay English-Hindi Parallel Corpus

## 🔧 Architecture

The model is based on the **Sequence-to-Sequence with Attention** approach:

* **Encoder**: Embeds and encodes the input English sentence using GRU
* **Attention Mechanism**: Computes alignment scores between decoder hidden state and encoder outputs (Bahdanau-style)
* **Decoder**: Generates the target Hindi sentence one word at a time using context vectors from attention

## 📊 Results

| Metric        | Value                                 |
| ------------- | ------------------------------------- |
| BLEU Score    | 1.2                                   |
| Epochs        | 15                                    |
| Training Time | 78+ hours                             |
| Hardware      | Laptop GPU (NVIDIA GeForce GTX / RTX) |

> **Note:** The BLEU score is low primarily due to limited GPU memory and compute time. However, the model architecture is scalable and ready for training on larger infrastructure.

## 🛠️ Technologies Used

* **Python**
* **PyTorch**
* **NLTK**
* **NumPy**, **Pandas**
* **TQDM**, **Matplotlib**

## 📁 Repository Structure

```
.
├── data/                    # Preprocessed training and validation data
├── models/                  # Encoder, Decoder, Attention implementation
├── utils/                   # Helper functions and evaluation scripts
├── notebooks/               # Training and analysis notebooks
├── checkpoints/             # Model checkpoints (optional)
├── main.py                  # Training loop
├── evaluate.py              # Inference and BLEU evaluation
└── README.md
```

## 📅 Dataset

We used the **IIT Bombay English-Hindi Parallel Corpus**. It contains parallel aligned sentences for machine translation research.

* Download Link: [https://www.cfilt.iitb.ac.in/iitb\_parallel/](https://www.cfilt.iitb.ac.in/iitb_parallel/)
* Preprocessing: Tokenization, lowercasing, and special tokens added.

## 📈 Training & Evaluation

Run the training script:

```bash
python main.py
```

To evaluate the model using BLEU score:

```bash
python evaluate.py
```

## 🧠 Key Learnings

* Built a neural translation system from scratch using only PyTorch
* Understood and implemented the Bahdanau attention mechanism
* Worked with large-scale bilingual data and sequence preprocessing
* Managed training on limited hardware with resource-aware optimizations

## 📌 Limitations & Future Work

* BLEU score is low due to hardware constraints
* Can be improved by:

  * Increasing model capacity
  * Using pre-trained embeddings (e.g., fastText)
  * Switching to transformer-based architecture
  * Training on cloud GPUs or TPUs

## 🧑‍💼 Author

**Maharshi Jani**

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
