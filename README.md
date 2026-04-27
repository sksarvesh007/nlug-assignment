# Named Entity Recognition (NER) Case Study

## Information Extraction from News Articles

### Objective
Identify and classify entities such as person, organization, location, and date from text using state-of-the-art NLP techniques.

### Dataset
CoNLL-2003 NER dataset - a standard benchmark for named entity recognition.

## Project Structure

```
.
├── requirements.txt              # Python dependencies
├── test_spacy_ner.py            # Test script for spaCy NER
├── test_transformers_ner.py     # Test script for Hugging Face Transformers NER
├── fine_tune_bert_ner.py        # BERT fine-tuning script
├── ner_case_study.ipynb         # Comprehensive Jupyter notebook
└── README.md                    # This file
```

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Option 1: Run Test Scripts

#### Test spaCy NER
```bash
python test_spacy_ner.py
```

#### Test Hugging Face Transformers NER
```bash
python test_transformers_ner.py
```

#### Fine-tune BERT (requires GPU for reasonable training time)
```bash
python fine_tune_bert_ner.py
```

### Option 2: Use Jupyter Notebook

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open `ner_case_study.ipynb`

3. Run cells sequentially to:
   - Test spaCy NER
   - Test Hugging Face Transformers NER
   - Fine-tune BERT on CoNLL-2003
   - Deploy the model

## Components

### 1. spaCy NER
- Fast and easy-to-use NER with pre-trained models
- Supports multiple entity types: PERSON, ORG, GPE, LOC, DATE, MONEY, etc.

### 2. Hugging Face Transformers NER
- State-of-the-art NER using pre-trained BERT models
- Uses `dbmdz/bert-large-cased-finetuned-conll03-english` model
- Fine-tuned on CoNLL-2003 dataset

### 3. BERT Fine-tuning
- Custom training on CoNLL-2003 dataset
- Improved accuracy for specific use cases
- Includes evaluation metrics (precision, recall, F1)

### 4. Model Deployment
- Ready-to-use deployment code
- Batch processing support
- Easy integration with applications

## Entity Types

### CoNLL-2003 Labels
- **PER**: Person names
- **ORG**: Organizations, companies, institutions
- **LOC**: Locations
- **MISC**: Miscellaneous entities

### spaCy Labels
- **PERSON**: People, including fictional
- **ORG**: Companies, agencies, institutions
- **GPE**: Countries, cities, states
- **LOC**: Non-GPE locations
- **DATE**: Dates and periods
- **MONEY**: Monetary values
- **CARDINAL**: Numerals

## Training Configuration

The BERT fine-tuning uses the following hyperparameters:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3
- Weight decay: 0.01
- Optimizer: AdamW

Adjust these in the notebook or script based on your computational resources.

## Notes

- The fine-tuning process requires significant computational resources (GPU recommended)
- For testing purposes, reduce `num_train_epochs` to 1
- The pre-trained model can be used directly without fine-tuning for quick results

## Next Steps

1. Install dependencies and run the test scripts
2. Open the notebook and explore different NER approaches
3. Uncomment training code to fine-tune the model on your data
4. Deploy the model using the provided deployment code
5. Integrate with your application for real-time NER

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recommended for training)
- GPU (optional but recommended for training)
