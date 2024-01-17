from NeuroX.neurox.data.extraction import transformers_extractor
from NeuroX.neurox.data.writer import ActivationsWriter
from transformers import AutoTokenizer, AutoModel
import json
import torch
import re
from IPython.display import clear_output

# code from https://github.com/fdalvi/NeuroX/blob/master/neurox/data/extraction/transformers_extractor.py with corrections for different probing types and 'local' model path

def get_model_and_tokenizer(model_path, tokenizer_name='cointegrated/rubert-tiny2', device="cpu", random_weights=False):
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()
    return model, tokenizer


def extract_representations(
    model_path,
    tokenizer_name,
    input_corpus,
    output_file,
    device="cpu",
    aggregation="last",
    output_type="json",
    random_weights=False,
    ignore_embeddings=False,
    decompose_layers=False,
    filter_layers=None,
    dtype="float32",
    include_special_tokens=False,
    probing_type='cls',
    ):

    print(f"Loading model: {model_path}")
    model, tokenizer = get_model_and_tokenizer(
        model_path=model_path, tokenizer_name='cointegrated/rubert-tiny2', device=device, random_weights=random_weights
    )

    print("Reading input corpus")
    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return
    
    print("Extracting representations from model")

    if probing_type == 'cls':
        a = {}
        for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
            print(sentence_idx, sentence)
            inputs = tokenizer(sentence, return_tensors='pt')
            outputs = model(**inputs)
            hidden_states = outputs[2] 
            cls_tokens = [layer[0][0] for layer in hidden_states] 
            cls_tokens_concatenated = torch.cat(cls_tokens, dim=0)
            cls_tokens_concatenated = cls_tokens_concatenated.detach().numpy()
            a[f"{sentence_idx}"] = cls_tokens_concatenated.tolist()

        with open(output_file, 'w') as f:
            json.dump(a, f)

    elif probing_type == 'avg' or probing_type == 'token':
        print("Preparing output file")
        writer = ActivationsWriter.get_writer(
            output_file,
            filetype=output_type,
            decompose_layers=decompose_layers,
            filter_layers=filter_layers,
            dtype=dtype,
        )
        tokenization_counts = {}  # Cache for tokenizer rules
        for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
            hidden_states, extracted_words = transformers_extractor.extract_sentence_representations(
                sentence,
                model,
                tokenizer,
                device=device,
                include_embeddings=(not ignore_embeddings),
                aggregation=aggregation,
                dtype=dtype,
                include_special_tokens=include_special_tokens,
                tokenization_counts=tokenization_counts,
            )

            print("Hidden states: ", hidden_states.shape)
            print("# Extracted words: ", len(extracted_words))

            writer.write_activations(sentence_idx, extracted_words, hidden_states)

        writer.close()

class GetEmbeddings:
    """"
    Receives .txt files with sentences and computes embeddings for them.
    """
    
    def __init__(self, path_trdata, path_tedata, path_work, probing_type):
        
        self.path_trdata = path_trdata
        self.path_tedata = path_tedata
        self.path_work = path_work
        self.probing_type = probing_type
        
        self.category = re.search(r'[a-zA-Z]+_[a-zA-Z]+(?=.txt)', path_trdata)[0]
        self.dataset = re.search(r'(?<=_)[a-zA-Z]+_[a-zA-Z]+(?=\/)', path_trdata)[0]
        
    def jsons(self, model):
        
        path = self.path_work + f'/large_data_{self.dataset}/data_{self.category}'
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        print()
        
        extract_representations(model_path=model,
        tokenizer_name='cointegrated/rubert-tiny2',
        input_corpus=self.path_trdata,
        output_file=path+'/activations_train.json',
        aggregation="average", #last, first   
        device=device,
        probing_type=self.probing_type)
        clear_output(wait=False)
        
        extract_representations(model_path=model,
        tokenizer_name='cointegrated/rubert-tiny2',
        input_corpus=self.path_tedata,
        output_file=path+'/activations_te.json',
        aggregation="average", #last, first
        device=device,
        probing_type=self.probing_type)
        clear_output(wait=False)