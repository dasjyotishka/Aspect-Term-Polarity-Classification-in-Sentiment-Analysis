# Aspect-Term-Polarity-Classification-in-Sentiment-Analysis
The goal of this project is to implement a classifier that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. The classifier takes as input 3 elements: a sentence, an aspect term occurring in the sentence, and its aspect category. For each input triple, it produces a polarity label: positive, negative or neutral.

# Names of the student working on the Assignment:

1. Jyotishka Das(jyotishka.das@student-cs.fr)
2. Akshay Shastri(akshay.shastri@student-cs.fr)
3. Vanshika Sharma(vanshika.sharma@student-cs.fr)
4. Michele Natacha Ela Essola (michele-natacha.ela-essola@student-cs.fr)

# Dataset:
The dataset is in TSV format, one instance per line.

Each line contains 5 tab-separated fields: the polarity of the opinion (the ground truth polarity label), the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which the term occurs and the opinion is expressed.
For instance, in the first line, the opinion polarity regarding the target term "wait staff", which has the aspect category SERVICE#GENERAL, is negative.In the example of the second line, the sentence is the same but the opinion is about a different aspect and a different target term (pie), and is positive.

There are 12 different aspects categories. The training set (filename: traindata.csv) has this format (5 fields) and contains 1503 lines, i.e. 1503 opinions. The training set was used to train the classifier. A development dataset (filename: devdata.csv) was used to test its performance. It has the same format as the training dataset. It has 376 lines, i.e. 376 opinions. 

# Architecture:

- For this assignment, we have used a pre-trained BERT model (princeton-nlp/unsup-simcse-bert-large-uncased) developed by Princeton NLP group from Hugging Face.
- BERT, which stands for Bidirectional Encoder Representations from Transformers, is a pre-trained language model that has made significant advancements in natural language processing (NLP). Developed by Google researchers in 2018, BERT uses a deep neural network architecture known as the Transformer to encode text data in both directions (i.e., from left-to-right and from right-to-left) and generate a contextualized representation of each word in a sentence.
- The princeton-nlp/unsup-simcse-bert-large-uncased model is a variant of the original BERT model that has been trained on a large corpus of unlabeled text data. This unsupervised learning process allows the model to learn about the structure and meaning of language in a more general way, without relying on specific task-oriented training data.
- One specific application of this model is the SimCSE (Similarity Classification via Sequence Embeddings) method, which utilizes BERT's pre-trained representations to compare the similarity between pairs of text sequences. This can be useful in a variety of NLP tasks, such as text classification, semantic search, and question answering.
-  We derived our features from this BERT model. Additional features were derived from CountVectorizer and LatentDirichletAllocation algorithms. An ensemble of these features were used to train a SVM for the aspect based polarity classification.

# Instructions for Execution:
Execute the testing.py file in the /src folder. It would train the model using trainingdata.csv and give the prediction accuracy from devdata.csv

# Explainations of the Code:

- We started with downloading and importing all the necessary libraries, importing data and the model.
- Then we performed label encoding of the sentiment and kept all the no/nor/not words as this improved radically the sentiment analysis. We also cleaned the data.
- Before we could hand our sentences to BERT, we did some minimal processing to put them in the format it requires:

	1. dataset['review_as'] = dataset.apply(lambda row: add_aspect(row['review'], row['aspect_category']), axis=1): This adds a new column called review_as to the dataset by applying the add_aspect() function to each row of the dataset. The add_aspect() function combines the review and aspect_category columns of each row to create a new string that includes the aspect category information as a special token in the review text.

	2. tokenized = dataset['review_as'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True)): This tokenizes the review_as column using a tokenizer provided by the PyTorch library. The encode() method of the tokenizer converts the text into a list of token ids, with special tokens such as [CLS] and [SEP] added to mark the beginning and end of the sequence.

	3. max_len = max([len(i) for i in tokenized.values]): This calculates the maximum length of the tokenized sequences in the dataset.

	4. padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]): This pads the tokenized sequences with zeros at the end to make them all the same length as the max_len.

	5. attention_mask = np.where(padded != 0, 1, 0): This creates an attention mask by setting the value to 1 for each token in the padded array that is not equal to 0 (i.e., the tokens that were added for padding).

	6. input_ids = torch.tensor(padded): This creates a PyTorch tensor from the padded array.

	7. attention_mask = torch.tensor(attention_mask): This creates a PyTorch tensor from the attention_mask array.

	8. with torch.no_grad():: This context manager turns off gradient tracking to speed up the computations and reduce memory usage.

	9. last_hidden_states = model(input_ids.to(torch.device(device)), attention_mask=attention_mask.to(torch.device(device))): This passes the input_ids and attention_mask tensors to the pre-trained transformer-based model specified in the model variable. The model returns a tuple containing the hidden states of all the tokens and the output of the model's prediction head.

	10. features = last_hidden_states[0][:,0,:].cpu().numpy(): This extracts the first token (which corresponds to the [CLS] token) from the last hidden state of the model and converts it to a numpy array using the cpu().numpy() method. This array contains the final features that can be used as input to a downstream machine learning model.

- The model() function ran our sentences through BERT. The results of the processing was returned into last_hidden_states. We derived our features from this. Additional features were derived from CountVectorizer and LatentDirichletAllocation algorithms. 

- After performing the preprocessing steps on our datasets, we applied our model (SVM) to the sentences.

# Note: 

We also tried out finetuning the BERT model but it yielded similar accuracy to the model we had finally selected for submission. This is because, a finetuned BERT model took a substantially longer time to train.

# Results and Accuracy:

Completed 5 runs.
Dev accs: [86.17, 86.17, 86.17, 86.17, 86.17]
Test accs: [-1, -1, -1, -1, -1]

Mean Dev Acc.: 86.17 (0.00)
Mean Test Acc.: -1.00 (0.00)

Exec time: 195.67 s. ( 39 per run )

