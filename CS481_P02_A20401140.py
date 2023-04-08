### 
### Import libraries
### 

try:
    import csv
    import re
    from nltk.corpus import stopwords
    import nltk
    import math
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from prettytable import PrettyTable
    from collections import defaultdict
    import sys
except Exception as e:
    print("An error occurred:", e)
    print('Error loading required libraries. Please check requirements.txt to make sure you have the required dependencies!')
    print('Program will exit. Dependencies required to work.')  

print('Sada, Mohammad Firas, A20401140 solution:')

IGNORE = 'NO'

if len(sys.argv) > 1:
    param = sys.argv[1]
    if param.upper() == 'YES' or param.upper() == 'NO':
        IGNORE = param.upper()
    else:
        raise ValueError("The IGNORE parameter should be 'YES' or 'NO'.")
if IGNORE == 'YES':
    print('Ignored pre-processing step: REMOVING STOP WORDS')


### 
### Loading the dataset
### 

try:
    print('\nLoading the dataset...\nThis may take a few minutes...')
    with open('input/dataset.csv', 'r') as file:
        train_file_lines = file.readlines()
        train_file_lines = [line.strip()[1:] for line in train_file_lines]
except:
    print('Error loading the dataset. Please make sure input/dataset.csv exists!')



#test_file_lines = test_file.readlines()

print('Finished loading the dataset...\nPreprocessing dataset...')

### 
### The assignment asks for a 80/20 split, therefore, we combine the two sets and split them later
### The dataset is huge, and 4,000,000 is a very huge length
### We chnage our approach and take a subset of 100,000

#dataset = train_file_lines + test_file_lines
dataset = train_file_lines

### 
### Data preparation: decode and extract labels
### 

dataset_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in dataset]



count_0 = 0
count_1 = 0
for elem in dataset_labels:
    if elem == 0:
        count_0 += 1
    elif elem == 1:
        count_1 += 1

### 
### Data preparation: extract taining data
### 

dataset = [x.split(' ', 1)[1][:-1] for x in dataset]

### 
### Data preparation: cleaning out URLs
### 

for i in range(len(dataset)):
    if 'www.' in dataset[i] or 'http:' in dataset[i] or 'https:' in dataset[i] or '.com' in dataset[i]:
        dataset[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", dataset[i])

### 
### Data preprocessing
### 

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
for i in range(len(dataset)):
    words = nltk.tokenize.wordpunct_tokenize(dataset[i].lower())
    if IGNORE == 'NO':
        words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    dataset[i] = words


class SentimentClassifier:

    def __init__(self):
        self.vocab = set()
        self.class_probs = {}
        self.word_counts = {}
        self.class_word_counts = {}

    def fit(self, dataset, dataset_labels):
        assert len(dataset) == len(dataset_labels), "Dataset and labels not same length."
        split_idx = int(0.8 * len(dataset))
        train_data, train_labels = dataset[:split_idx], dataset_labels[:split_idx]
        test_data, test_labels = dataset[split_idx:], dataset_labels[split_idx:]
        num_examples = len(train_labels)
        classes, counts = zip(*dict.fromkeys(train_labels, 0).items())
        for c in train_labels:
            self.class_probs[c] = (train_labels.count(c) + 1) / (num_examples + len(classes))
        self.word_counts = {c: {} for c in classes}
        self.class_word_counts = {c: 0 for c in classes}
        for x, c in zip(train_data, train_labels):
            self.class_word_counts[c] += len(x)
            for word in x:
                self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0
                self.word_counts[c][word] += 1
        # print("Vocabulary:", self.vocab)
        # print("Class probabilities:", self.class_probs)
        # print("Word counts:", self.word_counts)
        # print("Class word counts:", self.class_word_counts)
    def predict(self, dataset):
        predictions = []
        for x in dataset:
            scores = {c: math.log(self.class_probs[c]) for c in self.class_probs}
            for word in x:
                if word not in self.vocab:
                    continue
                for c in self.class_probs:
                    count = self.word_counts[c].get(word, 0) + 1
                    scores[c] += math.log(count / (self.class_word_counts[c] + len(self.vocab)))
            predictions.append(max(scores, key=scores.get))
        return predictions
    def predict_single(self, text):
        words = text.strip().split()
        score = defaultdict(float)
        for c in self.class_probs:
            score[c] = math.log(self.class_probs[c])
            for word in words:
                if word in self.vocab:
                    count = self.word_counts[c].get(word, 0) + 1
                    score[c] += math.log(count / (self.class_word_counts[c] + len(self.vocab)))
            pred_label = max(score, key=score.get)
            prob_positive = math.exp(score[1]) / (math.exp(score[1]) + math.exp(score[0]))
            prob_negative = math.exp(score[0]) / (math.exp(score[1]) + math.exp(score[0]))
        return pred_label, prob_positive, prob_negative

    
    def predict_text(self, text):
        pred_label, prob_positive, prob_negative = self.predict_single(text)
        if pred_label == 1:
            pred_label = 'Positive'
        else:
            pred_label = 'Negative'
        print(f'Sentence S:\n\n{text}\nwas classified as {pred_label}.')
        print(f"P(Positive|S): {prob_positive:.2f}\nP(Negative|S): {prob_negative:.2f}")

print('\nTraining classifier...')
print('The dataset is huge. We take a subset of 100,000 data points.\nThis might take a few minutes...')

clf = SentimentClassifier()
clf.fit(dataset, dataset_labels)
predictions = clf.predict(dataset)

print('Testing classifier...')

split_idx = int(0.8 * len(dataset))
test_data, test_labels = dataset[split_idx:], dataset_labels[split_idx:]
test_predictions = clf.predict(test_data)

tp, tn, fp, fn = 0, 0, 0, 0
for true_label, pred_label in zip(test_labels, test_predictions):
    if true_label == 1 and pred_label == 1:
        tp += 1
    elif true_label == 0 and pred_label == 0:
        tn += 1
    elif true_label == 1 and pred_label == 0:
        fn += 1
    elif true_label == 0 and pred_label == 1:
        fp += 1

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
npv = tn / (tn + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
f_score = 2 * precision * sensitivity / (precision + sensitivity)

print('\nTest results/metrics:')

table = PrettyTable()
table.field_names = ["Metric", "Value"]
table.add_row(["True positives", tp])
table.add_row(["True negatives", tn])
table.add_row(["False positives", fp])
table.add_row(["False negatives", fn])
table.add_row(["Sensitivity (recall)", "{:.2f}".format(sensitivity)])
table.add_row(["Specificity", "{:.2f}".format(specificity)])
table.add_row(["Precision", "{:.2f}".format(precision)])
table.add_row(["Negative predictive value", "{:.2f}".format(npv)])
table.add_row(["Accuracy", "{:.2f}".format(accuracy)])
table.add_row(["F-score", "{:.2f}".format(f_score)])
print(table)
print('\n')
while True:
    text = input("Enter your sentence:\n\n")
    if text == 'N':
        break
    clf.predict_text(text)
    while True:
        continue_prompt = input("\nDo you want to enter another sentence [Y/N]? ").upper()
        if continue_prompt == 'Y' or continue_prompt == 'N':
            break
        else:
            print("Invalid input. Please enter Y or N.")
    if continue_prompt != 'Y':
        break



