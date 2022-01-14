from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json

MODEL = "" # to be filled
data = json.load(open("constructii-qset-v1.json", encoding="utf-8"))

qids = []`
dataset = []
for item in data:
    if item['qid'] in qids:
        # search labels that have multiple qids, no need to calculate similarity on a label that has just 1 qid,
        # it will provide 100% everytime
        qids_set = set([x['qid'] for ind, x in enumerate(data) if data[ind]['label'] == item['label']])
        if len(qids_set) > 1:
            dataset.append(item)
    else:
        qids.append(item['qid'])

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL)
acc = 0
c = 0
total = len(dataset)
for qa_test in dataset:
    max_cosine = 0
    c += 1
    qid = 0
    input2 = tokenizer(qa_test['question'], padding="max_length", max_length=128, return_tensors='pt')
    qav = model(**input2).last_hidden_state.flatten().reshape(1, -1).tolist()
    for qa_train in [x for ind, x in enumerate(dataset) if
                     dataset[ind]['label'] == qa_test['label'] and dataset[ind]['question'] != qa_test["question"]]:
        cosine = cosine_similarity(qav, model(**tokenizer(qa_train['question'], padding="max_length", max_length=128,
                                                          return_tensors='pt')).last_hidden_state.flatten().reshape(1, -1).tolist())
        if max_cosine < cosine[0]:
            qid = qa_train['qid']
            max_cosine = cosine[0]
        # given the large number of similar questions in one label, 0.95 cosine would be a great similarity,
        # no need to go through all set
        if max_cosine > 0.95:
            break
    if qid == qa_test['qid']:
        acc += 1
        print(c, acc / c)

print("Accuracy is {}".format(acc / total))

