from transformers import AutoTokenizer, AutoModel
import json
import tensorflow as tf


MODEL = "bert-base-multilingual-cased"  # to be filled
data = json.load(open("qa_datasets/construction-qset-v1.json", encoding="utf-8"))

qids = []
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
acc = c = 0
total = len(dataset)
for qa_test in dataset:
    max_cosine = qid = 0
    c += 1
    tokens = tokenizer(qa_test['question'], padding="max_length", max_length=128, return_tensors='pt')
    a = tf.constant(model(**tokens).last_hidden_state.reshape(1, -1).tolist())
    for qa_train in [x for ind, x in enumerate(dataset) if
                     dataset[ind]['label'] == qa_test['label'] and dataset[ind]['question'] != qa_test["question"]]:
        b = tf.constant(model(**tokenizer(qa_train['question'], padding="max_length", max_length=128,
                                          return_tensors='pt')).last_hidden_state.reshape(1, -1).tolist())
        similarity = tf.reduce_sum(a[:, tf.newaxis] * b, axis=-1)
        similarity /= tf.norm(a[:, tf.newaxis], axis=-1) * tf.norm(b, axis=-1)

        cosine = similarity.numpy()
        # given the large number of similar questions in one label, 0.95 cosine would be a great similarity,
        # no need to go through all set
        if cosine[0] > 0.95:
            qid = qa_train['qid']
            break
        elif max_cosine < cosine[0]:
            qid = qa_train['qid']
            max_cosine = cosine[0]
    if qid == qa_test['qid']:
        acc += 1
        print(c, acc / c)

print("Accuracy is {}".format(acc / total))
