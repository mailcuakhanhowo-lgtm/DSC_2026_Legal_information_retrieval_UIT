import json
import os
import numpy as np

reference_dir = os.path.join('/app/input', 'ref')
prediction_dir = os.path.join('/app/input', 'res')
score_dir = '/app/output'

def read_json(file):
    with open(file, encoding="utf8") as f:
        return json.load(f)
    f.close()

def eval_retrieval(y_pred, y_true):
    try:
        y_pred = {k : v['answer'] for k, v in y_pred.items()}
        y_true = {k : v for k, v in y_true.items()}
        
        ids_preds = []
        ids_truth = []
        for k, v in y_pred.items():
            ids_preds.append(k)

        for k, v in y_true.items():
            ids_truth.append(k)

        if len(ids_preds) != len(ids_truth):
            print("Samples in predict not match with reference")
            raise Exception

        recall = np.array([len(set(y_true[k]) & set(y_pred.get(k, set())))/len(y_true[k]) if len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5 else 0 for k in ids_truth]).mean()
        precision = np.array([len(set(y_true[k]) & set(y_pred.get(k, set())))/len(y_pred[k]) if len(y_pred.get(k)) > 0 and len(y_pred.get(k)) <= 5 else 0 for k in ids_preds]).mean()

        return {'precision': precision, 'recall': recall}
    except Exception as e:
        print(e)
        raise e
        # return {'mrr': 0.0, 'recall': 0.0}


def main():
    scores = {'precision': 0.0, 'recall': 0.0}

    metadata = read_json(os.path.join(reference_dir, 'metadata.json'))
    truth = read_json(os.path.join(reference_dir, metadata['files']['reference']))

    if metadata['files'].get('input'):
        input_data = read_json(os.path.join(prediction_dir, metadata['files']['input']))
        eval_scores = eval_retrieval(input_data, truth)
        scores.update(eval_scores)
    else:
        print("Input file name not correct. ")
        raise Exception

    print("Final scores:", scores)

    with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
        score_file.write(json.dumps(scores))
    score_file.close()
    
    return 0

if __name__ == "__main__": 
    exit(main())
    # pass
        