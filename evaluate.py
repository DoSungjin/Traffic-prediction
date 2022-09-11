"""
final modification: thkim 220214

mse score
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


road_list = ['10', '100', '101', '120', '121', '140', '150', '160', '200', '201', '251', '270', '300', '301', '351', '352', '370', '400',
             '450', '500', '550', '600', '650', '652', '1000', '1020', '1040', '1100', '1200', '1510', '2510', '3000', '4510', '5510', '6000'] 


def load_csv(path):
    return pd.read_csv(path, skipinitialspace=True)

def evaluate(label, prediction):
    return mean_squared_error(label, prediction, squared=False)

def load_result(path, pred=False):
    result = load_csv(path)

    traffic_lst = []
    time_lst = []
    for i in road_list:
        traffic_lst.extend(list(result[i]))
        time_lst.extend(list(result['timestamp']))
    if pred is False:    # answer
        p_type_lst = []
        for i in road_list:
            p_type_lst.extend(list(result['public']))
    else:    # prediction
        p_type_lst = None
        
    return time_lst, traffic_lst, p_type_lst

def mse(answer_path, pred_path):

    a_id, answer, p_type_lst = load_result(answer_path)
    p_id, pred, _ = load_result(pred_path, pred=True)

    assert a_id == p_id, f'Please match the order with the sample submission : {a_id}'
    assert len(a_id) == len(p_id), 'The number of predictions and answers are not the same'
    assert set(p_id) == set(a_id), 'The prediction ids and answer ids are not the same'

    pub_a_id, pub_answer, prv_a_id, prv_answer = [], [], [], []
    pub_p_id, pub_pred, prv_p_id, prv_pred = [], [], [], []
    
    for idx, t in enumerate(p_type_lst):
        if t:
            pub_a_id.append(a_id[idx])
            pub_answer.append(answer[idx])
            pub_p_id.append(p_id[idx])
            pub_pred.append(pred[idx])            
            
        else:
            prv_a_id.append(a_id[idx])
            prv_answer.append(answer[idx])
            prv_p_id.append(p_id[idx])
            prv_pred.append(pred[idx]) 

    # sort
    pub_answer = pd.DataFrame({'file_name': pub_a_id, 'label': pub_answer})
    prv_answer = pd.DataFrame({'file_name': prv_a_id, 'label': prv_answer})
    pub_pred = pd.DataFrame({'file_name': pub_p_id, 'label': pub_pred})
    prv_pred = pd.DataFrame({'file_name': prv_p_id, 'label': prv_pred})
    
    score = evaluate(list(pub_answer['label']), list(pub_pred['label']))
    pScore = evaluate(list(prv_answer['label']), list(prv_pred['label']))
    
    return score, pScore

if __name__ == '__main__':
    answer = sys.argv[1]
    pred = sys.argv[2]

    try:
        import time
        start = time.time()
        score, pScore = mse(answer, pred)
        print(f'score={score},pScore = {pScore}')
        print(f'Elapsed Time: {time.time() - start}')

    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()
