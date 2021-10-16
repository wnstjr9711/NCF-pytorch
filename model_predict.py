from load_data import *
from model_train import *
from config import request
import RecSys_FM.load_data as FM

request_id = 0


def get(uid):
    return FM.get_prediction(uid)


def get_prediction(uid):
    global request_id
    request_id += 1
    request[uid] = request_id
    score = dict()
    if len(db_connect.get_user_average(uid)) != 0:
        dataset = DatasetLoader()
        apt_id_list = dataset.unique_apt
        user_id_list = [uid] * len(apt_id_list)
        model_train(dataset, uid, request_id)
        my_model = NCF(dataset.num_users, dataset.num_apt, config['hidden_layers'], config['dropouts'],
                       config['num_factors'], config['embedding_dropout'])
        my_model.load_state_dict(torch.load(os.path.join(config['model_path'], 'ncf.pth')))

        processed_test_input_df = pd.DataFrame({
            'user_id': [dataset.user_to_index[x] for x in user_id_list],
            'apt_id': [dataset.apt_to_index[x] for x in apt_id_list]
        })

        # 학습한 모델 load하기
        pred_results = [float(x) for x in my_model.predict(users=torch.LongTensor(processed_test_input_df.user_id.values),
                                                           apt=torch.LongTensor(processed_test_input_df.apt_id.values))]

        result_df = pd.DataFrame({
            'userId': user_id_list,
            'aptId': apt_id_list,
            'pred_ratings': pred_results
        })
        for i, j in enumerate(reversed(get(uid))):
            score[j] = i * 0.8
        for i, j in enumerate(reversed(list(result_df.sort_values(by='pred_ratings', ascending=False)['aptId']))):
            score[j] += i * 0.2


        user_average = np.mean(list(map(lambda x: x['price'], db_connect.get_user_average(uid))))

        order = list(map(lambda x: int(x), sorted(score, key=score.get, reverse=True)))

        result = list()
        price = dict()
        for i in db_connect.get_apt_price():
            price[i['id']] = i['price']

        for i in order:
            if abs(price[i] - user_average) <= 30000:
                result.append(i)
            if len(result) == 10:
                break
        return result
    else:
        return [1,2,3,4,5,6,7,8,9,10]


