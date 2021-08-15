from load_data import *
from model_train import *


def get_prediction(user_id_list, apt_id_list):
    dataset = DatasetLoader()
    apt_df = get_apt_df()
    # dict_apt_name = apt_df.to_dict()['title']
    if not os.path.exists(config['model_path'] + 'ncf.pth'):
        model_train(dataset)
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
        # 'aptName': [dict_apt_name[x] for x in apt_id_list],
        'pred_ratings': pred_results
    })

    return result_df.sort_values(by='pred_ratings', ascending=False)
