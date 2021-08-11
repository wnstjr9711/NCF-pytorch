from load_data import *
from train import *


def get_prediction(user_id_list, movie_id_list):
    dataset = DatasetLoader()
    movies_df = get_movies_df()
    moviename_dict = movies_df.to_dict()['title']
    if not os.path.exists(config['model_path'] + 'ncf.pth'):
        model_train(dataset)
    my_model = NCF(dataset.num_users, dataset.num_movies, config['hidden_layers'], config['dropouts'],
                   config['num_factors'], config['embedding_dropout'])
    my_model.load_state_dict(torch.load(config['model_path'] + 'ncf.pth'))

    processed_test_input_df = pd.DataFrame({
        'user_id': [dataset.user_to_index[x] for x in user_id_list],
        'movie_id': [dataset.movie_to_index[x] for x in movie_id_list]
    })
    # 학습한 모델 load하기
    pred_results = [float(x) for x in my_model.predict(users=torch.LongTensor(processed_test_input_df.user_id.values),
                                                       movies=torch.LongTensor(processed_test_input_df.movie_id.values))]
    result_df = pd.DataFrame({
        'userId': user_id_list,
        'movieId': movie_id_list,
        'movieName': [moviename_dict[x] for x in movie_id_list],
        # 'genres': [genres_dict[x] for x in movie_id_list],
        'pred_ratings': pred_results
    })

    return result_df.sort_values(by='pred_ratings', ascending=False)
