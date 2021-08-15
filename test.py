from model_predict import *

movie_id_list = [i for i in range(10001, 10040) if i != 10010 and i != 10031 and i != 10032]
user_id = 1000
user_id_list = [user_id] * len(movie_id_list)


print(get_prediction(user_id_list, movie_id_list))