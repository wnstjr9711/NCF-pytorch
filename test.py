from predict import *

movie_id_list = [10253, 10102, 10007]
user_id = 11242
user_id_list = [user_id] * len(movie_id_list)


print(get_prediction(user_id_list, movie_id_list))