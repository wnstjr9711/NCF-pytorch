from sklearn.model_selection import train_test_split
from config import *


class DatasetLoader:
    def __init__(self):
        self.train_df, val_temp_df = self.read_data(config['data_path'])

        self.min_rating = min(self.train_df.rate)
        self.max_rating = self.train_df.rate.max()

        self.unique_users = self.train_df.user.unique()
        self.num_users = len(self.unique_users)
        self.user_to_index = {original: idx for idx, original in enumerate(self.unique_users)}
        # 0 1 0 0 0 ... 0

        self.unique_movies = self.train_df.movie.unique()
        self.num_movies = len(self.unique_movies)
        self.movie_to_index = {original: idx for idx, original in enumerate(self.unique_movies)}

        self.val_df = val_temp_df[val_temp_df.user.isin(self.unique_users) & val_temp_df.movie.isin(self.unique_movies)]

    @staticmethod
    def read_data(data_path):
        df = pd.read_csv(os.path.join(data_path, 'rates.csv'))
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=1234, shuffle=True)
        return train_df, val_df

    def generate_trainset(self):
        # user 0, 0, 0, 1,2, 3,3, -> movie: 0,0,0,0,0,0,
        x_train = pd.DataFrame({'user': self.train_df.user.map(self.user_to_index),
                                'movie': self.train_df.movie.map(self.movie_to_index)})
        y_train = self.train_df['rate'].astype(np.float32)

        return x_train, y_train

    def generate_valset(self):
        x_val = pd.DataFrame({'user': self.val_df.user.map(self.user_to_index),
                              'movie': self.val_df.movie.map(self.movie_to_index)})
        y_val = self.val_df['rate'].astype(np.float32)
        return x_val, y_val


def get_movies_df():
    # Load all related dataframe
    movies_df = pd.read_csv(os.path.join(config['data_path'], 'movies.txt'), sep='\t', encoding='utf-8')
    movies_df = movies_df.set_index('movie')

    castings_df = pd.read_csv(os.path.join(config['data_path'], 'castings.csv'), encoding='utf-8')
    countries_df = pd.read_csv(os.path.join(config['data_path'], 'countries.csv'), encoding='utf-8')
    genres_df = pd.read_csv(os.path.join(config['data_path'], 'genres.csv'), encoding='utf-8')

    # Get genre information
    genres = [(list(set(x['movie'].values))[0], '/'.join(x['genre'].values)) for index, x in genres_df.groupby('movie')]
    combined_genres_df = pd.DataFrame(data=genres, columns=['movie', 'genres'])
    combined_genres_df = combined_genres_df.set_index('movie')

    # Get castings information
    castings = [(list(set(x['movie'].values))[0], x['people'].values) for index, x in castings_df.groupby('movie')]
    combined_castings_df = pd.DataFrame(data=castings, columns=['movie', 'people'])
    combined_castings_df = combined_castings_df.set_index('movie')

    # Get countries for movie information
    countries = [(list(set(x['movie'].values))[0], ','.join(x['country'].values)) for index, x in
                 countries_df.groupby('movie')]
    combined_countries_df = pd.DataFrame(data=countries, columns=['movie', 'country'])
    combined_countries_df = combined_countries_df.set_index('movie')

    movies_df = pd.concat([movies_df, combined_genres_df, combined_castings_df, combined_countries_df], axis=1)
    return movies_df
