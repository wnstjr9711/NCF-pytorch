from sklearn.model_selection import train_test_split
from config import *
import db_connect


class DatasetLoader:
    def __init__(self):
        self.train_df, val_temp_df = self.read_data()

        self.min_rating = 0
        self.max_rating = 1

        self.unique_users = self.train_df.user.unique()
        self.num_users = len(self.unique_users)
        self.user_to_index = {original: idx for idx, original in enumerate(self.unique_users)}

        self.unique_apt = self.train_df.apt.unique()
        self.num_apt = len(self.unique_apt)
        self.apt_to_index = {original: idx for idx, original in enumerate(self.unique_apt)}

        self.val_df = val_temp_df[val_temp_df.user.isin(self.unique_users) & val_temp_df.apt.isin(self.unique_apt)]

    @staticmethod
    def read_data():
        # df = pd.read_csv(os.path.join(config['data_path'], 'aptlog.csv'), usecols=['user', 'apt'])
        df = pd.DataFrame(db_connect.get_log())
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=1234, shuffle=True)
        train_df['searched'] = [1 for i in range(len(train_df))]
        val_df['searched'] = [1 for i in range(len(val_df))]
        return train_df, val_df

    def generate_trainset(self):
        # user 0, 0, 0, 1,2, 3,3, -> movie: 0,0,0,0,0,0,
        x_train = pd.DataFrame({'user': self.train_df.user.map(self.user_to_index),
                                'apt': self.train_df.apt.map(self.apt_to_index)})
        y_train = self.train_df['searched'].astype(np.float32)
        return x_train, y_train

    def generate_valset(self):
        x_val = pd.DataFrame({'user': self.val_df.user.map(self.user_to_index),
                              'apt': self.val_df.apt.map(self.apt_to_index)})
        y_val = self.val_df['searched'].astype(np.float32)
        return x_val, y_val

