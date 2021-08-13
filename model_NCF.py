from config import *
import torch.nn.functional as f


class NCF(nn.Module):
    def __init__(self, n_users, n_apt, hidden, dropouts, n_factors, embedding_dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.apt_emb = nn.Embedding(n_apt, n_factors)
        self.drop = nn.Dropout(embedding_dropout)
        self.hidden_layers = nn.Sequential(*list(self.generate_layers(n_factors*2, hidden, dropouts)))
        self.fc = nn.Linear(hidden[-1], 1)

    @staticmethod
    def generate_layers(n_factors, hidden, dropouts):
        assert len(dropouts) == len(hidden)
        idx = 0
        while idx < len(hidden):
            if idx == 0:
                yield nn.Linear(n_factors, hidden[idx])
            else:
                yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(dropouts[idx])
            idx += 1

    def forward(self, users, apt, min_rating=0, max_rating=1):
        concat_features = torch.cat([self.user_emb(users), self.apt_emb(apt)], dim=1)
        x = f.relu(self.hidden_layers(concat_features))
        # 0과 1사이의 숫자로 나타낸다
        out = torch.sigmoid(self.fc(x))
        return out

    def predict(self, users, apt):
        # return the score
        output_scores = self.forward(users, apt)
        return output_scores

