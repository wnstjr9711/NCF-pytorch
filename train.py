import math
from model_NCF import *
from tqdm import tqdm


class BatchIterator:
    def __init__(self, x, y, batch_size=32, shuffle=True):
        x, y = np.asarray(x), np.asarray(y)

        if shuffle:
            index = np.random.permutation(x.shape[0])
            x, y = x[index], y[index]

        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = int(math.ceil(x.shape[0] // batch_size))
        self._current = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self._current >= self.n_batches:
            raise StopIteration()
        k = self._current
        self._current += 1
        bs = self.batch_size
        return self.x[k * bs:(k + 1) * bs], self.y[k * bs:(k + 1) * bs]


def batches(x, y, bs=32, shuffle=True):
    for x_batch, y_batch in BatchIterator(x, y, bs, shuffle):
        x_batch = torch.LongTensor(x_batch)
        y_batch = torch.FloatTensor(y_batch)
        yield x_batch, y_batch.view(-1, 1)


def model_train(ds):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x_train, y_train = ds.generate_trainset()
    x_valid, y_valid = ds.generate_valset()
    print(f'TrainSet Info: {ds.num_users} users, {ds.num_movies} movies')

    model = NCF(
        n_users=ds.num_users, n_movies=ds.num_movies,
        n_factors=config['num_factors'], hidden=config['hidden_layers'],
        embedding_dropout=config['embedding_dropout'], dropouts=config['dropouts']
    )
    model.to(device)

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    max_patience = config['total_patience']
    num_patience = 0
    best_loss = np.inf

    criterion = nn.MSELoss(reduction='sum')
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    result = dict()
    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        for batch in batches(x_train, y_train, shuffle=True, bs=batch_size):
            x_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            # with torch.no_grad() 와 동일한 syntax 입니다
            with torch.set_grad_enabled(True):
                outputs = model(x_batch[:, 0], x_batch[:, 1], ds.min_rating, ds.max_rating)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            training_loss += loss.item()
        result['train'] = training_loss / len(x_train)

        # Apply Early Stopping criteria and save best model params
        val_outputs = model(torch.LongTensor(x_valid.user.values).to(device),
                            torch.LongTensor(x_valid.movie.values).to(device), ds.min_rating, ds.max_rating)
        val_loss = criterion(val_outputs.to(device), torch.FloatTensor(y_valid.values).view(-1, 1).to(device))
        result['val'] = float((val_loss / len(x_valid)).data)

        if val_loss < best_loss:
            print('Save new model on epoch: %d' % (epoch + 1))
            best_loss = val_loss
            result['best_loss'] = val_loss
            if not os.path.exists(config['model_path']):
                os.mkdir(config['model_path'])
            torch.save(model.state_dict(), config['model_path'] + 'ncf.pth')
            num_patience = 0
        else:
            num_patience += 1

        print(f'[epoch: {epoch+1}] train: {result["train"]} - val: {result["val"]}')

        if num_patience >= max_patience:
            print(f"Early Stopped after epoch {epoch+1}")
            break
    return result
