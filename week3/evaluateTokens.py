import fasttext

model = '/workspace/datasets/fasttext/title_model'
modelBin = f'{model}.bin'
inputData = '/workspace/datasets/fasttext/titles.txt'

tokens = [
    'iphone', 'galaxy', 'lg', 'bose', 'denon',
    'beats', 'imac', 'chromebook', 'asus', 'headphones',
    'tv', 'tablet', 'monitor', 'bluetooth', 'mouse',
    'nintendo', 'xbox', 'bluray', 'intel', 'amd'
]

model = fasttext.train_unsupervised(input=inputData, model='skipgram', verbose=0, minCount=10)

for token in tokens:
    neighbors = model.get_nearest_neighbors(token)
    data = []
    for (score, neighbor) in neighbors:
        data.append(f'{neighbor}: {score}')
    print(token)
    print('-------------------------')

    print('\n'.join([f'* {x}' for x in data]))
    print()