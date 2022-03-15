import fasttext
import itertools
import sys
import pandas as pd
from multiprocessing import Pool

output_dir = '/workspace/datasets'
base_model = 'query_data'
training_file_base = output_dir + '/labeled_query_data_mq_{}.train'
testing_file_base = output_dir + '/labeled_query_data_mq_{}.test'
output_file = output_dir + '/training_results.h5'
min_queries_list = [1, 100, 1000]
epochs = [5, 10, 15, 20, 25]
lrs = [.40, .35, .25, .15, .05]
wordNgrams = [1, 2]
tests = [1, 3, 5]

results = []

'''
# Single Threaded
for(min_queries, epoch, lr, wordNgram) in itertools.product(min_queries_list, epochs, lrs, wordNgrams):
    print('-----------------------------------------')
    print(f'Min Queries: {min_queries} Epochs: {epoch} Learning Rate: {lr} wordNgrams: {wordNgram}')
    cur_result = {'Minimum Queries': min_queries, 'Epochs': epoch, 'Learning Rate': lr, 'wordNgrams': wordNgram}
    loop = True
    while(loop):
        try:
            model = fasttext.train_supervised(input=training_file_base.format(min_queries), lr=lr, epoch=epoch, wordNgrams=wordNgram, verbose=0)
            loop = False
        except:
            print('Trying again...')
    for test in tests:
        (recs, precision, recall) = model.test(testing_file_base.format(min_queries), k=test)
        cur_result[f'Recs@{test}'] = recs
        cur_result[f'P@{test}'] = precision
        cur_result[f'R@{test}'] = recall
    results.append(cur_result)
    print(pd.DataFrame([cur_result]))
'''

# Multi Threaded
def thread_task(args: tuple) -> dict:
    (min_queries, epoch, lr, wordNgram) = args
    print('-----------------------------------------')
    print(f'Min Queries: {min_queries} Epochs: {epoch} Learning Rate: {lr} wordNgrams: {wordNgram}')
    cur_result = {'Minimum Queries': min_queries, 'Epochs': epoch, 'Learning Rate': lr, 'wordNgrams': wordNgram}
    loop = True
    while(loop):
        try:
            model = fasttext.train_supervised(input=training_file_base.format(min_queries), lr=lr, epoch=epoch, wordNgrams=wordNgram, verbose=0)
            loop = False
        except:
            print('Trying again...')
    for test in tests:
        (recs, precision, recall) = model.test(testing_file_base.format(min_queries), k=test)
        cur_result[f'Recs@{test}'] = recs
        cur_result[f'P@{test}'] = precision
        cur_result[f'R@{test}'] = recall

    return cur_result

combos = list(itertools.product(min_queries_list, epochs, lrs, wordNgrams))
with Pool(5) as p:
    results = p.map(thread_task, combos)


print(f'Saving results to {output_file}')
store = pd.HDFStore(output_file)
store['df'] = pd.DataFrame(results)
store.close()
print('Done!')