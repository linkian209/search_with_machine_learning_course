import argparse
import os
import random
import xml.etree.ElementTree as ET
import re
import pandas as pd
from pathlib import Path
from nltk.stem import SnowballStemmer

def transform_name(product_name, stemmer):
    if stemmer:
        return stemmer.stem(product_name)
    retval = product_name.lower()
    # Replace punctuation with spaces
    re.sub(r'[^\w\s]', ' ', retval)
    return retval

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

general.add_argument("--stem", action="store_true", help="Use a stemmer instead of normalizing product name")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate
stem = args.stem

data = {'category':[], 'product':[]}

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                        # Choose last element in categoryPath as the leaf categoryId
                        cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                        # Replace newline chars with spaces so fastText doesn't complain
                        name = child.find('name').text.replace('\n', ' ')
                        stemmer = SnowballStemmer("english") if stem else None
                        data['category'].append(f'__label__{cat}')
                        data['product'].append(transform_name(name, stemmer))

    # Now let's enforce minimum products
    df = pd.DataFrame(data=data)
    if min_products > 1:
        categories = df['category'].value_counts()[lambda x: x < min_products].index
        df.drop(df['category'].isin(categories)[lambda x: x].index, inplace=True)

    # Now lets write out data
    data = df.apply(lambda x: f"__label__{x['category']} {x['product']}", axis=1)
    for datum in data.values:
        output.write(f"{datum}\n")
