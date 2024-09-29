import subprocess
import os
from zipfile import ZipFile
import gzip
import shutil

PATH = 'data/'

if __name__ == '__main__':


    subprocess.run("mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json", shell=True, check=True)
    subprocess.run(['kaggle', 'competitions', 'download', '-c', 'acquire-valued-shoppers-challenge', '-p', 'data'])
    
    
    # Extract the zip file
    with ZipFile(PATH + os.listdir(PATH)[0], 'r') as zip:
        zip.extractall('extracted/')

    # Extract the transactions zip file
    with gzip.open('extracted/transactions.csv.gz', 'rb') as f_in:
        with open('extracted/transactions.csv', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Clean-up
    os.remove(PATH + os.listdir(PATH)[0])
    for directory in os.listdir('extracted/'):
        if directory != 'transactions.csv':
            os.remove('extracted/' + directory)
    
    
