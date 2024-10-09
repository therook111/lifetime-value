
COMPANIES = [
    '10000', '101200010', '101410010', '101600010', '102100020', '102700020',
    '102840020', '103000030', '103338333', '103400030', '103600030',
    '103700030', '103800030', '104300040', '104400040', '104470040',
    '104900040', '105100050', '105150050', '107800070'
]
CATEGORICAL_FEATURES = (['chain', 'dept', 'category', 'brand', 'productmeasure'])
CHUNK_SIZE = 10**6
data_dir = 'extracted/transactions.csv'
NUMERIC_FEATURES = ['log_calibration_value'] 
BATCH_SIZE = 1024