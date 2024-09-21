import requests
import pandas as pd
import time
import datetime
import os
from logs import Logger
from multiprocessing import Pool, Manager
from tqdm import tqdm
import yaml

# Configuration du logger
logger = Logger().getLogger("dataDownloader")

class BinanceDataDownloader:

    def __init__(self, symbols, interval, startDate, endDate, outputDir='data'):
        self.symbols = symbols
        self.interval = interval
        self.startDate = startDate
        self.endDate = endDate
        self.outputDir = outputDir

    def getBinanceKlines(self, symbol, interval, startTime, endTime=None):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': startTime,
            'endTime': endTime,
            'limit': 1000
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data

    def fetchData(self, args):
        symbol, interval, startStr, endStr, progressQueue = args
        startTime = int(datetime.datetime.strptime(startStr, "%d/%m/%Y").timestamp() * 1000)
        endTime = int(datetime.datetime.strptime(endStr, "%d/%m/%Y").timestamp() * 1000)
        
        allKlines = []
        while startTime < endTime:
            klines = self.getBinanceKlines(symbol, interval, startTime, endTime)
            if len(klines) == 0 or not isinstance(klines,list):
                logger.info(f'Completed fetching data for {symbol}')
                break
            allKlines.extend(klines)
            startTime = klines[-1][0] + 1 if klines else startTime
            progressQueue.put(1)
            
            if len(allKlines) >= 20000:
                self.saveData(symbol, interval, startStr, endStr, allKlines)
                allKlines = []
        
        if allKlines:
            self.saveData(symbol, interval, startStr, endStr, allKlines)
        
        

    def saveData(self, symbol, interval, startStr, endStr, data):
        df = self.createDataFrame(data)
        symbolDir = os.path.join(self.outputDir, symbol)
        os.makedirs(symbolDir, exist_ok=True)
        startTime = datetime.datetime.strptime(startStr, "%d/%m/%Y").strftime("%Y%m%d")
        endTime = datetime.datetime.strptime(endStr, "%d/%m/%Y").strftime("%Y%m%d")
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f'{symbol}_{interval}_{startTime}_to_{endTime}_{timestamp}.csv'
        
        filepath = os.path.join(symbolDir, filename)
        df.to_csv(filepath, sep=';', index=False)

    def createDataFrame(self, data):
        columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                   'Number of Trades', 'Taker Buy Base Asset Volume']
        all_columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
                       'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
                       'Taker Buy Quote Asset Volume', 'Ignore']
        df = pd.DataFrame(data, columns=all_columns)
        df = df[columns]  # Keep only necessary columns
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
        return df

    def run(self):
        dateRanges = self.getDateRanges(self.startDate, self.endDate)
        manager = Manager()
        progressQueue = manager.Queue()
        tasks = [(symbol, self.interval, start, end, progressQueue) for symbol in self.symbols for start, end in dateRanges]

        totalTasks = len(tasks)
        
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(self.fetchData, tasks), total=totalTasks):
                pass

    def getDateRanges(self, startDate, endDate, daysPerChunk=30):
        start = datetime.datetime.strptime(startDate, "%d/%m/%Y")
        end = datetime.datetime.strptime(endDate, "%d/%m/%Y")
        current = start
        dateRanges = []
        while current < end:
            nextDate = current + datetime.timedelta(days=daysPerChunk)
            if nextDate > end:
                nextDate = end
            dateRanges.append((current.strftime("%d/%m/%Y"), nextDate.strftime("%d/%m/%Y")))
            current = nextDate
        return dateRanges

def loadConfig(configFile):
    with open(configFile, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config = loadConfig('data/config.yml')
    
    symbols = config['symbols']
    interval = config['interval']
    startDate = config['start_date']
    endDate = config['end_date']
    
    if endDate == 'now':
        endDate = datetime.datetime.now().strftime("%d/%m/%Y")
    
    loader = BinanceDataDownloader(symbols, interval, startDate, endDate)
    loader.run()