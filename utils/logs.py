
import logging
import os
import sys
from logging.handlers import SysLogHandler
from ansi2html import Ansi2HTMLConverter
from singleton import Singleton

class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        logFmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logFmt, style="{")
        return formatter.format(record)

class HTMLFormatter(logging.Formatter):
    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.converter = Ansi2HTMLConverter()

    def format(self, record):
        # Format the record using ColorFormatter
        color_formatter = ColorFormatter(self.fmt)
        message = color_formatter.format(record)
        # Convert ANSI sequences to HTML
        return self.converter.convert(message, full=False)

def getLogger(logName):
    return Logger().getLogger(logName)

class Logger(metaclass=Singleton):
    def __init__(self):
        self.syslog = False
        self.filelog = False
        self.logPath = '/var/log/actkv2'
        self.debug = False

        try:
            if not os.path.exists(self.logPath):
                os.makedirs(self.logPath)
            self.filelog = True
        except PermissionError:
            if os.path.exists('/dev/log'):
                self.syslog = True

        self.rootLogger = logging.getLogger()
        logLevel = logging.DEBUG if self.debug else logging.INFO
        self.rootLogger.setLevel(logLevel)

        # Set up color formatter for console output
        colorFormatter = ColorFormatter('{asctime} - {levelname:>3.3} - {name:>16.16} - {message}')
        streamHandler = logging.StreamHandler(sys.stdout)
        streamHandler.setFormatter(colorFormatter)
        self.rootLogger.addHandler(streamHandler)

        # HTML formatter for file/syslog
        htmlFormatter = HTMLFormatter('{asctime} - {levelname:>3.3} - {name:>16.16} - {message}')
        if self.filelog:
            fileHandler = logging.FileHandler(os.path.join(self.logPath, 'kernel.log'))
            fileHandler.setFormatter(htmlFormatter)
            self.rootLogger.addHandler(fileHandler)
        elif self.syslog:
            syslogHandler = SysLogHandler('/dev/log')
            syslogHandler.setFormatter(htmlFormatter)
            self.rootLogger.addHandler(syslogHandler)
        else:
            print('No persistent log handler; filelog or syslog not available!')

    def setRootLogLevel(self, level):
        self.rootLogger.setLevel(level)

    def getLogger(self, name):
        return logging.getLogger(name)
