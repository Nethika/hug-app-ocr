import io
import os
import time
import math
import sys
#from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

import numpy as np

import json
import string




class Watcher:
    DIRECTORY_TO_WATCH = "model_start"
    

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.DIRECTORY_TO_WATCH, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except:
            self.observer.stop()
            print("Watcher for textHandler Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

        if event.is_directory:
            return None

        elif event.event_type == 'created' and event.src_path[-5:] == ".json":
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)


        elif event.event_type == 'modified':
            # Taken any action here when a file is modified.
            print("Received modified event - %s." % event.src_path)

        elif event.event_type == 'deleted':
            # Taken any action here when a file is modified.
            print("Recieved deleted event - %s." % event.src_path)

        else:
            print("Event recieved: %s." % event.event_type)


if __name__ == '__main__':
    w = Watcher()
    w.run()
