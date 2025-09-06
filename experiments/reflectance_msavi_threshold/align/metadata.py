"""
DJI P4 Metadata Management
"""

import math
import os
# from pathlib import Path
from typing import Union
from datetime import datetime, timedelta

import exiftool
import pytz

class Metadata(object):
    def __init__(self, filename: str, exiftool_path=None, exiftool_obj=None):
        if exiftool_obj is not None:
            self.exif = exiftool_obj.get_metadata(filename)[0]
            return
        if exiftool_path is not None:
            self.exiftoolPath = exiftool_path
        elif os.environ.get('exiftoolpath') is not None:
            self.exiftoolPath = os.path.normpath(os.environ.get('exiftoolpath'))
        else:
            self.exiftoolPath = None
        if not os.path.isfile(filename):
            raise IOError("Input path is not a file")
        with exiftool.ExifToolHelper() as exift:
            self.exif = exift.get_metadata(filename)[0]

    def get_all(self):
        return self.exif
    
    def get_item(self, item, index=None):
        """ Get metadata item by Namespace:Parameter"""
        val = None
        try:
            assert len(self.exif) > 0
            val = self.exif[item]
            if index is not None:
                try:
                    if isinstance(val, unicode):
                        val = val.encode('ascii', 'ignore')
                except NameError:
                    # throws on python 3 where unicode is undefined
                    pass
                if isinstance(val, str) and len(val.split(',')) > 1:
                    val = val.split(',')
                val = val[index]
        except KeyError:
            pass
        except IndexError:
            print("Item {0} is length {1}, index {2} is outside this range.".format(
                item,
                len(self.exif[item]),
                index))
        return val

    def size(self, item):
        """get the size (length) of a metadata item"""
        val = self.get_item(item)
        try:
            if isinstance(val, unicode):
                val = val.encode('ascii', 'ignore')
        except NameError:
            # throws on python 3 where unicode is undefined
            pass
        if isinstance(val, str) and len(val.split(',')) > 1:
            val = val.split(',')
        if val is not None:
            return len(val)
        else:
            return 0
        
    def print_all(self):
        for item in self.get_all():
            print(f"{item}: {self.get_item(item)}")

    def position(self):
        """get the WGS-84 latitude, longitude tuple as signed decimal degrees"""
        lat = self.get_item('EXIF:GPSLatitude')
        latref = self.get_item('EXIF:GPSLatitudeRef')
        if latref == 'S':
            lat *= -1.0
        lon = self.get_item('EXIF:GPSLongitude')
        lonref = self.get_item('EXIF:GPSLongitudeRef')
        if lonref == 'W':
            lon *= -1.0
        alt = self.get_item('EXIF:GPSAltitude')
        return lat, lon, alt

    def utc_time(self):
        """ Get the timezone-aware datetime of the image capture """
        str_time = self.get_item('EXIF:DateTimeOriginal')
        if str_time:
            utc_time = datetime.strptime(str_time, "%Y:%m:%d %H:%M:%S")
            subsec = float(f"0.{self.get_item('EXIF:SubSecTime')}") if self.get_item('EXIF:SubSecTime') else 0.0
            negative = 1.0
            if subsec < 0:
                negative = -1.0
                subsec *= -1.0
            subsec *= negative
            ms = subsec * 1e3
            utc_time += timedelta(milliseconds=ms)
            timezone = pytz.timezone('UTC')
            utc_time = timezone.localize(utc_time)
        else:
            utc_time = None
        return utc_time
    
    
        
