
import os
import re
import shutil
import tempfile
from urllib.parse import urlparse, urljoin

import requests

from vot import ToolkitException

class NetworkException(ToolkitException):
    pass

def get_base_url(url):
    return url.rsplit('/', 1)[0]
    
def is_absolute_url(url):
    return bool(urlparse(url).netloc)

def join_url(url_base, url_path):
    if is_absolute_url(url_path):
        return url_path
    return urljoin(url_base, url_path)

def get_url_from_gdrive_confirmation(contents):
    url = ''
    for line in contents.splitlines():
        m = re.search(r'href="(\/uc\?export=download[^"]+)', line)
        if m:
            url = 'https://docs.google.com' + m.groups()[0]
            url = url.replace('&amp;', '&')
            return url
        m = re.search('confirm=([^;&]+)', line)
        if m:
            confirm = m.groups()[0]
            url = re.sub(r'confirm=([^;&]+)', r'confirm='+confirm, url)
            return url
        m = re.search(r'"downloadUrl":"([^"]+)', line)
        if m:
            url = m.groups()[0]
            url = url.replace('\\u003d', '=')
            url = url.replace('\\u0026', '&')
            return url


def is_google_drive_url(url):
    m = re.match(r'^https?://drive.google.com/uc\?id=.*$', url)
    return m is not None

def download_json(url):
    try:
        return requests.get(url).json()
    except requests.exceptions.RequestException as e:
        raise NetworkException("Unable to read JSON file {}".format(e))


def download(url, output, callback=None, chunk_size=1024*32):
    with requests.session() as sess:

        is_gdrive = is_google_drive_url(url)
        
        while True:
            res = sess.get(url, stream=True)
            
            if not res.status_code == 200:
                raise NetworkException("File not available")
            
            if 'Content-Disposition' in res.headers:
                # This is the file
                break
            if not is_gdrive:
                break

            # Need to redirect with confiramtion
            gurl = get_url_from_gdrive_confirmation(res.text)

            if gurl is None:
                raise NetworkException("Permission denied for {}".format(gurl))
            url = gurl

        if output is None:
            if is_gdrive:
                m = re.search('filename="(.*)"',
                            res.headers['Content-Disposition'])
                output = m.groups()[0]
            else:
                output = os.path.basename(url)

        output_is_path = isinstance(output, str)

        if output_is_path:
            tmp_file = tempfile.mktemp()
            filehandle = open(tmp_file, 'wb')
        else:
            tmp_file = None
            filehandle = output

        try:
            total = res.headers.get('Content-Length')

            if total is not None:
                total = int(total)

            for chunk in res.iter_content(chunk_size=chunk_size):
                filehandle.write(chunk)
                if callback:
                    callback(len(chunk), total)
            if tmp_file:
                filehandle.close()
                shutil.copy(tmp_file, output)
        except IOError:
            raise NetworkException("Error when downloading file")
        finally:
            try:
                if tmp_file:
                    os.remove(tmp_file)
            except OSError:
                pass

        return output

def download_uncompress(url, path):
    from vot.utilities import extract_files
    _, ext = os.path.splitext(urlparse(url).path)
    tmp_file = tempfile.mktemp(suffix=ext)
    try:
        download(url, tmp_file)
        extract_files(tmp_file, path)
    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)
        