import shutil
from pathlib import Path

class ZipDownloader:

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
    
    def zip_input_dir(self):
        return shutil.make_archive(base_name="converted_files", format='zip', root_dir=self.input_dir)
