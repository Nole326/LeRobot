"""Lossless actor RGB storage; diagnostics stay outside the actor dataset."""
import hashlib
import os
import subprocess
from pathlib import Path

FFMPEG = os.environ.get('PICKORANGE_FFMPEG', 'ffmpeg')


class RGBVideo:
    def __init__(self,path):
        self.path=Path(path)
        assert not self.path.exists()
        self.frames=0
        self.decoder=None
        self.digest=hashlib.sha256()
        self.process=subprocess.Popen([str(FFMPEG),'-v','error','-n','-f','rawvideo',
            '-pix_fmt','rgb24','-s','640x480','-r','30','-i','pipe:0',
            '-c:v','libx264rgb','-crf','0','-preset','veryfast','-threads','2',str(self.path)],stdin=subprocess.PIPE)

    def write(self,rgb):
        assert rgb.shape==(480,640,3) and str(rgb.dtype)=='uint8'
        raw=rgb.tobytes()
        self.process.stdin.write(raw)
        self.digest.update(raw)
        self.frames+=1

    def close(self):
        self.process.stdin.close()
        assert self.process.wait()==0, 'RGB encoder failed'
        reader=self.decoder=subprocess.Popen([str(FFMPEG),'-v','error','-threads','2','-i',str(self.path),
            '-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)
        actual=hashlib.sha256(); total=0
        while raw:=reader.stdout.read(1024*1024):
            actual.update(raw); total+=len(raw)
        reader.stdout.close()
        assert reader.wait()==0
        assert total==self.frames*480*640*3 and actual.digest()==self.digest.digest()
        return dict(frames=self.frames,fps=30,width=640,height=480,pixel_format='rgb24',
            codec='libx264rgb',lossless=True,raw_rgb_sha256=actual.hexdigest(),
            decoded_exact=True,bytes=self.path.stat().st_size)

    def abort(self):
        """Release encoder/decoder processes after partial or failed capture."""
        for process in (self.process, self.decoder):
            if process is None:
                continue
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe is not None and not pipe.closed:
                    try:
                        pipe.close()
                    except OSError:
                        pass  # A terminated encoder may already have closed its pipe.
