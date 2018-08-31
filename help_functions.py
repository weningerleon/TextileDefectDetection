from pathlib import PosixPath, WindowsPath
import platform

def getStem(name):
    target_name = None

    if platform.system() == 'Windows':
        target_name = WindowsPath(name).stem
    elif platform.system() == 'Linux':
        target_name = PosixPath(name).stem
    else:
        print("Fancy Operation System: " + str(platform.system()) + ", unfortunately not supported here")
        raise Exception('Unknown OS: ' + platform.system())

    return target_name
