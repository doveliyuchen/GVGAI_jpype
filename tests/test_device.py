import os

def test_deviceEnv():
    try:
        is_gpu = os.path.exists("/usr/local/cuda/version.txt")
        if is_gpu:
            from tensorflow.python.client import device_lib
            devices = "".join([device.device_type for device in device_lib.list_local_devices()])
            flag = devices.find("GPU")
            if flag == -1:
                assert False, "No GPU device available"
    except Exception as er:
        assert False, er
