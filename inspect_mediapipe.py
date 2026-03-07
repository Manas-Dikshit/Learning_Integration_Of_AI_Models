import mediapipe, pkgutil, os
print('mediapipe file:', getattr(mediapipe, '__file__', None))
print('mediapipe path:', getattr(mediapipe, '__path__', None))
try:
    modules = [m.name for m in pkgutil.iter_modules(mediapipe.__path__)]
except Exception as e:
    modules = f'error: {e}'
print('mediapipe modules:', modules)
try:
    pkg_dir = os.path.dirname(mediapipe.__file__)
    print('dir listing:', os.listdir(pkg_dir))
except Exception as e:
    print('dir listing error:', e)
