import pkgutil, importlib, re
__path__ = pkgutil.extend_path(__path__, __name__)
for _,module,_ in pkgutil.walk_packages(path=__path__, prefix=__name__+'.'):
    name = re.search("[^.]*$", module).group(0)
    globals()[name] = getattr(importlib.import_module(module), name)
