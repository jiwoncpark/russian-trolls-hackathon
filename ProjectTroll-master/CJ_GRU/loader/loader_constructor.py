def get_loader(obj):
    # finds the loader based on the net name
    mod = __import__('loader.'+obj.loader, fromlist=[obj.loader])
    func = getattr(mod, obj.loader)
    return func(obj)

