def get_net(obj):
    # finds the loader based on the net name
    mod = __import__('net.'+obj.net, fromlist=[obj.net])
    func = getattr(mod, obj.net)
    return func(obj)

