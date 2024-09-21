# Ensure unicity of object
# To be used like this
# class MyClass(BaseClass, metaclass=Singleton):
#    pass
# <NOTE> Make a difference between Singleton and RigidSingleton.
# Singleton may result in different Objects if instanciated with different args
# RigidSingleton is one an only one Object independenlty of passed args


kwd_mark = object()


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        key = (cls,) + args + (kwd_mark,) + tuple(sorted(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__call__(*args, **kwargs)
            # print(cls._instances)
        return cls._instances[key]


class RigidSingleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance
