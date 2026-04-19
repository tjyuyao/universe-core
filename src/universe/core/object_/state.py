# 定义哪些类型可以作为 State
STATE_TYPES = (str, int, float, bool, list, dict, type(None))


class StateType(type):
    """元类：让 State 类可以灵活地 isinstance 检查"""
    
    def __instancecheck__(cls, instance):
        # 如果是 State 的子类实例
        if super().__instancecheck__(instance):
            return True
        # 如果是 primitive type
        if isinstance(instance, STATE_TYPES):
            return True
        return False
    
    def __subclasscheck__(cls, subclass):
        if super().__subclasscheck__(subclass):
            return True
        # 允许 primitive types 被视为 State 的子类
        if issubclass(subclass, STATE_TYPES):
            return True
        return False


class State(metaclass=StateType):
    """State 基类
    
    可以被子类化创建自定义状态类，
    也可以直接使用 primitive types (str, int, float, list, dict, bool, None)
    """
    pass