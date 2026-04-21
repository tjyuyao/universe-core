from typing import Any, TypeVar, get_args, get_origin


class GenericsMeta(type):
    """从参数化基类的 __orig_bases__ 中提取泛型实参并存入 `_generics`。"""

    _generics: tuple[Any, ...]

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> "GenericsMeta":
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Always reset; never inherit the parent's tuple via MRO.
        cls._generics = ()

        # Only inspect *this* class's own parameterized bases.
        orig_bases = cls.__dict__.get("__orig_bases__", ())
        for origin in orig_bases:
            origin_type = get_origin(origin)
            if origin_type in bases:
                args = get_args(origin)
                # Skip the generic-definition case like Generic[T], Base[T]
                if args and not all(isinstance(a, TypeVar) for a in args):
                    cls._generics = args
                    break

        return cls
