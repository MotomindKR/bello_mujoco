"""Lightweight dataclass configuration decoding."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints


def dataclass_from_dict(
    cls,
    data: dict[str, Any],
    *,
    strict: bool = False,
    require_all_fields: bool = False,
    path: str | None = None,
):
    if not isinstance(data, dict):
        raise TypeError(f"{path or cls.__name__} must be a mapping")
    field_names = {field.name for field in fields(cls)}
    if strict:
        unknown = sorted(set(data) - field_names)
        if unknown:
            location = path or cls.__name__
            names = ", ".join(unknown)
            raise ValueError(f"unknown config key(s) for {location}: {names}")
    if require_all_fields:
        missing = sorted(field_names - set(data))
        if missing:
            location = path or cls.__name__
            names = ", ".join(missing)
            raise ValueError(f"missing required config key(s) for {location}: {names}")
    kwargs = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        if field.name not in data:
            continue
        field_path = field.name if path is None else f"{path}.{field.name}"
        kwargs[field.name] = coerce_value(
            type_hints.get(field.name, field.type),
            data[field.name],
            strict=strict,
            require_all_fields=require_all_fields,
            path=field_path,
        )
    return cls(**kwargs)


def coerce_value(
    annotation,
    value,
    *,
    strict: bool = False,
    require_all_fields: bool = False,
    path: str | None = None,
):
    if value is None:
        return None
    if annotation is Path or annotation == "Path":
        return Path(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return dataclass_from_dict(
            annotation,
            value,
            strict=strict,
            require_all_fields=require_all_fields,
            path=path,
        )
    if "Path" in str(annotation) and "None" in str(annotation):
        return Path(value)
    origin = get_origin(annotation)
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return {
            coerce_value(
                key_type,
                item_key,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            ): coerce_value(
                value_type,
                item_value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=f"{path}.{item_key}" if path is not None else str(item_key),
            )
            for item_key, item_value in value.items()
        }
    if origin is tuple:
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                coerce_value(
                    args[0],
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item in value
            )
        if args:
            if len(value) != len(args):
                location = path or "tuple"
                raise ValueError(
                    f"{location} must contain exactly {len(args)} values, got "
                    f"{len(value)}"
                )
            return tuple(
                coerce_value(
                    item_type,
                    item,
                    strict=strict,
                    require_all_fields=require_all_fields,
                    path=path,
                )
                for item_type, item in zip(args, value, strict=False)
            )
        return tuple(value)
    if origin is not None and type(None) in get_args(annotation):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none:
            return coerce_value(
                non_none[0],
                value,
                strict=strict,
                require_all_fields=require_all_fields,
                path=path,
            )
    if str(annotation).startswith("tuple["):
        return tuple(value)
    return value
