"""Typed boundary around the Vast.ai SDK."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class VastOffer:
    id: int
    gpu_name: str
    dph_total: float
    dlperf_per_dphtotal: float
    reliability: float
    geolocation: str
    cpu_ram_mb: int
    gpu_ram_mb: int
    num_gpus: int
    cuda_max_good: float
    disk_space_gb: float
    direct_port_count: int
    end_date: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VastOffer:
        try:
            return cls(
                id=int(value["id"]),
                gpu_name=str(value["gpu_name"]),
                dph_total=float(value["dph_total"]),
                dlperf_per_dphtotal=float(value["dlperf_per_dphtotal"]),
                reliability=float(value["reliability"]),
                geolocation=str(value["geolocation"]),
                cpu_ram_mb=int(value["cpu_ram"]),
                gpu_ram_mb=int(value["gpu_ram"]),
                num_gpus=int(value["num_gpus"]),
                cuda_max_good=float(value["cuda_max_good"]),
                disk_space_gb=float(value["disk_space"]),
                direct_port_count=int(value["direct_port_count"]),
                end_date=float(value["end_date"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Vast offer response: {value!r}") from exc


class VastApi(Protocol):
    def search_offers(
        self, query: str, *, order: str, limit: int, storage: float
    ) -> list[Mapping[str, Any]]: ...

    def create_instance(self, offer_id: int, **kwargs: Any) -> int: ...

    def show_instance(self, instance_id: int) -> Mapping[str, Any] | None: ...

    def show_instances(self) -> list[Mapping[str, Any]]: ...

    def destroy_instance(self, instance_id: int) -> None: ...


class VastSdkClient:
    """Validate the response shapes used from the lock-pinned Vast SDK."""

    def __init__(self, api_key: str) -> None:
        from vastai import VastAI

        self._client = VastAI(api_key=api_key)

    def search_offers(
        self, query: str, *, order: str, limit: int, storage: float
    ) -> list[Mapping[str, Any]]:
        value = self._client.search_offers(
            query=query,
            order=order,
            limit=limit,
            storage=storage,
        )
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) or "id" not in item for item in value
        ):
            raise RuntimeError(f"unexpected Vast offer response: {value!r}")
        return value

    def create_instance(self, offer_id: int, **kwargs: Any) -> int:
        value = self._client.create_instance(id=offer_id, **kwargs)
        try:
            return int(value["new_contract"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Vast did not return a new instance id: {value!r}"
            ) from exc

    def show_instance(self, instance_id: int) -> Mapping[str, Any] | None:
        value = self._client.show_instance(id=instance_id)
        if value is None:
            return None
        if not isinstance(value, Mapping) or "id" not in value:
            raise RuntimeError(f"unexpected Vast instance response: {value!r}")
        return value

    def show_instances(self) -> list[Mapping[str, Any]]:
        value = self._client.show_instances()
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) or "id" not in item for item in value
        ):
            raise RuntimeError(f"unexpected Vast instances response: {value!r}")
        return value

    def destroy_instance(self, instance_id: int) -> None:
        value = self._client.destroy_instance(id=instance_id)
        if not isinstance(value, Mapping) or value.get("success") is not True:
            raise RuntimeError("Vast did not confirm instance deletion")


def exception_http_status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if type(status) is int else None
